"""Publish recorded dataset actions as if they were the VLA's output.

Drives ``JakaTabletopEnv.step()`` with the actions recorded in the training set and
publishes the resulting protocol-v1 frames on the real deployment port (28701), in real
time at the control frequency. No policy server, no GPU, no observations -- the robot side
sees exactly the byte stream it would see in production, so it can be tested on its own.

Why ``actions[t, :33]`` IS the VLA output: the trunk is 40-wide on disk, but only the
first 33 columns are action. ``[0:27]`` are the reference joints in POLICY order,
``[27]``/``[28]`` absolute roll/pitch, ``[29]`` ``yaw_vel``, ``[30:33]`` ``anchor_lin_vel``
in the anchor body frame -- which is precisely the 33-dim layout the policy emits and
``JakaTabletopEnv.step()`` consumes. ``[33:40]`` is ``anchor_pos_w`` / ``anchor_quat_w``,
a DEBUG copy of the recording's own world pose; feeding it would be meaningless (the env
integrates the pose itself from ``[30:33]``) and this script never touches it.

Three fidelity points that are easy to get wrong:

* **One env instance for the whole run.** ``JakaTabletopEnv.reset()`` deliberately does
  not reset ``_frame_index`` (the receiver keeps its dedup baseline across its own
  ``clear()``, so a frame index that goes backwards makes every later frame be dropped
  downstream) and has no public way to reseed the anchor. Constructing a new env per
  episode would restart ``frame_index`` at 0 and silently starve the receiver. Reusing one
  instance is also the physically right thing here: the episodes are consecutive takes of a
  single session -- episode k's first anchor pose continues episode k-1's last one (yaw
  91.3 -> 91.3, 93.5 -> 93.7, z 0.851 -> 0.840 across ep0/ep1/ep2) -- so carrying the anchor
  state across episodes reproduces the recording rather than stitching unrelated takes.
* **The chunk-boundary stall is reproduced on purpose.** Every ``--open-loop-horizon``
  steps, production spends ~70-120 ms inferring and publishes nothing at all; the receiver
  treats a >50 ms gap as a stall and re-anchors its timeline there. A perfectly smooth
  replay never exercises that path, so ``--infer-pause-ms`` (default 80) inserts the gap at
  each chunk boundary. Set it to 0 to test the no-stall regime instead.
* **Cadence is not a knob to slow down with.** The env integrates ``world_vel * 1/control_hz``
  per step, and the receiver re-anchors on any >50 ms gap, so stretching the frame interval
  to get slow motion would both change the published trajectory's speed and make every
  single frame look like a stall. Run it at the real frequency.

The first frame of each window is buffered but not published (2-frame sliding window, so N
steps put N-1 messages on the wire; frame k reaches the receiver inside step k+1's window).
``--dump`` records exactly what was handed to the socket, which is what to diff against
whatever the robot side logs when checking the wire contract.

Run on the robot machine (native, no container), with the receiver already subscribed::

    cd ~/Workspace/OpenHLM/src/GR00T-WholeBodyControl4OpenHLM/openpi-eval
    export PYTHONPATH=~/Workspace/OpenHLM/src/GR00T-WholeBodyControl4OpenHLM

    # safety valve: exercise the whole loop without binding/ publishing anything
    ~/Workspace/OpenHLM/src/openpi4OpenHLM/.venv/bin/python \
      jaka_dataset_action_publisher.py --episodes 0 --dry-run

    # one take, real port, production-like stall pattern
    ~/Workspace/OpenHLM/src/openpi4OpenHLM/.venv/bin/python \
      jaka_dataset_action_publisher.py --episodes 0

    # a static take (ep22 barely moves) if you want the robot to hold still first
    ~/Workspace/OpenHLM/src/openpi4OpenHLM/.venv/bin/python \
      jaka_dataset_action_publisher.py --episodes 22

``--episodes`` is a tyro tuple: SPACE separated (``--episodes 0 1 2``), not commas. Passing
the flag with no values (bare ``--episodes``) selects all 47. ``--repeat N`` replays the
sequence N times, but the anchor state carries over, so the wrap is a pose discontinuity --
use it only when that is acceptable.

Stopping: Ctrl+C closes the env cleanly. The published pose is a REFERENCE trajectory, so
nothing needs to be commanded back to a default pose, unlike main.py's robot path.

**Do not restart this script against a live receiver.** Each run starts `frame_index` at 0,
and the receiver keeps its dedup baseline across its own ``clear()``, so a second run is
silently dropped in full (the robot just stops responding, with nothing in the log). Either
cover the whole test in one invocation (``--episodes`` / ``--repeat`` / ``--max-frames 0``)
or restart the receiver in the same breath as the publisher.
"""

# flake8: noqa: E402
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import tyro

from jaka_tabletop_env import PERM_POLICY_TO_SIM, JakaTabletopEnv, _euler_xyz_to_quat_wxyz

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
# Same two layouts the replay test handles: the training machine keeps the dataset under
# data/teleop_jaka_mf/simple/..., the robot machine under data/simple/...
_DATASET_LEAF = ("simple", "JakaTabletopPickTeleop-v0", "level-0")
DATASET_ROOT_CANDIDATES = (
    os.path.join(_REPO_ROOT, "data", "teleop_jaka_mf", *_DATASET_LEAF),
    os.path.join(_REPO_ROOT, "data", *_DATASET_LEAF),
)

ACTION_DIM = 33  # what the policy emits and what step() accepts; [33:40] is DEBUG-only


def _detect_dataset_root() -> str:
    for cand in DATASET_ROOT_CANDIDATES:
        if os.path.isfile(os.path.join(cand, "meta", "info.json")):
            return cand
    return DATASET_ROOT_CANDIDATES[0]


DEFAULT_DATASET_ROOT = _detect_dataset_root()


def episode_lengths(root: str) -> dict:
    lengths = {}
    with open(os.path.join(root, "meta", "episodes.jsonl")) as fh:
        for line in fh:
            rec = json.loads(line)
            lengths[int(rec["episode_index"])] = int(rec["length"])
    return lengths


def load_actions(root: str, episode: int) -> np.ndarray:
    """Read only the action trunk for one episode -- no images, no state.

    Skipping ``head_image_left`` is not just an optimisation: those columns hold inline
    PNGs, and decoding them is the slowest part of a full read. This publisher feeds
    ``step()`` and nothing else, so nothing else is needed.
    """
    path = os.path.join(root, "data", "chunk-000", f"episode_{episode:06d}.parquet")
    table = pq.read_table(path, columns=["actions"])
    actions = np.array(table["actions"].to_pylist(), dtype=np.float32)
    if actions.shape[1] < ACTION_DIM:
        raise SystemExit(f"{path}: actions are {actions.shape[1]}-wide, expected >= {ACTION_DIM}")
    return actions[:, :ACTION_DIM]  # exactly the policy's output layout


def anchor_seed_from_dataset(actions: np.ndarray) -> tuple[tuple, tuple]:
    """The first frame's anchor world pose, as (pos, rpy) for the env constructor.

    Only the yaw is taken from the recorded quat: roll/pitch are absolute columns already,
    and the position's x/y is an arbitrary per-episode origin (only z is load-bearing,
    being the tracker's absolute ``root_z_mf``).
    """
    from scipy.spatial.transform import Rotation

    pos = tuple(float(v) for v in actions[0, 33:36])
    quat_xyzw = np.roll(actions[0, 36:40].astype(np.float64), -1)  # wxyz -> xyzw
    yaw = float(Rotation.from_quat(quat_xyzw).as_euler("xyz")[2])
    return pos, (float(actions[0, 27]), float(actions[0, 28]), yaw)


@dataclasses.dataclass
class Args:
    dataset_root: str = DEFAULT_DATASET_ROOT
    episodes: tuple[int, ...] = (0,)  # SPACE separated, e.g. --episodes 0 1 2; bare = all 47
    max_frames: int = 0               # >0 = stop after this many frames overall
    repeat: int = 1                   # replays the episode sequence this many times

    control_hz: int = 30
    open_loop_horizon: int = 25       # steps per "chunk"; matches main.py
    infer_pause_ms: float = 80.0      # production's stall at each chunk boundary
                                      # (~70-120 ms measured); 0.0 = perfectly smooth

    motion_address: str = "*"
    motion_port: int = 28701          # the real deployment port
    num_frames_to_send: int = 2       # 2 = sliding window [i-1, i]

    # Defaults are main.py's deployment defaults, so the published poses sit in the same
    # frame production uses. --anchor-from-dataset instead reproduces the recording's own
    # absolute origin (its yaw is ~88 deg at the start of ep0, not 0).
    anchor_pos: tuple[float, float, float] = (0.0, 0.0, 0.83)
    anchor_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)
    anchor_from_dataset: bool = False

    mock: bool = True                 # nothing publishes :28711/:28712 offline
    dry_run: bool = False             # build the env with publish=False: binds nothing
    dump: str = ""                    # npz path; records every frame handed to the socket


def main(args: Args):
    root = os.path.abspath(args.dataset_root)
    if not os.path.isfile(os.path.join(root, "meta", "info.json")):
        checked = "\n".join(f"    {c}" for c in DATASET_ROOT_CANDIDATES)
        raise SystemExit(
            f"dataset not found at: {root}\n"
            f"  (looked for meta/info.json -- an existing directory without one is still wrong)\n"
            f"auto-detected candidates that were tried:\n{checked}\n"
            f"pass --dataset-root <path to the level-0 dir> if it lives elsewhere"
        )

    lengths = episode_lengths(root)
    episodes = list(args.episodes) if args.episodes else sorted(lengths)
    for ep in episodes:
        if ep not in lengths:
            raise SystemExit(f"episode {ep} not in {root} (max {max(lengths)})")

    actions_cache = {ep: load_actions(root, ep) for ep in episodes}

    anchor_pos, anchor_rpy = args.anchor_pos, args.anchor_rpy
    seed_note = "main.py's deployment defaults"
    if args.anchor_from_dataset:
        anchor_pos, anchor_rpy = anchor_seed_from_dataset(actions_cache[episodes[0]])
        seed_note = f"ep{episodes[0]} frame 0 (recording's own origin)"

    print(f"dataset  : {root}")
    print(f"episodes : {episodes}  ({sum(len(actions_cache[e]) for e in episodes)} frames)")
    print(f"anchor   : pos {tuple(round(v, 3) for v in anchor_pos)}  "
          f"rpy {tuple(round(v, 3) for v in anchor_rpy)}  <- {seed_note}")
    print(f"pacing   : {args.control_hz} Hz, chunk {args.open_loop_horizon} steps, "
          f"stall {args.infer_pause_ms:.0f} ms at each boundary")
    print(f"publish  : {'DRY RUN (nothing bound)' if args.dry_run else f'tcp://{args.motion_address}:{args.motion_port}'}")

    env = JakaTabletopEnv(
        control_hz=args.control_hz,
        mock=args.mock,
        motion_zmq_address=args.motion_address,
        motion_zmq_port=args.motion_port,
        num_frames_to_send=args.num_frames_to_send,
        initial_anchor_pos=anchor_pos,
        initial_anchor_rpy=anchor_rpy,
        publish=not args.dry_run,
    )

    target_dt = 1.0 / args.control_hz
    n_published = 0
    t_start = time.time()
    frames = {
        "frame_index": [], "joint_pos": [], "body_pos_w": [], "body_quat_w": [],
        "action_joint_policy": [],
    }
    next_report = args.control_hz

    try:
        for rep in range(max(args.repeat, 1)):
            for ep in episodes:
                actions = actions_cache[ep]
                print(f"\n[ep{ep}] {len(actions)} frames"
                      + (f"  (repeat {rep + 1}/{args.repeat})" if args.repeat > 1 else ""))

                if ep != episodes[0] or rep > 0:
                    # Clears the 2-frame window (so the first step of the take does not
                    # publish a window mixing two takes) while frame_index and the anchor
                    # state keep advancing -- see the module docstring.
                    env.reset()

                # One "chunk" per open_loop_horizon steps, so the stall pattern matches
                # production's: infer ~80 ms, then publish 25 frames at 30 Hz.
                for t0 in range(0, len(actions), args.open_loop_horizon):
                    if args.infer_pause_ms > 0 and not (t0 == 0 and ep == episodes[0] and rep == 0):
                        time.sleep(args.infer_pause_ms / 1000.0)
                    for t in range(t0, min(t0 + args.open_loop_horizon, len(actions))):
                        start = time.time()
                        if args.dump:
                            # Snapshot the pose step() is ABOUT to publish, using the env's
                            # own helpers so the dump cannot drift from the wire. It has to
                            # be read before the call: _body_pos_w and _accumulated_yaw are
                            # advanced after publishing, so afterwards they already hold the
                            # next frame's pose.
                            pub_joint_pos = actions[t][0:27][PERM_POLICY_TO_SIM]
                            pub_body_pos = env._body_pos_w.copy()
                            pub_body_quat = _euler_xyz_to_quat_wxyz(
                                float(actions[t][27]), float(actions[t][28]), env._accumulated_yaw
                            )
                        env.step(actions[t])
                        n_published += 1
                        if args.dump:
                            frames["frame_index"].append(n_published - 1)
                            frames["joint_pos"].append(pub_joint_pos)       # SIM order, as sent
                            frames["body_pos_w"].append(pub_body_pos)
                            frames["body_quat_w"].append(pub_body_quat)
                            frames["action_joint_policy"].append(actions[t][0:27].copy())
                        if n_published >= next_report:
                            next_report += args.control_hz
                            hz = n_published / max(time.time() - t_start, 1e-9)
                            print(f"   frame {n_published:6d}   ep{ep} t={t:4d}   "
                                  f"mean {hz:5.1f} Hz   t+{time.time() - t_start:6.1f}s")
                        if args.max_frames and n_published >= args.max_frames:
                            raise KeyboardInterrupt
                        elapsed = time.time() - start
                        if elapsed < target_dt:
                            time.sleep(target_dt - elapsed)

    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        elapsed = time.time() - t_start
        print(f"\npublished {n_published} steps in {elapsed:.1f} s "
              f"({n_published / max(elapsed, 1e-9):.1f} Hz mean)")
        if not args.dry_run:
            print(f"frame_index advanced to {env._frame_index - 1}; "
                  f"the receiver must have seen each index once (N steps -> N-1 messages)")
        if args.dump and n_published:
            out = Path(args.dump)
            out.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                out,
                frame_index=np.asarray(frames["frame_index"], dtype=np.int64),
                joint_pos=np.asarray(frames["joint_pos"], dtype=np.float32),   # SIM order, as sent
                body_pos_w=np.asarray(frames["body_pos_w"], dtype=np.float32),
                body_quat_w=np.asarray(frames["body_quat_w"], dtype=np.float32),  # wxyz
                action_joint_policy=np.asarray(frames["action_joint_policy"], dtype=np.float32),
                dataset_root=np.array(root), episodes=np.asarray(episodes),
                control_hz=np.array(args.control_hz),
                infer_pause_ms=np.array(args.infer_pause_ms),
                anchor_pos=np.asarray(anchor_pos, dtype=np.float64),
                anchor_rpy=np.asarray(anchor_rpy, dtype=np.float64),
            )
            print(f"wrote {out} ({n_published} frames as handed to env.step)")
        env.close()
        print("env closed.")


if __name__ == "__main__":
    main(tyro.cli(Args))
