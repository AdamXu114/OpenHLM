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
integrates the pose itself from ``[30:33]``). The optional initial seed is read from these debug fields.

The conversion manifest identifies base_link trajectories. For older unmarked datasets,
--assume-base-link is an explicit assertion about the original poses. A dataset marked
with a different anchor requires a kinematic conversion before replay.

Use --anchor-from-dataset to seed the integrated pose with the first selected episode's
debug position and yaw, or use the deployment default (0, 0, 0.604 m) or --anchor-pos X Y Z. A single publisher keeps its
stream ID, pose and monotonic frame indices throughout the invocation. For independently
seeded episodes use separate runs. Each default two-frame window includes the previous
and current frames. --infer-pause-ms inserts a pause at each open-loop chunk boundary.

Example offline conversion check::

    python jaka_dataset_action_publisher.py --dataset-root <base_dataset> --episodes 0 \
        --anchor-from-dataset --dry-run

For local simulation, set --motion-address 127.0.0.1 and the simulator's receive port.
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

from jaka_tabletop_env import (
    DEFAULT_INITIAL_ANCHOR_POS, JakaTabletopEnv, PERM_POLICY_TO_SIM, _euler_xyz_to_quat_wxyz,
    validate_base_replay_dataset,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
# Search the training and deployment machines for the converted jaka_pick dataset.
_DATASET_LEAF = ("jaka_pick",)
DATASET_ROOT_CANDIDATES = (
    os.path.join(_REPO_ROOT, "record_teleop", "record_data", "lerobot_train", *_DATASET_LEAF),
    os.path.join(os.path.expanduser("~"), "lerobot_data", *_DATASET_LEAF),
    os.path.join(_REPO_ROOT, "data", *_DATASET_LEAF),
    os.path.join(_REPO_ROOT, "data", "teleop_jaka_mf", "simple", "JakaTabletopPickTeleop-v0", "level-0"),
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
    if actions.ndim != 2 or not len(actions) or actions.shape[1] < ACTION_DIM:
        raise ValueError(f"{path}: expected a nonempty array of actions at least {ACTION_DIM} wide")
    return actions


def anchor_seed_from_dataset(actions: np.ndarray) -> tuple[tuple, tuple]:
    """The first frame's anchor world pose, as (pos, rpy) for the env constructor.

    Only the yaw is taken from the recorded quat: roll/pitch are absolute columns already,
    and the position's x/y is an arbitrary per-episode origin (only z is load-bearing,
    being the tracker's absolute ``root_z_mf``).
    """
    from scipy.spatial.transform import Rotation

    if actions.shape[1] < 40:
        raise ValueError("--anchor-from-dataset requires the recorded world position and quaternion in actions[33:40]")
    pos = tuple(float(v) for v in actions[0, 33:36])
    quat_xyzw = np.roll(actions[0, 36:40].astype(np.float64), -1)  # wxyz -> xyzw
    yaw = float(Rotation.from_quat(quat_xyzw).as_euler("xyz")[2])
    return pos, (float(actions[0, 27]), float(actions[0, 28]), yaw)


@dataclasses.dataclass
class Args:
    dataset_root: str = DEFAULT_DATASET_ROOT
    episodes: tuple[int, ...] = (0,)  # SPACE separated, e.g. --episodes 0 1 2; bare = all episodes
    max_frames: int = 0               # >0 = stop after this many frames overall
    repeat: int = 1                   # replays the episode sequence this many times

    control_hz: int = 30
    open_loop_horizon: int = 25       # steps per "chunk"; matches main.py
    infer_pause_ms: float = 80.0      # production's stall at each chunk boundary
                                      # (~70-120 ms measured); 0.0 = perfectly smooth

    motion_address: str = "*"
    motion_port: int = 28701          # the real deployment port
    num_frames_to_send: int = 2       # 2 = sliding window [i-1, i]

    # For a verified, unmarked base_link dataset only. A known waist dataset is rejected.
    assume_base_link: bool = False
    anchor_pos: tuple[float, float, float] = DEFAULT_INITIAL_ANCHOR_POS
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

    validate_base_replay_dataset(root, assume_base_link=args.assume_base_link)
    if args.open_loop_horizon < 1:
        raise ValueError("open_loop_horizon must be positive")
    lengths = episode_lengths(root)
    episodes = list(args.episodes) if args.episodes else sorted(lengths)
    for ep in episodes:
        if ep not in lengths:
            raise SystemExit(f"episode {ep} not in {root} (max {max(lengths)})")

    actions_cache = {ep: load_actions(root, ep) for ep in episodes}

    anchor_pos, anchor_rpy = args.anchor_pos, args.anchor_rpy
    seed_note = "specified initial anchor pose"
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
    t_start = time.monotonic()
    frames = {
        "frame_index": [], "joint_pos": [], "body_pos_w": [], "body_quat_w": [],
        "action_joint_policy": [],
    }
    next_report = args.control_hz

    try:
        for rep in range(max(args.repeat, 1)):
            for ep_index, ep in enumerate(episodes):
                actions = actions_cache[ep]
                print(f"\n[ep{ep}] {len(actions)} frames"
                      + (f"  (repeat {rep + 1}/{args.repeat})" if args.repeat > 1 else ""))

                if ep_index > 0 or rep > 0:
                    # Clears the 2-frame window (so the first step of the take does not
                    # publish a window mixing two takes) while frame_index and the anchor
                    # state keep advancing -- see the module docstring.
                    env.reset()

                # One "chunk" per open_loop_horizon steps, so the stall pattern matches
                # production's: infer ~80 ms, then publish 25 frames at 30 Hz.
                for t0 in range(0, len(actions), args.open_loop_horizon):
                    if args.infer_pause_ms > 0 and not (t0 == 0 and ep_index == 0 and rep == 0):
                        time.sleep(args.infer_pause_ms / 1000.0)
                    for t in range(t0, min(t0 + args.open_loop_horizon, len(actions))):
                        start = time.monotonic()
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
                        env.step(actions[t, :ACTION_DIM])
                        n_published += 1
                        if args.dump:
                            frames["frame_index"].append(n_published - 1)
                            frames["joint_pos"].append(pub_joint_pos)       # SIM order, as sent
                            frames["body_pos_w"].append(pub_body_pos)
                            frames["body_quat_w"].append(pub_body_quat)
                            frames["action_joint_policy"].append(actions[t][0:27].copy())
                        if n_published >= next_report:
                            next_report += args.control_hz
                            hz = n_published / max(time.monotonic() - t_start, 1e-9)
                            print(f"   frame {n_published:6d}   ep{ep} t={t:4d}   "
                                  f"mean {hz:5.1f} Hz   t+{time.monotonic() - t_start:6.1f}s")
                        if args.max_frames and n_published >= args.max_frames:
                            raise KeyboardInterrupt
                        elapsed = time.monotonic() - start
                        if elapsed < target_dt:
                            time.sleep(target_dt - elapsed)

    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        elapsed = time.monotonic() - t_start
        print(f"\npublished {n_published} steps in {elapsed:.1f} s "
              f"({n_published / max(elapsed, 1e-9):.1f} Hz mean)")
        if not args.dry_run:
            print(f"Last local frame index: {env._frame_index - 1}. "
                  "Check the receiver's tracking diagnostics for delivery.")
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
