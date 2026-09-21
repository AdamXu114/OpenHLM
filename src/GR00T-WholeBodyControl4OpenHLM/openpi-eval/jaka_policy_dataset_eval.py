"""Grade the VLA policy offline against the recording it was trained on.

Reads episodes from the training set, feeds each chunk-start frame (state + head camera + prompt)
to the live policy server exactly the way ``main.py`` does, and compares the returned (50, 33)
action chunk against the actions recorded in the dataset. Nothing here touches the real robot:
no ``JakaTabletopEnv``, no ZMQ, no observation channels -- just dataset -> websocket -> numbers.

This isolates ONE question: given the same inputs the policy would get in production, how far are
its actions from the recorded ones? The env was already proven to publish recorded actions
faithfully (the open-loop replay test on the real robot), so a wire-correct env plus this
measurement splits "the chain is broken" from "the policy is off".

Reading the numbers -- the whole point of the flag set:

* ``--repeats N`` measures the policy against ITSELF. The server samples (``num_steps=10``), so
  the same observation returns a slightly different chunk every call. That self-consistency error
  is the **noise floor**. If pred-vs-GT is about the size of pred-vs-pred, the recording sits inside
  the policy's own sampling spread and there is no systematic defect to chase; if pred-vs-GT is
  much larger, the gap is real. This is the single most important control in the script.
* ``--prompts`` A/B-tests the prompt. The dataset's task string is literally ``"..."`` (a merge-script
  fallback) while ``main.py:64`` sends a full English sentence, so the policy is asked for something
  it never trained on. ``--prompts both`` scores the same frames under each.
* **Joint vs anchor split.** A policy that tracks the arm well but drifts in ``anchor_lin_vel``
  produces a robot that reaches correctly but stands in the wrong place. Joint error is reported
  per limb; the anchor channels separately.
* **Velocity scale (``R``).** The recording's ``anchor_lin_vel`` is inflated relative to its own
  ``anchor_pos_w`` (measured earlier: ~1.28x -- its divisor is the pico publish period, not 1/30).
  The policy can only have learned that inflated convention. So ``R_pred`` should land near
  ``R_gt``: if it lands near 1.0 instead, the policy regressed to the physically correct speed and
  a robot following it under-travels by ~22%; if it lands far from both, that is a real defect.
  ``R_pred == R_gt`` together with a large ``drift_*_vs_posw`` means the inconsistency is in the
  DATA, inherited faithfully -- not something the policy introduced.
* **Per-row curve.** Error is tracked vs position inside the chunk (row 0..49). Production executes
  the first ``--open-loop-horizon`` rows, so the curve says whether the open-loop window is a lever.

Integration uses ``jaka_tabletop_env``'s own quaternion helpers, so the pose comparison cannot drift
from the convention the robot actually executes. Both the predicted and the recorded arm are seeded
from the SAME recorded pose at the chunk start, isolating the two action streams from accumulated
state error: this is a per-chunk comparison, not an end-to-end rollout.

Run on the deployment machine with the policy server already up in another window::

    cd ~/Workspace/OpenHLM/src/GR00T-WholeBodyControl4OpenHLM/openpi-eval
    export PYTHONPATH=~/Workspace/OpenHLM/src/GR00T-WholeBodyControl4OpenHLM

    # smoke: one episode, deployment prompt, first 4 chunks
    ~/Workspace/OpenHLM/src/openpi4OpenHLM/.venv/bin/python jaka_policy_dataset_eval.py \
      --episodes 0 --max-chunks 4

    # the actual diagnosis: 5 episodes, both prompts, 3 samples each -> noise floor included
    ~/Workspace/OpenHLM/src/openpi4OpenHLM/.venv/bin/python jaka_policy_dataset_eval.py \
      --episodes 0 1 2 3 4 --prompts both --repeats 3 \
      --out-dir visualization/policy_eval --dump visualization/policy_eval/ep0-4.npz

    # dense sweep of one episode (every frame starts a chunk: ~25x the inference cost)
    ~/Workspace/OpenHLM/src/openpi4OpenHLM/.venv/bin/python jaka_policy_dataset_eval.py \
      --episodes 0 --stride 1 --dump visualization/policy_eval/ep0_dense.npz

``--episodes`` is a tyro tuple: SPACE separated, not commas. Bare ``--episodes`` = all 47.
Cost is ``chunks x prompts x repeats`` inferences at ~80 ms each, plus one PNG decode per chunk.

Connection behaviour, because it bites every first run: the server blocks its event loop while
inferring, and its FIRST inference after a restart also compiles the model -- far longer than the
~80 ms steady state. openpi's client cannot survive that: it gives up at a 10 s handshake deadline
and retries only on ``ConnectionRefusedError``, so a compiling server looks exactly like a dead one
(``TimeoutError`` on the handshake, or ``InvalidMessage`` when the connection is dropped). Retrying
does not help and can even prevent progress, because each attempt is killed before the compile can
finish. So this script opens ONE long-lived connection up front, with keepalive off and a
``--warmup-timeout-s`` deadline (default 300 s), and pays the compile there; only then does it start
the measured loop, where ``--connect-retries`` x ``--retry-delay-s`` ride out ordinary stalls. The
warmup prints a heartbeat every 15 s (``... still waiting on the first inference (45s)``) and gives
up at ``--warmup-timeout-s``, so "still compiling" and "wedged" are told apart in the run itself
rather than by staring at a blank terminal. Once the server has served a request, later runs against
the SAME process connect immediately -- leave it running rather than restarting it between rounds of
testing.
"""

# flake8: noqa: E402
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import dataclasses
import io
import json
import socket
import time
from pathlib import Path
from typing import Literal

import numpy as np
import pyarrow.parquet as pq
import tyro
import websockets.sync.client
from PIL import Image
from scipy.spatial.transform import Rotation
from openpi_client import image_tools
from openpi_client import msgpack_numpy
from openpi_client import websocket_client_policy

# The env's own helpers: the pose comparison must use the exact convention the robot executes,
# never a reimplementation of it. Both are importable from jaka_tabletop_env (the quat pair is
# re-exported there from sonic_g1_env), so PYTHONPATH must be set exactly as for main.py.
from jaka_tabletop_env import _euler_xyz_to_quat_wxyz, _quat_rotate_wxyz

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
_DATASET_LEAF = ("simple", "JakaTabletopPickTeleop-v0", "level-0")
DATASET_ROOT_CANDIDATES = (
    os.path.join(_REPO_ROOT, "data", "teleop_jaka_mf", *_DATASET_LEAF),
    os.path.join(_REPO_ROOT, "data", *_DATASET_LEAF),
)

DEPLOY_PROMPT = "Pick up the purple soft finger on the table and place it on the mouse pad."
ACTION_DIM = 33   # policy output width; the dataset's [33:40] is DEBUG-only ground truth
POSE_LO, POSE_HI = 33, 40
DT = 1.0 / 30.0

# Record-level keys that are indices, not metrics -- must not be averaged into the summary.
_ID_KEYS = {"episode", "t0", "n_rows"}


def _detect_dataset_root() -> str:
    for cand in DATASET_ROOT_CANDIDATES:
        if os.path.isfile(os.path.join(cand, "meta", "info.json")):
            return cand
    return DATASET_ROOT_CANDIDATES[0]


DEFAULT_DATASET_ROOT = _detect_dataset_root()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def episode_lengths(root: str) -> dict:
    lengths = {}
    with open(os.path.join(root, "meta", "episodes.jsonl")) as fh:
        for line in fh:
            rec = json.loads(line)
            lengths[int(rec["episode_index"])] = int(rec["length"])
    return lengths


def dataset_task(root: str) -> str:
    """The prompt the policy was actually trained with (literally ``"..."`` for this dataset)."""
    with open(os.path.join(root, "meta", "tasks.jsonl")) as fh:
        return json.loads(fh.readline())["task"]


def action_names(root: str) -> list:
    with open(os.path.join(root, "meta", "info.json")) as fh:
        return json.load(fh)["features"]["actions"]["names"]


class Episode:
    """One episode, with PNGs decoded on demand.

    The image column is read once (the bytes live inline in the parquet), but decoding is lazy:
    only chunk-start frames are ever looked at, which is 1/``stride`` of the rows and the biggest
    saving in the script. ``state``/``actions`` are small and loaded outright.
    """

    def __init__(self, root: str, index: int):
        path = os.path.join(root, "data", "chunk-000", f"episode_{index:06d}.parquet")
        table = pq.read_table(path, columns=["head_image_left", "state", "actions"])
        # struct<bytes, path>; this dataset has no videos/ dir, so bytes is the inline PNG.
        self._blobs = table["head_image_left"].combine_chunks().field("bytes")
        self.states = np.array(table["state"].to_pylist(), dtype=np.float32)      # (T, 30)
        self.actions = np.array(table["actions"].to_pylist(), dtype=np.float32)   # (T, 40)
        self.index = index
        self._cache = {}

    def __len__(self) -> int:
        return len(self.actions)

    def image(self, t: int) -> np.ndarray:
        """RGB uint8 -- Pillow, never cv2: cv2 returns BGR silently and the policy would see
        swapped channels with no error anywhere."""
        if t not in self._cache:
            png = self._blobs[t].as_py()
            self._cache[t] = np.asarray(
                Image.open(io.BytesIO(png)).convert("RGB"), dtype=np.uint8
            )
        return self._cache[t]


# ---------------------------------------------------------------------------
# Policy client
# ---------------------------------------------------------------------------

def listener_ready(host: str, port: int, attempts: int, delay: float) -> bool:
    """Wait (bounded) for something to accept TCP on host:port.

    Distinguishes "no server" from "server busy": with nothing listening, openpi's own
    ``_wait_for_server`` reconnect loop would spin forever and the script would hang with no
    explanation, so this fails loudly instead.
    """
    for i in range(max(attempts, 1)):
        try:
            with socket.create_connection((host, port), timeout=3.0):
                return True
        except OSError:
            if i == attempts - 1:
                return False
            print(f"    nothing listening on {host}:{port} yet "
                  f"[{i + 1}/{attempts}] -- is serve_policy.py running?")
            time.sleep(delay)
    return False


def recv_with_progress(conn, timeout_s: float, label: str, tick_s: float = 15.0):
    """``recv`` bounded by a deadline, printing a heartbeat while it waits.

    Without this the wait is a terminal that prints nothing at all -- indistinguishable from a wedged
    server, which is exactly the ambiguity that made the first deployment attempt unreadable. Ticking
    with a short ``recv(tick)`` is safe: a timed-out ``recv`` does NOT discard the message, because
    websockets pushes the frames it already read back onto the queue, so the next call still returns
    it (``websockets.sync.messages.Assembler.get`` -> ``reset_queue``).
    """
    t0 = time.time()
    while True:
        left = timeout_s - (time.time() - t0)
        if left <= 0:
            raise TimeoutError(f"no reply for {label} within {timeout_s:.0f}s")
        try:
            return conn.recv(min(tick_s, left))
        except TimeoutError:
            print(f"      ... still waiting on {label} ({time.time() - t0:.0f}s)", flush=True)


def warm_up(host: str, port: int, timeout_s: float) -> float:
    """Absorb the first-inference compile on ONE long-lived connection, and return its wall time.

    openpi's own client cannot do this: it gives up at a 10 s handshake deadline and retries only on
    ``ConnectionRefusedError``, so a server still compiling its first inference is indistinguishable
    from a dead one -- and retrying makes it worse, because each attempt is killed before the compile
    can finish. Connecting here with a generous deadline, keepalive off, and a dummy request lets the
    expensive first inference complete once, up front, on a socket nobody will abort mid-way. The
    policy is stateless, so this costs nothing but time; once it returns, every later call is fast.

    Each read is deadline-bounded and prints a heartbeat, so a stalled server is reported with the
    phase and the elapsed time instead of hanging the script with a blank terminal.
    """
    request = {
        "head_image_left": np.zeros((224, 224, 3), np.uint8),   # black: values are not used
        "state": np.zeros(30, np.float32),
        "prompt": DEPLOY_PROMPT,
    }
    t0 = time.time()
    with websockets.sync.client.connect(
        f"ws://{host}:{port}", compression=None, max_size=None,
        open_timeout=timeout_s, ping_interval=None,   # no ping: it must NOT be killed mid-compile
    ) as conn:
        print(f"      handshake ok ({time.time() - t0:.1f}s); reading server metadata", flush=True)
        recv_with_progress(conn, 30.0, "server metadata")   # sent immediately after the handshake
        print("      sending the dummy request -- the FIRST inference compiles the model, "
              "so this is the slow one", flush=True)
        conn.send(msgpack_numpy.Packer().pack(request))
        response = recv_with_progress(conn, timeout_s, "the first inference")
        if isinstance(response, str):                 # the server reports infer errors as text
            raise SystemExit(f"server raised during the warmup inference:\n    {response[:2000]}")
    return time.time() - t0


class PolicyClient:
    """Websocket policy with reconnect-and-WAIT, for transient failures (not the warmup).

    The server blocks its event loop during inference, so a call that lands mid-inference dies on a
    10 s handshake deadline or a 20 s ping timeout. Those are transport hiccups, not logic errors:
    the retry must WAIT for the loop to free up, since an immediate reconnect just gets its handshake
    dropped while the server is still busy. The one-off compile that makes the first inference slow
    is handled separately by ``warm_up``, so these retries only need to ride out short stalls.
    """

    def __init__(self, host: str, port: int, retries: int = 5, delay: float = 10.0):
        self._host, self._port = host, port
        self._retries, self._delay = max(retries, 0), delay
        self._client = None

    def _connect(self):
        if self._client is None:
            self._client = websocket_client_policy.WebsocketClientPolicy(self._host, self._port)
        return self._client

    def infer(self, request: dict) -> np.ndarray:
        last, t0 = None, time.time()
        for attempt in range(self._retries + 1):
            try:
                return np.asarray(self._connect().infer(request)["actions"], dtype=np.float32)
            except Exception as e:  # noqa: BLE001 - any transport failure is worth another wait
                last = e
                self._client = None
                if attempt < self._retries:
                    print(f"    attempt {attempt + 1}/{self._retries + 1} failed "
                          f"({type(e).__name__}) after {time.time() - t0:.0f}s; the server is "
                          f"probably still compiling its first inference -- "
                          f"waiting {self._delay:.0f}s")
                    time.sleep(self._delay)
        raise RuntimeError(
            f"policy server at {self._host}:{self._port} did not answer in "
            f"{self._retries + 1} attempts over {time.time() - t0:.0f}s "
            f"(last: {type(last).__name__}: {last}).\n"
            f"    If serve_policy.py is still loading or compiling, raise --connect-retries and/or "
            f"--retry-delay-s and try again; once the model has served one request, later runs "
            f"against the SAME server process connect immediately (leave it running)."
        ) from last

    def close(self):
        client, self._client = self._client, None
        if client is not None and hasattr(client, "close"):
            try:
                client.close()
            except Exception:  # noqa: BLE001 - teardown must never mask the result
                pass


def build_request(image: np.ndarray, state: np.ndarray, prompt: str) -> dict:
    """Same key set as ``main.py:233`` for jaka: bare keys, head image only, no wrist views.

    The wrist keys must NOT be sent -- JakaInputs synthesizes the masked views server-side, and an
    extra key changes the request's shape. ``resize_with_pad`` is a no-op here (the dataset PNGs are
    already 224x224) but is kept so the request stays byte-identical to production.
    """
    return {
        "head_image_left": image_tools.resize_with_pad(image, 224, 224),
        "state": np.asarray(state, dtype=np.float32),
        "prompt": prompt,
    }


# ---------------------------------------------------------------------------
# Integration -- the env's convention, via the env's helpers
# ---------------------------------------------------------------------------

def pose_seed(gt_rows: np.ndarray) -> tuple:
    """(position, yaw) at frame 0 of a chunk, from the recording's DEBUG pose columns.

    Only yaw is taken from the quat: roll/pitch are absolute columns already, and the position's
    x/y is an arbitrary per-episode origin (only z is absolute, being the tracker's ``root_z_mf``).
    """
    pos = np.asarray(gt_rows[0, POSE_LO:POSE_LO + 3], dtype=np.float64)
    quat_xyzw = np.roll(np.asarray(gt_rows[0, POSE_LO + 3:POSE_HI], dtype=np.float64), -1)  # wxyz->xyzw
    return pos, float(Rotation.from_quat(quat_xyzw).as_euler("xyz")[2])


def integrate_anchor(actions: np.ndarray, seed_pos: np.ndarray, seed_yaw: float) -> tuple:
    """World poses (and world-frame velocities) at frame starts, mirroring ``JakaTabletopEnv.step``.

    ``yaw[j] = seed + sum_{m<j} yaw_vel[m]*dt``; the quat is built from the yaw BEFORE row j's own
    yaw_vel is applied; the body-frame velocity is rotated by that quat; and ``pos[k] = seed +
    sum_{j<k} w[j]*dt`` -- frame k carries the pose at the START of its interval. Verified against
    the env's published frames to 2.5e-05 m over 47 episodes.
    """
    n = len(actions)
    yaw_vel = actions[:, 29].astype(np.float64)
    yaw = seed_yaw + np.concatenate([[0.0], np.cumsum(yaw_vel[:-1]) * DT])
    world = np.empty((n, 3), dtype=np.float64)
    for j in range(n):
        quat = _euler_xyz_to_quat_wxyz(float(actions[j, 27]), float(actions[j, 28]), float(yaw[j]))
        world[j] = _quat_rotate_wxyz(quat, actions[j, 30:33].astype(np.float64))
    pos = np.empty((n, 3), dtype=np.float64)
    pos[0] = seed_pos
    pos[1:] = seed_pos + np.cumsum(world[:-1] * DT, axis=0)
    return pos, world


def proj_ratio(world_vel: np.ndarray, dp_true: np.ndarray) -> float:
    """Displacement-weighted scale of an integrated velocity against a measured displacement.

    ``sum(w*dt . u_hat) / sum|dp|`` with ``u_hat`` the direction of the measured displacement: 1.0
    means the velocity integrates to exactly that displacement, 1.28 means it over-travels 28%.
    Displacement-weighted because per-frame ratios are dominated by near-stationary rows where both
    quantities are jitter (a trap measured earlier: ratios of 1e4 on a 1e-3 m/s frame).

    ``world_vel`` must already be rotated into the world frame -- the quantity compared against a
    world displacement. Note this rotates with the env's INTEGRATED-yaw orientation; the 1.28
    figure measured earlier used the recorded quat, so the two can differ slightly.
    """
    travelled = float(np.linalg.norm(dp_true))
    if travelled <= 1e-9:
        return float("nan")
    integrated = np.asarray(world_vel, np.float64).sum(axis=0) * DT
    return float(np.dot(integrated, dp_true / travelled) / travelled)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def joint_groups(names: list, n_joints: int = 27) -> dict:
    """Indices of the joint block grouped by limb, from the dataset's own names."""
    groups = {}
    for i, name in enumerate(names[:n_joints]):
        low = name.lower()
        if any(k in low for k in ("shoulder", "elbow", "wrist")):
            key = ("left" if low.startswith("left") else "right") + "_arm"
        elif any(k in low for k in ("hip", "knee", "ankle")):
            key = ("left" if low.startswith("left") else "right") + "_leg"
        elif "waist" in low:
            key = "waist"
        elif "neck" in low:
            key = "neck"
        else:
            key = "other"
        groups.setdefault(key, []).append(i)
    return groups


def mae(a, b) -> float:
    return float(np.mean(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))))


def pearson(a, b) -> float:
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    if a.size < 2 or a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def compare_actions(pred: np.ndarray, ref: np.ndarray, groups: dict, n_rows: int,
                    seed_pos: np.ndarray, seed_yaw: float) -> dict:
    """Every per-chunk scalar: ``pred`` graded against ``ref``.

    ``ref`` is either the dataset rows (40 wide: the action block is compared column-by-column and
    the DEBUG pose columns supply the physical reference displacement) or another prediction
    (33 wide: action-space metrics only). The recording's own world displacement over the graded
    rows is what makes the velocity-scale metrics possible; when it is unavailable those keys are
    simply absent, which the aggregator skips.

    ``n_rows`` is capped by the episode end, NOT by the open-loop horizon: the policy emits 50 rows
    and all of them are graded, with the executed/unexecuted boundary left visible in the row curve
    rather than silently truncating the comparison.
    """
    p = np.asarray(pred[:n_rows, :ACTION_DIM], np.float64)
    r_full = np.asarray(ref[:n_rows], np.float64)
    g = r_full[:, :ACTION_DIM]
    out = {"n_rows": int(n_rows)}
    # Trimmed together with the integration below, so the two always span the same rows.
    posw_delta = (r_full[-1, POSE_LO:POSE_LO + 3] - r_full[0, POSE_LO:POSE_LO + 3]
                  if r_full.shape[1] >= POSE_HI else None)

    for key, idx in groups.items():
        out[f"joint_{key}_mrad"] = mae(p[:, idx], g[:, idx]) * 1000.0
    out["joint_all_mrad"] = mae(p[:, :27], g[:, :27]) * 1000.0
    out["roll_mrad"] = mae(p[:, 27], g[:, 27]) * 1000.0
    out["pitch_mrad"] = mae(p[:, 28], g[:, 28]) * 1000.0
    out["yaw_vel"] = mae(p[:, 29], g[:, 29])

    # Anchor linear velocity: bias (systematic offset) and r (does it at least track the shape) are
    # kept apart, because a constant offset and a decorrelated signal need different fixes.
    for ax, name in enumerate("xyz"):
        out[f"linvel_{name}_mae"] = mae(p[:, 30 + ax], g[:, 30 + ax])
        out[f"linvel_{name}_bias"] = float(np.mean(p[:, 30 + ax] - g[:, 30 + ax]))
        out[f"linvel_{name}_r"] = pearson(p[:, 30 + ax], g[:, 30 + ax])
    out["linvel_mae"] = mae(p[:, 30:33], g[:, 30:33])
    out["linvel_r"] = pearson(p[:, 30:33], g[:, 30:33])

    # Pose consequence, both arms from the SAME seed so only the action streams differ.
    pos_p, world_p = integrate_anchor(p, seed_pos, seed_yaw)
    pos_g, world_g = integrate_anchor(g, seed_pos, seed_yaw)

    out["drift_pred_vs_gt_mm"] = float(np.linalg.norm(pos_p[-1] - pos_g[-1])) * 1000.0
    out["path_gt_mm"] = float(np.linalg.norm(pos_g[-1] - pos_g[0])) * 1000.0
    out["step_err_mm"] = float(np.mean(
        np.linalg.norm((pos_p[1:] - pos_p[:-1]) - (pos_g[1:] - pos_g[:-1]), axis=1))) * 1000.0

    if posw_delta is not None:
        posw_end = seed_pos + np.asarray(posw_delta, np.float64)
        out["drift_pred_vs_posw_mm"] = float(np.linalg.norm(pos_p[-1] - posw_end)) * 1000.0
        out["drift_gt_vs_posw_mm"] = float(np.linalg.norm(pos_g[-1] - posw_end)) * 1000.0
        out["path_posw_mm"] = float(np.linalg.norm(posw_delta)) * 1000.0
        out["R_gt"] = proj_ratio(world_g, posw_delta)
        out["R_pred"] = proj_ratio(world_p, posw_delta)

    out["row_joint_mrad"] = ((np.abs(p[:, :27] - g[:, :27]).mean(axis=1)) * 1000.0).tolist()
    out["row_linvel"] = (np.abs(p[:, 30:33] - g[:, 30:33]).mean(axis=1)).tolist()
    return out


def mean_of(records: list, key: str) -> float:
    vals = [r[key] for r in records if key in r and not np.isnan(r[key])]
    return float(np.mean(vals)) if vals else float("nan")


def q_of(records: list, key: str, q: float) -> float:
    vals = np.array([r[key] for r in records if key in r and not np.isnan(r[key])])
    return float(np.quantile(vals, q)) if vals.size else float("nan")


def pad_stack(chunks: list, width: int) -> np.ndarray:
    """Stack ragged chunks into one array, NaN-padding the short ones (episode-end chunks)."""
    rows = max(len(c) for c in chunks)
    out = np.full((len(chunks), rows, width), np.nan, dtype=np.float32)
    for i, c in enumerate(chunks):
        out[i, :len(c), :c.shape[1]] = c
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Args:
    dataset_root: str = DEFAULT_DATASET_ROOT
    episodes: tuple[int, ...] = (0,)   # SPACE separated; bare = all 47
    max_chunks: int = 0                # >0 = stop after this many chunks per episode
    stride: int = 25                   # chunk starts every N frames; 1 = every frame (~25x cost)
    open_loop_horizon: int = 25        # marks the executed/unexecuted boundary in the row curve

    remote_host: str = "127.0.0.1"
    remote_port: int = 8000
    connect_retries: int = 5           # attempts per inference, for transient stalls
    retry_delay_s: float = 10.0        # wait between those attempts
    warmup_timeout_s: float = 300.0    # deadline for the one-off first inference (model compile)

    prompt: str = DEPLOY_PROMPT        # what the deployed client sends
    prompts: Literal["deploy", "train", "both"] = "deploy"  # "both" scores each frame twice
    repeats: int = 1                   # >1 = also measure the pred-vs-pred noise floor

    out_dir: str = ""                  # summary.json goes here (default: print only)
    dump: str = ""                     # npz with every pred/GT chunk, re-analyzable without GPU


def resolve_prompts(args: Args, train_prompt: str) -> list:
    if args.prompts == "deploy":
        return [("deploy", args.prompt)]
    if args.prompts == "train":
        return [("train", train_prompt)]
    return [("deploy", args.prompt), ("train", train_prompt)]


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

    prompt_arms = resolve_prompts(args, dataset_task(root))
    groups = joint_groups(action_names(root))

    print(f"dataset   : {root}")
    print(f"episodes  : {episodes}  ({sum(lengths[e] for e in episodes)} frames, stride {args.stride})")
    print(f"server    : {args.remote_host}:{args.remote_port}")
    print(f"prompts   : " + " | ".join(f"{k}={t!r}" for k, t in prompt_arms))
    print(f"repeats   : {args.repeats}"
          + ("   (pred-vs-pred noise floor enabled)" if args.repeats > 1 else ""))
    print("joint grps: " + ", ".join(f"{k}({len(v)})" for k, v in sorted(groups.items())))

    # Short grace period only: a server that has not been started yet is a typo, not a warmup.
    if not listener_ready(args.remote_host, args.remote_port, 2, args.retry_delay_s):
        raise SystemExit(
            f"nothing is accepting TCP on {args.remote_host}:{args.remote_port}.\n"
            f"    Start the policy server first (doc/07 section 2), e.g.:\n"
            f"      cd ~/Workspace/OpenHLM/src/openpi4OpenHLM\n"
            f"      ./.venv/bin/python scripts/serve_policy.py --env JAKA --num-steps 10"
        )

    print(f"warmup    : one dummy inference on a long-lived connection (the first one compiles the "
          f"model; up to {args.warmup_timeout_s:.0f}s)")
    try:
        dt = warm_up(args.remote_host, args.remote_port, args.warmup_timeout_s)
    except Exception as e:  # noqa: BLE001 - report the phase that stalled, don't guess
        raise SystemExit(
            f"warmup inference did not complete: {type(e).__name__}: {e}\n"
            f"    Run jaka_policy_server_probe.py against the same server: it prints the elapsed time "
            f"of the handshake, the metadata read, and the first inference separately, which says "
            f"whether the server is slow, wedged, or raising."
        ) from e
    print(f"warmup    : {dt:.1f}s (later calls should be tens of ms)")

    client = PolicyClient(args.remote_host, args.remote_port,
                          args.connect_retries, args.retry_delay_s)
    records, rolled = [], {"pred": [], "gt": [], "ep": [], "t0": [], "prompt": []}
    t_start, n_calls = time.time(), 0

    try:
        for ep in episodes:
            episode = Episode(root, ep)
            starts = list(range(0, len(episode), max(args.stride, 1)))
            if args.max_chunks:
                starts = starts[:args.max_chunks]
            print(f"\n[ep{ep}] {len(episode)} frames, {len(starts)} chunks")

            for t0 in starts:
                gt = episode.actions[t0:t0 + 50]
                if len(gt) < 2:
                    continue
                # Observations come from the DATASET, never from a live env.
                image, state = episode.image(t0), episode.states[t0]
                seed_pos, seed_yaw = pose_seed(gt)

                for key, text in prompt_arms:
                    req = build_request(image, state, text)
                    preds = []
                    for _ in range(max(args.repeats, 1)):
                        chunk = client.infer(req)
                        # Fail on the shape here, with the actual numbers, rather than let a wrong
                        # action_dim surface later as a broadcasting error inside the metrics.
                        if chunk.ndim != 2 or chunk.shape[1] < ACTION_DIM:
                            raise SystemExit(
                                f"server returned actions of shape {chunk.shape}; expected "
                                f"(horizon, >= {ACTION_DIM}). The checkpoint's action_dim does not "
                                f"match this env's action layout."
                            )
                        preds.append(chunk)
                        n_calls += 1

                    n_rows = min(len(preds[0]), len(gt))
                    rec = compare_actions(preds[0], gt, groups, n_rows, seed_pos, seed_yaw)
                    rec.update({"episode": ep, "t0": t0, "prompt": key})
                    # Noise floor: the SAME observation resampled, scored with the same metrics.
                    for i in range(1, len(preds)):
                        self_rec = compare_actions(preds[i], preds[0], groups, n_rows,
                                                   seed_pos, seed_yaw)
                        rec[f"self_joint_mrad_{i}"] = self_rec["joint_all_mrad"]
                        rec[f"self_linvel_mae_{i}"] = self_rec["linvel_mae"]
                        rec[f"self_drift_mm_{i}"] = self_rec["drift_pred_vs_gt_mm"]
                    records.append(rec)

                    if args.dump:
                        rolled["pred"].append(preds[0][:n_rows, :ACTION_DIM])
                        rolled["gt"].append(gt[:n_rows])
                        rolled["ep"].append(ep)
                        rolled["t0"].append(t0)
                        rolled["prompt"].append(key)

                if len(records) % 5 == 0:
                    print(f"   {len(records):4d} chunks  ({n_calls} inferences, "
                          f"{time.time() - t_start:.0f}s)")

    except KeyboardInterrupt:
        print("\nstopped early -- reporting what completed.")
    finally:
        client.close()

    if not records:
        raise SystemExit("no chunks evaluated")

    # ---------------- report ----------------
    eps_seen = sorted({r["episode"] for r in records})
    n_rows_max = max(r["n_rows"] for r in records)
    print("\n" + "=" * 100)
    print(f"chunks {len(records)}   episodes {eps_seen}   inferences {n_calls}   "
          f"elapsed {time.time() - t_start:.0f}s")
    print(f"compared rows 1..{n_rows_max} per chunk (open-loop horizon = {args.open_loop_horizon}; "
          f"the row curve marks the boundary)")

    for key, text in prompt_arms:
        recs = [r for r in records if r["prompt"] == key]
        if not recs:
            continue
        print("\n" + "=" * 100)
        print(f"PROMPT ARM '{key}'  ({text!r})   chunks {len(recs)}")
        print("=" * 100)
        print("  joint error (mrad, mean over chunk rows)")
        for gkey in sorted(groups):
            print(f"    {gkey:12s} {mean_of(recs, f'joint_{gkey}_mrad'):8.1f}")
        print(f"    {'ALL joints':12s} {mean_of(recs, 'joint_all_mrad'):8.1f}")

        print("  anchor orientation / yaw rate (absolute error)")
        print(f"    roll      {mean_of(recs, 'roll_mrad'):8.1f} mrad")
        print(f"    pitch     {mean_of(recs, 'pitch_mrad'):8.1f} mrad")
        print(f"    yaw_vel   {mean_of(recs, 'yaw_vel'):8.3f} rad/s")

        print("  anchor lin_vel (m/s)  -- bias = pred-GT (systematic), r = shape agreement")
        for ax in "xyz":
            print(f"    {ax}: mae {mean_of(recs, f'linvel_{ax}_mae'):7.4f}   "
                  f"bias {mean_of(recs, f'linvel_{ax}_bias'):+7.4f}   "
                  f"r {mean_of(recs, f'linvel_{ax}_r'):6.3f}")
        print(f"    all: mae {mean_of(recs, 'linvel_mae'):7.4f}   "
              f"r {mean_of(recs, 'linvel_r'):6.3f}")

        print("  pose consequence (both arms seeded from the recorded pose at the chunk start)")
        print(f"    end-point drift pred vs GT    {mean_of(recs, 'drift_pred_vs_gt_mm'):8.2f} mm  "
              f"(p95 {q_of(recs, 'drift_pred_vs_gt_mm', 0.95):7.2f})")
        print(f"    end-point drift pred vs pos_w {mean_of(recs, 'drift_pred_vs_posw_mm'):8.2f} mm")
        print(f"    end-point drift GT   vs pos_w {mean_of(recs, 'drift_gt_vs_posw_mm'):8.2f} mm")
        print(f"    mean per-step position error  {mean_of(recs, 'step_err_mm'):8.2f} mm")
        print(f"    travelled per chunk: GT integral {mean_of(recs, 'path_gt_mm'):7.1f} mm, "
              f"recorded pos_w {mean_of(recs, 'path_posw_mm'):7.1f} mm")

        r_gt, r_pred = mean_of(recs, "R_gt"), mean_of(recs, "R_pred")
        print(f"    velocity scale R = sum(w*dt.u)/sum|dp_posw|:   GT {r_gt:.3f}   pred {r_pred:.3f}")
        if not np.isnan(r_gt) and not np.isnan(r_pred):
            if abs(r_pred - 1.0) < 0.08 < abs(r_gt - 1.0):
                print("      -> pred integrates to the PHYSICAL displacement while the recording does "
                      "not: the policy did NOT inherit the recording's velocity scale.")
            elif abs(r_pred - r_gt) < 0.1:
                print("      -> pred matches the recording's own scale: the inconsistency is in the "
                      "DATA, inherited faithfully (pred-vs-GT stays small, pred-vs-pos_w does not).")
            else:
                print("      -> pred's scale matches NEITHER 1.0 nor the recording: real defect.")

        if args.repeats > 1:
            sj = [mean_of(recs, f"self_joint_mrad_{i}") for i in range(1, args.repeats)]
            sl = [mean_of(recs, f"self_linvel_mae_{i}") for i in range(1, args.repeats)]
            sd = [mean_of(recs, f"self_drift_mm_{i}") for i in range(1, args.repeats)]
            print(f"  NOISE FLOOR -- same observation, fresh sample (repeats={args.repeats})")
            print(f"    joint  {np.nanmean(sj):8.1f} mrad   vs GT {mean_of(recs, 'joint_all_mrad'):8.1f}")
            print(f"    linvel {np.nanmean(sl):8.4f} m/s    vs GT {mean_of(recs, 'linvel_mae'):8.4f}")
            print(f"    drift  {np.nanmean(sd):8.2f} mm     vs GT {mean_of(recs, 'drift_pred_vs_gt_mm'):8.2f}")
            floor = np.nanmean(sj)
            ratio = mean_of(recs, "joint_all_mrad") / floor if floor > 1e-9 else float("inf")
            print(f"    -> pred-vs-GT joint error is {ratio:.2f}x the noise floor. "
                  + ("At or below ~1.5x, the recording sits inside the policy's own sampling spread: "
                     "no systematic defect to chase."
                     if ratio <= 1.5 else
                     "Well above the floor: the gap from the recording is systematic, not sampling."))

        if n_rows_max > args.open_loop_horizon:
            print(f"  error vs row index inside the chunk (first {args.open_loop_horizon} = executed)")
            n_show = min(n_rows_max, 50)
            rows_j, rows_l = [], []
            for i in range(n_show):
                rows_j.append(mean_of([{"v": r["row_joint_mrad"][i]} for r in recs
                                       if len(r["row_joint_mrad"]) > i], "v"))
                rows_l.append(mean_of([{"v": r["row_linvel"][i]} for r in recs
                                       if len(r["row_linvel"]) > i], "v"))
            for i in range(0, n_show, 5):
                mark = "exec" if i < args.open_loop_horizon else "    "
                print(f"    row {i:2d} {mark}  joint {rows_j[i]:8.1f} mrad   "
                      f"linvel {rows_l[i]:7.4f} m/s")
            exec_j = np.nanmean(rows_j[:args.open_loop_horizon])
            tail_j = np.nanmean(rows_j[args.open_loop_horizon:])
            if not np.isnan(tail_j):
                print(f"    executed mean {exec_j:.1f} vs beyond-horizon mean {tail_j:.1f} mrad "
                      + ("(flat: the open-loop window is not the lever)"
                         if abs(tail_j - exec_j) < 0.25 * max(exec_j, 1e-9)
                         else "(grows past the horizon: a shorter window would help)"))

    # ---------------- artifacts ----------------
    summary = {
        "dataset_root": root, "episodes": eps_seen, "chunks": len(records),
        "inferences": n_calls, "stride": args.stride,
        "open_loop_horizon": args.open_loop_horizon, "repeats": args.repeats,
        "server": f"{args.remote_host}:{args.remote_port}",
        "prompts": {k: t for k, t in prompt_arms},
        "dt": DT, "action_dim_compared": ACTION_DIM,
        "note": ("GT columns [33:40] are not policy outputs; used only as DEBUG ground truth "
                 "for the pose seed and the recorded world displacement"),
        "per_prompt": {},
        "chunks_detail": [{k: v for k, v in r.items() if not isinstance(v, list)} for r in records],
    }
    for key, _ in prompt_arms:
        recs = [r for r in records if r["prompt"] == key]
        if not recs:
            continue
        keys = sorted({k for r in recs for k in r
                       if isinstance(r[k], (int, float)) and k not in _ID_KEYS
                       and not k.startswith("self_")})
        arm = {k: mean_of(recs, k) for k in keys}
        arm.update({f"{k}_p95": q_of(recs, k, 0.95) for k in keys})
        if args.repeats > 1:
            for name, key_fmt in (("joint_mrad", "self_joint_mrad_{}"),
                                  ("linvel_mae", "self_linvel_mae_{}"),
                                  ("drift_mm", "self_drift_mm_{}")):
                arm[f"noise_floor_{name}"] = float(np.nanmean(
                    [mean_of(recs, key_fmt.format(i)) for i in range(1, args.repeats)]))
        summary["per_prompt"][key] = arm

    if args.out_dir:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / "summary.json"
        with open(path, "w") as fh:
            json.dump(summary, fh, indent=2)
        print(f"\nwrote {path}")

    if args.dump and rolled["pred"]:
        out = Path(args.dump)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out,
            pred=pad_stack(rolled["pred"], ACTION_DIM),      # (n_chunks, rows, 33)
            gt=pad_stack(rolled["gt"], POSE_HI),             # (n_chunks, rows, 40)
            chunk_episode=np.asarray(rolled["ep"], dtype=np.int64),
            chunk_t0=np.asarray(rolled["t0"], dtype=np.int64),
            chunk_prompt=np.asarray(rolled["prompt"]),
            dataset_root=np.array(root),
            prompts=np.asarray([t for _, t in prompt_arms]),
            open_loop_horizon=np.array(args.open_loop_horizon),
            stride=np.array(args.stride), repeats=np.array(args.repeats),
        )
        print(f"wrote {out} ({len(rolled['pred'])} chunks)")


if __name__ == "__main__":
    main(tyro.cli(Args))
