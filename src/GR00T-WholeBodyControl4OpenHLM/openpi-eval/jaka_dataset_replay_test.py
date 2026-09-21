"""Offline dataset replay for the Jaka deployment env, against a live policy server.

Feeds recorded frames from ``teleop_jaka_mf/simple/JakaTabletopPickTeleop-v0/level-0``
into ``JakaTabletopEnv`` (via a subclass that overrides ``get_observation``), lets the
env request inference from a running ``serve_policy.py``, and compares what came back
with the recorded ground truth. Two things are under the microscope:

* ``actions[30:33]`` -- ``anchor_lin_vel`` in the ANCHOR BODY frame, clipped to +-2.0;
* the reconstructed anchor world position -- integrating the action trunk with the same
  convention the env uses, checked against the dataset's raw ``actions[33:36]``
  (``anchor_pos_w``, the DEBUG column that is a verbatim copy of the recording).

The position error is decomposed so a bad number points at a cause rather than a
mystery. For every chunk, four trajectories are integrated from the SAME seed with the
same integrator, only the inputs differ:

    A  GT lin_vel   + TRUE quat (from actions[36:40])   -> lin_vel/dt/body-frame convention
    B  GT lin_vel   + yaw integrated from yaw_vel       -> cost of the deployment shortcut
    C  pred lin_vel + yaw integrated from yaw_vel       -> what deployment actually publishes
    D  pred lin_vel + TRUE quat                         -> pure policy lin_vel error

``B - A`` is the shortcut alone, ``D - A`` is the policy's lin_vel, ``C`` is the two
together. Separately, the whole episode is integrated continuously (never reseeded),
which is exactly what the env publishes, and those frames are decoded off the wire and
compared to the offline math.

Two extra checks exist because the obvious reading of this data can be wrong:

* **lag scan** -- the recorded ``lin_vel`` is a difference over the interval ENDING at
  its frame, while the integrator sums forward from the frame's own pose. If that is how
  the trunk was built, every leg is offset by one frame and leg A's residual would be the
  per-step displacement, not an error. The scan re-scores each leg at shifts -2..+2 and
  reports which shift is actually best, so the convention is measured, not assumed.
* **convention probe** -- per-frame least-squares ``dt`` (``<R(q)v, dp> / ||R(q)v||^2``)
  against both readings, plus the explained-displacement ratio. This is what tells you
  whether the pico publish period behind ``lin_vel`` matches the dataset's 1/30. Only
  frames where the anchor is genuinely moving carry that information, so every statistic
  is reported twice (all frames, and ``_restricted`` above the speed/displacement floor)
  and ``proj_ratio`` -- the displacement-weighted scale factor -- is the one to trust.
  The same section classifies the recorded rows (moving / jitter / dropout / clipped),
  because the trunk's velocity and position channels are not self-consistent row by row.

Run on the DEPLOYMENT machine (native, no container/sudo, same machine as the policy
server). The dataset is auto-detected under the repo -- ``data/simple/...`` there,
``data/teleop_jaka_mf/simple/...`` on the training machine::

    cd ~/Workspace/OpenHLM/src/GR00T-WholeBodyControl4OpenHLM/openpi-eval
    export PYTHONPATH=~/Workspace/OpenHLM/src/GR00T-WholeBodyControl4OpenHLM

    # terminal 1: policy server (GPU machine, needs no pi05_base weights)
    cd ~/Workspace/OpenHLM/src/openpi4OpenHLM
    ./.venv/bin/python scripts/serve_policy.py --env JAKA --num-steps 10

    # terminal 2: this script -- smoke first, then the full 47 episodes
    ~/Workspace/OpenHLM/src/openpi4OpenHLM/.venv/bin/python jaka_dataset_replay_test.py \
      --mode both --episodes 0 --max-chunks 4
    ~/Workspace/OpenHLM/src/openpi4OpenHLM/.venv/bin/python jaka_dataset_replay_test.py \
      --mode both

``--mode gt`` needs no server at all (it replays the recorded actions), which is the
cheapest way to check the plumbing before spending GPU time.

Useful flags: ``--episodes 0 1 2`` (SPACE separated -- tyro takes a tuple, so
``--episodes 0,1,2`` is a parse error; empty = all 47), ``--max-chunks N`` to stop early,
``--dataset-root`` if the data is somewhere else, ``--remote-host/--remote-port`` for the
policy server, ``--prompt`` to override the dataset's own instruction, ``--out-dir``.
Outputs land in the cwd-independent ``<script dir>/visualization/dataset_replay/<stamp>/``.

Same script inside the training container, for reference::

    docker exec -i -w /workspace/OpenHLM/src/GR00T-WholeBodyControl4OpenHLM/openpi-eval \
      -e PYTHONPATH=/workspace/OpenHLM/src/GR00T-WholeBodyControl4OpenHLM \
      xujinfan_dev /workspace/OpenHLM/src/openpi4OpenHLM/.venv/bin/python \
      jaka_dataset_replay_test.py --mode gt --episodes 0

The action PUB writes to 28701, the real deployment port. That is fine as long as no
``RealtimeMotionBufferVla`` / robot is subscribed -- nothing else needs to listen for the
wire check, which subscribes itself. **Before running this with a live receiver, pass
``--motion-port 28799``**, or the replayed trajectory will drive the robot.

Known trap: the server blocks its event loop while inferring, so the first client request
after a server restart can die on the 20 s ping timeout. ``PolicyClient`` reconnects once.
"""

# flake8: noqa: E402
import sys
import os

# Mirrors main.py: makes the GR00T repo root importable (openpi_client is already
# installed in the venv, this is for jaka_tabletop_env / sonic_g1_env).
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import dataclasses
import io
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import pyarrow.parquet as pq
import tyro
from PIL import Image
from scipy.spatial.transform import Rotation

try:
    import matplotlib
    matplotlib.use("Agg")  # headless: no X display on either machine
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except ImportError:
    # The deployment venv only has to run the policy server, so matplotlib may be
    # missing there. The plots are diagnostics -- the numbers live in summary.json and
    # per_chunk.npz -- so losing them must not kill a run that already burned GPU time.
    plt = None
    HAVE_MPL = False

from openpi_client import image_tools
from openpi_client import websocket_client_policy

from jaka_tabletop_env import JakaTabletopEnv, PERM_POLICY_TO_SIM
# Reuse the wire-contract test's authoritative implementations: ``decode_binary_v1`` is
# AST-extracted from the receiver's motion_buffer.py, ``quat_from_rpy`` from
# action_trunk_reconstruct.py, so both agree with the offline reference by construction.
from jaka_tabletop_env_test import decode_binary_v1, quat_from_rpy

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
# The dataset lives under the repo on both machines, but at different depths: the
# training machine keeps it under ``data/teleop_jaka_mf/simple/...``, the deployment
# machine under ``data/simple/...``. Auto-detect so the same file runs on either, and
# let --dataset-root override when it is somewhere else entirely.
_DATASET_LEAF = ("simple", "JakaTabletopPickTeleop-v0", "level-0")
DATASET_ROOT_CANDIDATES = (
    os.path.join(_REPO_ROOT, "data", "teleop_jaka_mf", *_DATASET_LEAF),
    os.path.join(_REPO_ROOT, "data", *_DATASET_LEAF),
)


def _detect_dataset_root() -> str:
    for cand in DATASET_ROOT_CANDIDATES:
        if os.path.isfile(os.path.join(cand, "meta", "info.json")):
            return cand
    return DATASET_ROOT_CANDIDATES[0]


DEFAULT_DATASET_ROOT = _detect_dataset_root()

DT = 1.0 / 30.0  # dataset fps, see meta/info.json ("fps": 30)
CLIP = 2.0       # actions[30:33] was clipped to +-2.0 at record time
# Motion floor for fitting the lin_vel/dt convention. Below these the recorded velocity
# and the recorded displacement are both jitter, and their ratio is meaningless -- see
# convention_probe. 0.2 m/s over one frame is 6.7 mm, well clear of the step jitter.
SPEED_FLOOR = 0.2   # m/s
DISP_FLOOR = 5e-3   # m, per frame
SHIFTS = (-2, -1, 0, 1, 2)
LEG_ORDER = ("A", "A_eff", "B", "C", "C_eff", "D")  # A_eff/C_eff are diagnostics only
LEGS_SCANNED = ("A", "B", "C", "D")                 # lag_scan runs on these


# ---------------------------------------------------------------------------
# Integrator -- the one piece of math everything else is measured against
# ---------------------------------------------------------------------------

def yaw_of_quat(quat_wxyz) -> float:
    """Extract the yaw of a wxyz quaternion in the same euler convention as quat_from_rpy.

    The dataset stores the anchor's absolute orientation only in ``actions[36:40]``;
    the recorded ``yaw_vel`` is a rate, so the absolute yaw has to come from here.
    """
    q = np.asarray(quat_wxyz, dtype=np.float64)
    return float(Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_euler("xyz")[2])


def _yaws_of_quats(quats_wxyz) -> np.ndarray:
    q = np.asarray(quats_wxyz, dtype=np.float64).reshape(-1, 4)
    return Rotation.from_quat(np.roll(q, -1, axis=-1)).as_euler("xyz")[:, 2]


def _rotate(quat_wxyz, vec) -> np.ndarray:
    """``R(q) . v`` for a batch of wxyz quats and vectors (scipy, independent of the env)."""
    return Rotation.from_quat(np.roll(np.asarray(quat_wxyz, np.float64), -1, axis=-1)).apply(
        np.asarray(vec, np.float64)
    )


def integrate_anchor(
    lin_vel,
    pos_seed,
    *,
    quat=None,
    roll=None,
    pitch=None,
    yaw_vel=None,
    yaw_seed: float = 0.0,
    dt: float = DT,
):
    """Integrate an anchor trunk back to a world pose trajectory.

    Mirrors ``action_trunk_reconstruct.reconstruct_anchor_motion`` and
    ``JakaTabletopEnv.step``::

        yaw[j]   = yaw_seed + sum_{m<j} yaw_vel[m] * dt      # yaw[0] == yaw_seed
        quat[j]  = quat_from_rpy(roll[j], pitch[j], yaw[j])
        world[j] = R(quat[j]) . lin_vel[j]                    # body frame -> world
        pos[k]   = pos_seed + sum_{j<k} world[j] * dt         # frame k is the START of its interval

    Pass ``quat`` to use a known orientation (the TRUE anchor quat, leg A/D) instead of
    integrating ``yaw_vel`` (the deployment shortcut, leg B/C).

    Returns ``(pos[T,3], quat[T,4] wxyz, yaw[T])``.
    """
    lin_vel = np.asarray(lin_vel, dtype=np.float64)
    T = lin_vel.shape[0]
    if T == 0:
        return np.zeros((0, 3)), np.zeros((0, 4)), np.zeros(0)

    if quat is None:
        yaw = np.empty(T, dtype=np.float64)
        yaw[0] = float(yaw_seed)
        if T > 1:
            yaw[1:] = float(yaw_seed) + np.cumsum(np.asarray(yaw_vel, np.float64)[:-1] * dt)
        quat = np.asarray(
            quat_from_rpy(np.asarray(roll, np.float64), np.asarray(pitch, np.float64), yaw),
            dtype=np.float64,
        )
    else:
        quat = np.asarray(quat, dtype=np.float64)
        yaw = _yaws_of_quats(quat)

    world = _rotate(quat, lin_vel)
    pos = np.empty((T, 3), dtype=np.float64)
    pos[0] = np.asarray(pos_seed, dtype=np.float64)
    if T > 1:
        pos[1:] = pos[0] + np.cumsum(world[:-1] * dt, axis=0)
    return pos, quat, yaw


# ---------------------------------------------------------------------------
# Policy client
# ---------------------------------------------------------------------------

class PolicyClient:
    """Thin wrapper: reconnect once if the first request hits the server's blocked loop."""

    def __init__(self, host: str, port: int):
        self.host, self.port = host, port
        self._client = websocket_client_policy.WebsocketClientPolicy(host, port)
        self._retried = False

    def infer(self, request: dict) -> dict:
        try:
            return self._client.infer(request)
        except Exception as e:  # noqa: BLE001 - the known first-connect timeout
            if self._retried:
                raise
            self._retried = True
            print(f"   [warn] infer failed ({type(e).__name__}: {e}) -- reconnecting once "
                  f"(expected if the server restarted after this client connected)")
            self._client = websocket_client_policy.WebsocketClientPolicy(self.host, self.port)
            return self._client.infer(request)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def check_meta(root: str) -> dict:
    """Guard the two conventions that would silently corrupt everything downstream.

    The merged dataset is already in POLICY order (arms first) -- if this ever changes,
    ``state``/``actions`` indices mean something different and every number below is
    wrong, so fail loudly rather than measuring the wrong thing.
    """
    with open(os.path.join(root, "meta", "info.json")) as fh:
        info = json.load(fh)
    state_names = info["features"]["state"]["names"]
    action_names = info["features"]["actions"]["names"]
    assert abs(float(info["fps"]) - 1.0 / DT) < 1e-6, f"dataset fps {info['fps']} != 30"
    assert "shoulder" in state_names[0].lower(), f"state[0] is {state_names[0]!r}, expected an arm joint"
    assert state_names[24] == "waist_yaw_joint", f"state[24] is {state_names[24]!r}"
    assert action_names[:27] == state_names[:27], "actions[0:27] and state[0:27] disagree"
    assert action_names[27:33] == ["root_roll", "root_pitch", "yaw_vel",
                                   "anchor_lin_vel_x", "anchor_lin_vel_y", "anchor_lin_vel_z"]
    assert action_names[33:40] == ["anchor_pos_w_x", "anchor_pos_w_y", "anchor_pos_w_z",
                                   "anchor_quat_w_w", "anchor_quat_w_x", "anchor_quat_w_y",
                                   "anchor_quat_w_z"]
    return info


def load_episode(root: str, episode: int) -> dict:
    """Read one episode's frames. Images are inline PNG bytes; nothing else is decoded."""
    path = os.path.join(root, "data", "chunk-000", f"episode_{episode:06d}.parquet")
    table = pq.read_table(path)
    n = table.num_rows

    # Pillow, not cv2: PIL gives RGB like the training pipeline, cv2 silently gives BGR.
    images = np.empty((n, 224, 224, 3), dtype=np.uint8)
    # head_image_left is struct<bytes, path>; combine_chunks() gives a StructArray whose
    # "bytes" field is the inline PNG (this dataset has no videos/ directory).
    blobs = table["head_image_left"].combine_chunks().field("bytes")
    for i, blob in enumerate(blobs):
        images[i] = np.asarray(Image.open(io.BytesIO(blob.as_py())).convert("RGB"), dtype=np.uint8)

    return {
        "index": episode,
        "images": images,
        "states": np.array(table["state"].to_pylist(), dtype=np.float32),
        "actions": np.array(table["actions"].to_pylist(), dtype=np.float32),
    }


def episode_lengths(root: str) -> dict:
    lengths = {}
    with open(os.path.join(root, "meta", "episodes.jsonl")) as fh:
        for line in fh:
            rec = json.loads(line)
            lengths[int(rec["episode_index"])] = int(rec["length"])
    return lengths


def dataset_task(root: str) -> str:
    """The prompt the policy was TRAINED with (see the module docstring's prompt note)."""
    with open(os.path.join(root, "meta", "tasks.jsonl")) as fh:
        return json.loads(fh.readline())["task"]


# ---------------------------------------------------------------------------
# Convention probe: is the recorded lin_vel a forward or a backward difference,
# and does its dt match the dataset's 1/30?
# ---------------------------------------------------------------------------

def convention_probe(actions: np.ndarray) -> dict:
    """Least-squares ``dt`` for both readings of ``anchor_lin_vel``.

    For each frame i with body velocity ``w = R(q_i) . v_i``, the displacement the trunk
    claims is ``w * dt``. Comparing that against the actual displacement gives

        dt_hat = <w, dp> / <w, w>

    with ``dp = P[i+1] - P[i]`` (forward reading: the frame's velocity drives the NEXT
    interval) or ``dp = P[i] - P[i-1]`` (backward reading: the frame's velocity describes
    the interval just ENDED). Whichever lands near 1/30 is how the trunk was built; the
    two are only distinguishable where the velocity changes, so read this together with
    the lag scan. ``explained`` = median(||dp|| / (||w||*dt)) is the scale check and does
    not assume a direction.

    Both are MEDIANS over frames and so are only meaningful on frames where the anchor is
    genuinely moving: on near-stationary frames ``dp`` is recorder jitter, ``w*dt`` is
    noise, and the ratio of two near-zero numbers is arbitrary (ratios of 1e4 were
    observed at speeds of 1e-3 m/s). Every statistic therefore comes in two flavours --
    ``*`` over all unclipped moving-ish frames, ``*_restricted`` over frames above
    ``SPEED_FLOOR``/``DISP_FLOOR`` -- and a large gap between them means the unrestricted
    number is contaminated, not that the convention is wrong. ``proj_ratio`` is the
    displacement-weighted scale factor ``sum(w.u) / sum(||dp||)``: unlike the median it is
    dominated by the fast frames, so it is the statistic that actually answers "is the
    recorded velocity the right MAGNITUDE" (1.0 = yes).

    ``defect`` then classifies the recorded rows, because parts of the trunk are simply
    not self-consistent with ``[33:36]``: rows where the velocity reads ~0 while the
    position moves (a dropped pico payload -- ``dropout``), rows where the velocity is
    alive but the position does not move (``jitter``), and the +/-2.0 saturating rows.
    These are properties of the recording, and they are what makes the drift grow inside
    an open-loop chunk, so they are reported rather than treated as an implementation bug.
    """
    P = actions[:, 33:36].astype(np.float64)
    V = actions[:, 30:33].astype(np.float64)
    Q = actions[:, 36:40].astype(np.float64)
    T = P.shape[0]
    W = _rotate(Q, V)
    speed = np.linalg.norm(W, axis=1)
    clipped = np.abs(V).max(axis=1) >= CLIP - 1e-3

    out = {"n_frames": int(T), "n_clipped": int(clipped.sum())}
    for name, dp in (("fwd", np.diff(P, axis=0)),          # dp[i] pairs with frame i
                     ("bwd", np.diff(P, axis=0))):          # dp[i-1] pairs with frame i
        if name == "fwd":
            idx = np.arange(T - 1)
            d = dp
        else:
            idx = np.arange(1, T)
            d = dp
        w = W[idx]
        norm2 = (w * w).sum(axis=1)
        # Only frames with real motion and no clip carry information about dt.
        valid = (norm2 > 1e-6) & (speed[idx] > 1e-3) & (~clipped[idx])
        # ...and only frames whose POSITION moved carry information about the SCALE; a
        # frame can clear the speed floor and still have jitter for a displacement.
        disp_all = np.linalg.norm(d, axis=1)
        strict = (speed[idx] > SPEED_FLOOR) & (disp_all > DISP_FLOOR) & (~clipped[idx])
        if valid.sum() < 5:
            out[name] = {"n_valid": int(valid.sum())}
            continue
        dt_hat = (w[valid] * d[valid]).sum(axis=1) / norm2[valid]
        disp = np.linalg.norm(d[valid], axis=1)
        claim = speed[idx][valid] * DT
        good = claim > 1e-4
        entry = {
            "n_valid": int(valid.sum()),
            "dt_hat_median": float(np.median(dt_hat)),
            "dt_hat_p25": float(np.percentile(dt_hat, 25)),
            "dt_hat_p75": float(np.percentile(dt_hat, 75)),
            "implied_fps": float(1.0 / np.median(dt_hat)) if abs(np.median(dt_hat)) > 1e-9 else float("nan"),
            "frac_dt_hat_near_1_30": float(np.mean(np.abs(dt_hat - DT) < 0.15 * DT)),
            "explained_median": float(np.median(disp[good] / claim[good])) if good.any() else float("nan"),
        }
        entry["n_restricted"] = int(strict.sum())
        if strict.sum() >= 5:
            w_s, d_s = w[strict], d[strict]
            n2_s = norm2[strict]
            dt_s = (w_s * d_s).sum(axis=1) / n2_s
            # Displacement-weighted projection: sum of the component of the claimed motion
            # along the true step, over the true path. 1.0 => magnitudes agree. This is
            # the honest scale number -- the per-frame median hides the moving frames,
            # where the excess actually lives, behind the many near-still ones.
            u = d_s / np.linalg.norm(d_s, axis=1)[:, None]
            entry["dt_hat_median_restricted"] = float(np.median(dt_s))
            entry["explained_median_restricted"] = float(
                np.median(np.linalg.norm(d_s, axis=1) / (speed[idx][strict] * DT)))
            entry["proj_ratio"] = float((w_s * DT * u).sum() / np.linalg.norm(d_s, axis=1).sum())
            # The per-frame divisor the recorder actually used. If the trunk's velocity is
            # (world displacement)/(pico publish period) rather than /(1/30), integrating
            # it at dt=1/30 scales every step by 1/proj_ratio, and the pico period is this.
            entry["implied_divisor_ms"] = float(DT / entry["proj_ratio"]) * 1000.0 if entry["proj_ratio"] > 0 else float("nan")
        out[name] = entry

    # Frames where velocity and displacement disagree in kind: the pico payload the
    # recorder consumed was a stale one (stall) or the same one twice (duplicate).
    sp, disp = speed[1:], np.linalg.norm(np.diff(P, axis=0), axis=1)
    out["stall_frames"] = int(np.sum((sp < 1e-3) & (disp > 1e-3)))
    out["dup_frames"] = int(np.sum((disp < 1e-4) & (sp > 0.05)))
    out["step_disp_mean"] = float(disp.mean())
    out["step_disp_p95"] = float(np.percentile(disp, 95))
    # Defect accounting: where the recorded velocity and the recorded position disagree
    # in KIND, and how much of the GT path each class carries. `claim`/`true` are the
    # path lengths the two channels each imply over that class.
    cls = {
        "moving": (~clipped[1:]) & (sp > SPEED_FLOOR) & (disp > 2e-3),
        "jitter": (~clipped[1:]) & (sp > 0.05) & (disp <= 2e-3),
        "dropout": (~clipped[1:]) & (sp <= 0.05) & (disp > 5e-3),
        "clipped": clipped[1:],
    }
    defect = {}
    claimed = speed[1:] * DT
    for name, mask in cls.items():
        defect[name] = {
            "n": int(mask.sum()),
            "true_mm": float(disp[mask].sum()) * 1000.0,
            "claim_mm": float(claimed[mask].sum()) * 1000.0,
        }
    defect["true_path_mm"] = float(disp.sum()) * 1000.0
    defect["claim_path_mm"] = float(claimed.sum()) * 1000.0
    defect["n_any"] = int(T - 1)
    out["defect"] = defect
    # The dt the recorded lin_vel actually behaves like. Integrating at 1/30 when the
    # trunk was built with this value is a pure SCALE error: the reconstruction then
    # recovers only dt_eff*fps of the recorded displacement. Fitted on the restricted
    # (genuinely moving) frames; see the docstring for why the unrestricted fit is junk.
    fwd = out.get("fwd", {})
    out["dt_eff"] = fwd.get("dt_hat_median_restricted", fwd.get("dt_hat_median", float("nan")))
    return out


# ---------------------------------------------------------------------------
# Env: dataset frames in, real ZMQ frames out
# ---------------------------------------------------------------------------

class DatasetJakaEnv(JakaTabletopEnv):
    """``JakaTabletopEnv`` whose observations come from the dataset instead of ZMQ.

    ``mock=True`` skips connecting the :28711/:28712 observation sockets (the replay
    supplies the frames); the action PUB socket is still bound, so ``step`` exercises the
    real conversion + wire path.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._img = None
        self._state = None
        self._t = 0

    def load_episode(self, images: np.ndarray, states: np.ndarray) -> None:
        self._img, self._state, self._t = images, states, 0

    def seek(self, t: int) -> None:
        self._t = int(t)

    def get_observation(self) -> dict:
        return {
            "head_image_left": self._img[self._t],
            "state": self._state[self._t].copy(),
        }


def build_request(obs: dict, prompt: str) -> dict:
    """Same key set as ``main.py:233`` build_policy_request for jaka (bare keys, one image)."""
    image = image_tools.resize_with_pad(obs["head_image_left"], 224, 224)  # no-op at 224x224
    return {"head_image_left": image, "state": obs["state"], "prompt": prompt}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def mae(a, b) -> float:
    return float(np.mean(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))))


def rmse(a, b) -> float:
    d = np.asarray(a, np.float64) - np.asarray(b, np.float64)
    return float(np.sqrt(np.mean(d * d)))


def maxabs(a, b) -> float:
    return float(np.max(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))))


def pearson(a, b) -> float:
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    if a.size < 2 or a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def nanmean(vals) -> float:
    vals = [v for v in vals if v is not None and not np.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


def pos_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Position error of a reconstructed trajectory vs the raw truth.

    xy is compared as displacement (the pico world origin is arbitrary per episode and
    every leg starts at ``gt[0]``, so this is a relative comparison); z is compared
    absolutely (the tracker's ``root_z_mf`` is an absolute world z).
    """
    pred = np.asarray(pred, np.float64)
    gt = np.asarray(gt, np.float64)
    d_xy = pred[:, :2] - gt[:, :2]
    d_z = pred[:, 2] - gt[:, 2]
    step_err = np.linalg.norm(np.diff(pred, axis=0) - np.diff(gt, axis=0), axis=1)
    return {
        "end_xy": float(np.linalg.norm(d_xy[-1])),
        "end_z": float(abs(d_z[-1])),
        "max_xy": float(np.max(np.linalg.norm(d_xy, axis=1))),
        "max_z": float(np.max(np.abs(d_z))),
        "step_mean": float(np.mean(step_err)) if step_err.size else float("nan"),
        "step_p95": float(np.percentile(step_err, 95)) if step_err.size else float("nan"),
    }


def lag_scan(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Re-score one leg at frame shifts -2..+2.

    The integrator puts ``pred[k]`` at the pose the frame claims for the interval starting
    at k. If the recorded ``lin_vel`` is instead a difference over the interval ENDING at
    its frame, the whole reconstruction is one frame early, and the honest residual is the
    one measured at shift +1. Comparing only at shift 0 would then report the per-step
    displacement as if it were integration error.
    """
    pred = np.asarray(pred, np.float64)
    gt = np.asarray(gt, np.float64)
    T = pred.shape[0]
    out = {}
    for s in SHIFTS:
        ks = np.arange(max(0, -s), min(T, T - s))
        if ks.size < 2:
            continue
        p = pred[ks] - pred[0]
        g = gt[ks + s] - gt[0]
        out[str(s)] = {
            "xy": float(np.mean(np.linalg.norm(p[:, :2] - g[:, :2], axis=1))),
            "z": float(np.mean(np.abs(p[:, 2] - g[:, 2]))),
        }
    return out


def linvel_metrics(pred: np.ndarray, gt: np.ndarray, quat_gt: np.ndarray | None = None) -> dict:
    pred = np.asarray(pred, np.float64)
    gt = np.asarray(gt, np.float64)
    out = {
        "mae": [mae(pred[:, i], gt[:, i]) for i in range(3)],
        "rmse": [rmse(pred[:, i], gt[:, i]) for i in range(3)],
        "max": [maxabs(pred[:, i], gt[:, i]) for i in range(3)],
        "corr": [pearson(pred[:, i], gt[:, i]) for i in range(3)],
        "gt_abs_mean": [float(np.mean(np.abs(gt[:, i]))) for i in range(3)],
        "gt_abs_max": [float(np.max(np.abs(gt[:, i]))) for i in range(3)],
        "mae_all": mae(pred, gt),
        "rmse_all": rmse(pred, gt),
        "max_all": maxabs(pred, gt),
    }
    if quat_gt is not None:
        # Body-frame per-axis error mixes a heading error into a velocity error: the same
        # body-frame vector means a different world direction if the yaw is off. Rotating
        # both by the TRUE quat removes the heading and leaves the velocity error alone.
        werr = _rotate(quat_gt, pred - gt)
        out["world_mae"] = [mae(werr[:, i], np.zeros(len(werr))) for i in range(3)]
        out["world_mae_all"] = float(np.mean(np.abs(werr)))
        out["world_step_mm"] = out["world_mae_all"] * DT * 1000.0  # per-step displacement error
    return out


# ---------------------------------------------------------------------------
# Wire capture
# ---------------------------------------------------------------------------

def drain_into(sub, frames: dict, *, expect: int | None = None, timeout_s: float = 0.25,
               discard: bool = False) -> None:
    """Pull published protocol-v1 frames into ``{frame_index: (joint_pos, pos, quat)}``.

    Called after every chunk (short timeout) so the PUB never accumulates a whole
    episode's worth of messages, and once at the end with ``expect=T``.
    """
    import zmq

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            msg = sub.recv(zmq.NOBLOCK)
        except zmq.Again:
            if expect is not None and len(frames) >= expect:
                return
            time.sleep(0.005)
            continue
        if discard:
            continue
        joint_pos, body_pos_w, body_quat_w, frame_index = decode_binary_v1(msg)
        for row in range(len(frame_index)):
            frames[int(frame_index[row])] = (
                joint_pos[row].copy(), body_pos_w[row].copy(), body_quat_w[row].copy()
            )


def check_wire(frames: dict, executed: np.ndarray, pos_env: np.ndarray, quat_env: np.ndarray) -> dict:
    """Compare decoded ZMQ frames against the offline math and the input actions.

    Frame index k is the pose at the START of step k and equals ``pos_env[k]``; the
    sliding window means frame k only reaches the wire in step k+1's message, so N steps
    publish N-1 messages while still covering frames 0..N-1.
    """
    T = executed.shape[0]
    missing = [i for i in range(T) if i not in frames]
    out = {"published": len(frames), "expected": T, "n_missing": len(missing),
           "missing": missing[:10]}
    if missing:
        return out

    joint_err = max(maxabs(frames[i][0], executed[i, :27][PERM_POLICY_TO_SIM]) for i in range(T))
    pos_err = max(maxabs(frames[i][1], pos_env[i]) for i in range(T))
    # Quaternion sign is not unique, so compare the rotation, not the 4 numbers.
    quat_err = max(
        float(min(maxabs(frames[i][2], quat_env[i]), maxabs(frames[i][2], -quat_env[i])))
        for i in range(T)
    )
    out.update({"joint_max_err": joint_err, "pos_max_err": pos_err, "quat_max_err": quat_err})
    return out


# ---------------------------------------------------------------------------
# One (episode, mode) run
# ---------------------------------------------------------------------------

def run_mode(ep: dict, mode: str, args, client, prompt: str, sub, dt_eff=None) -> dict:
    """Replay one episode in ``gt`` or ``policy`` mode; returns per-chunk records.

    ``dt_eff`` enables the diagnostic ``*_eff`` legs (see ``chunk_record``); it never
    changes what the env publishes, only what is compared offline.
    """
    T = ep["actions"].shape[0]
    if args.max_chunks:
        T = min(T, args.max_chunks * args.open_loop_horizon)
    gt_actions = ep["actions"][:T]

    pos0 = gt_actions[0, 33:36].astype(np.float64)
    roll0, pitch0 = float(gt_actions[0, 27]), float(gt_actions[0, 28])
    yaw0 = yaw_of_quat(gt_actions[0, 36:40])

    # Per-episode seeding through the public constructor args: x/y are arbitrary in the
    # pico world but z is load-bearing, and the yaw seed is what yaw_vel accumulates onto.
    env = DatasetJakaEnv(
        control_hz=30,
        mock=True,
        motion_zmq_address="127.0.0.1",
        motion_zmq_port=args.motion_port,
        initial_anchor_pos=tuple(float(v) for v in pos0),
        initial_anchor_rpy=(roll0, pitch0, yaw0),
    )
    frames = {}
    try:
        env.load_episode(ep["images"], ep["states"])
        # A fresh env restarts frame_index at 0, which is what makes the decoded index a
        # direct row index into pos_env below. Every (episode, mode) rebinds the PUB.
        time.sleep(0.3)  # PUB slow-joiner: let the new bind reach the already-connected SUB
        drain_into(sub, frames, timeout_s=0.1, discard=True)  # drop anything left over

        executed = np.zeros((T, 33), dtype=np.float32)
        chunks = []
        infer_ms = []
        t = 0
        while t < T:
            H = min(args.open_loop_horizon, T - t)
            horizon = None
            if mode == "policy":
                env.seek(t)
                start = time.time()
                pred = np.asarray(client.infer(build_request(env.get_observation(), prompt))["actions"])
                infer_ms.append((time.time() - start) * 1e3)
                if pred.shape[0] < H or pred.shape[1] < 33:
                    raise RuntimeError(f"policy returned {pred.shape}, need at least ({H}, 33)")
                action_chunk = pred[:H, :33].astype(np.float32)
                # Rows H..49 are thrown away by the deployment; keep them for the free
                # error-vs-horizon curve (a notch at j=0 would mean the index map is off).
                avail = min(pred.shape[0], T - t)
                horizon = (np.asarray(pred[:avail, :33], np.float64),
                           gt_actions[t:t + avail, 30:33].astype(np.float64))
            else:
                action_chunk = gt_actions[t:t + H, :33]

            for k in range(H):
                env.step(action_chunk[k])
                executed[t + k] = action_chunk[k]

            drain_into(sub, frames, timeout_s=0.02)
            chunks.append(chunk_record(gt_actions, t, H, action_chunk, mode, horizon, dt_eff))
            t += H
    finally:
        env.close()
        time.sleep(0.2)  # let the port settle before the next env binds it

    # Whole-episode continuous integration: never reseeded, so this is exactly what the
    # env published (its yaw accumulates across the whole run just like this).
    pos_env, quat_env, yaw_env = integrate_anchor(
        executed[:, 30:33], pos0,
        roll=executed[:, 27], pitch=executed[:, 28], yaw_vel=executed[:, 29], yaw_seed=yaw0, dt=DT,
    )
    # Same velocities, TRUE orientations: leg A (gt mode) / leg D (policy mode), continuous.
    pos_true_quat = integrate_anchor(executed[:, 30:33], pos0, quat=gt_actions[:, 36:40])[0]

    drain_into(sub, frames, expect=T, timeout_s=3.0)
    wire = check_wire(frames, executed, pos_env, quat_env)
    gt_pos = gt_actions[:, 33:36].astype(np.float64)

    return {
        "episode": ep["index"],
        "mode": mode,
        "frames": T,
        "chunks": chunks,
        "infer_ms_mean": float(np.mean(infer_ms)) if infer_ms else None,
        "wire": wire,
        "episode_pos": {
            "gt": gt_pos,
            "env": pos_env,
            "true_quat": pos_true_quat,
            "yaw_env": yaw_env,
            "yaw_gt": _yaws_of_quats(gt_actions[:, 36:40]),
        },
        "episode_metrics": pos_metrics(pos_env, gt_pos),
        "episode_lag": lag_scan(pos_env, gt_pos),
        "episode_lag_true_quat": lag_scan(pos_true_quat, gt_pos),
    }


def chunk_record(gt_actions: np.ndarray, t0: int, H: int, action_chunk: np.ndarray,
                 mode: str, horizon=None, dt_eff=None) -> dict:
    """Integrate the four legs for one chunk, all from the same seed.

    ``dt_eff`` additionally produces the ``*_eff`` legs: the same integration with the
    measured effective dt instead of 1/30. That is a diagnostic only -- the env and the
    deployment always use 1/control_hz -- but it separates "the convention is wrong" from
    "the recorded trunk was built with a different dt than the frames imply".
    """
    gt = gt_actions[t0:t0 + H]
    pos_seed = gt[0, 33:36].astype(np.float64)
    yaw_seed = yaw_of_quat(gt[0, 36:40])
    pos_gt = gt[:, 33:36].astype(np.float64)
    gt_vel = gt[:, 30:33].astype(np.float64)
    quat_gt = gt[:, 36:40]

    pos_A = integrate_anchor(gt_vel, pos_seed, quat=quat_gt)[0]
    pos_B, _, yaw_B = integrate_anchor(
        gt_vel, pos_seed,
        roll=gt[:, 27], pitch=gt[:, 28], yaw_vel=gt[:, 29], yaw_seed=yaw_seed,
    )

    clip_mask = np.abs(gt_vel).max(axis=1) >= CLIP - 1e-3
    rec = {
        "t0": int(t0),
        "H": int(H),
        "pos_gt": pos_gt,
        "pos_A": pos_A,
        "pos_B": pos_B,
        "lin_vel_gt": gt_vel,
        "m_gt": {"A": pos_metrics(pos_A, pos_gt), "B": pos_metrics(pos_B, pos_gt)},
        "lag_gt": {"A": lag_scan(pos_A, pos_gt), "B": lag_scan(pos_B, pos_gt)},
        "seed": pos_seed,
        "max_abs_yaw_vel": float(np.max(np.abs(gt[:, 29]))),
        "clip_frac": float(clip_mask.mean()),
        "yaw_gt": float(yaw_of_quat(gt[-1, 36:40])),
        "yaw_B": float(yaw_B[-1]),
        "step_disp_mm": float(np.mean(np.linalg.norm(np.diff(pos_gt, axis=0), axis=1))) * 1000.0,
    }

    if dt_eff is not None and np.isfinite(dt_eff) and dt_eff > 0:
        pos_A_eff = integrate_anchor(gt_vel, pos_seed, quat=quat_gt, dt=dt_eff)[0]
        rec["pos_A_eff"] = pos_A_eff
        rec["m_gt"]["A_eff"] = pos_metrics(pos_A_eff, pos_gt)

    if mode == "policy":
        pred_vel = action_chunk[:, 30:33].astype(np.float64)
        err = np.abs(pred_vel - gt_vel)
        pos_C = integrate_anchor(
            pred_vel, pos_seed,
            roll=action_chunk[:, 27], pitch=action_chunk[:, 28],
            yaw_vel=action_chunk[:, 29], yaw_seed=yaw_seed,
        )[0]
        pos_D = integrate_anchor(pred_vel, pos_seed, quat=quat_gt)[0]
        rec.update({
            "pred_actions": action_chunk.astype(np.float32),
            "lin_vel_pred": pred_vel,
            "pos_C": pos_C,
            "pos_D": pos_D,
            "linvel": linvel_metrics(pred_vel, gt_vel, quat_gt),
            "joint_mae": mae(action_chunk[:, :27], gt[:, :27]),
            "joint_mae_arms": mae(action_chunk[:, :12], gt[:, :12]),
            "joint_mae_legs": mae(action_chunk[:, 12:24], gt[:, 12:24]),
            "joint_mae_waist_neck": mae(action_chunk[:, 24:27], gt[:, 24:27]),
            "err_clipped": float(err[clip_mask].mean()) if clip_mask.any() else float("nan"),
            "err_unclipped": float(err[~clip_mask].mean()) if (~clip_mask).any() else float("nan"),
        })
        rec["m_pred"] = {"C": pos_metrics(pos_C, pos_gt), "D": pos_metrics(pos_D, pos_gt)}
        rec["lag_pred"] = {"C": lag_scan(pos_C, pos_gt), "D": lag_scan(pos_D, pos_gt)}
        if "pos_A_eff" in rec:
            pos_C_eff = integrate_anchor(
                pred_vel, pos_seed,
                roll=action_chunk[:, 27], pitch=action_chunk[:, 28],
                yaw_vel=action_chunk[:, 29], yaw_seed=yaw_seed, dt=dt_eff,
            )[0]
            rec["pos_C_eff"] = pos_C_eff
            rec["m_pred"]["C_eff"] = pos_metrics(pos_C_eff, pos_gt)
        if horizon is not None:
            rec["horizon_pred"], rec["horizon_gt"] = horizon
    return rec


# ---------------------------------------------------------------------------
# Aggregation / reporting
# ---------------------------------------------------------------------------

def leg_metric(rec: dict, leg: str, key: str) -> float:
    m = rec["m_pred"][leg] if leg in rec.get("m_pred", {}) else rec["m_gt"][leg]
    return m[key]


def leg_lag(rec: dict, leg: str) -> dict:
    return rec["lag_pred"][leg] if leg in rec.get("lag_pred", {}) else rec["lag_gt"][leg]


def summarise_lag(recs: list, leg: str) -> dict:
    """Mean error per shift, and which shift is best."""
    per_shift = {}
    for s in SHIFTS:
        xy = [leg_lag(c, leg)[str(s)]["xy"] for c in recs if str(s) in leg_lag(c, leg)]
        z = [leg_lag(c, leg)[str(s)]["z"] for c in recs if str(s) in leg_lag(c, leg)]
        if xy:
            per_shift[str(s)] = {"xy": nanmean(xy), "z": nanmean(z)}
    if not per_shift:
        return {}
    best = min(per_shift, key=lambda k: per_shift[k]["xy"])
    return {
        "best_shift": int(best),
        "best_xy": per_shift[best]["xy"],
        "xy_at_0": per_shift.get("0", {}).get("xy", float("nan")),
        "z_at_0": per_shift.get("0", {}).get("z", float("nan")),
        "best_z": per_shift[best]["z"],
        "per_shift": per_shift,
    }


def aggregate(runs: list) -> dict:
    summary = {
        "n_episodes": len({r["episode"] for r in runs}),
        "n_frames": int(sum(r["frames"] for r in runs)),
        "n_chunks": int(sum(len(r["chunks"]) for r in runs)),
    }

    per_mode = {}
    for mode in sorted({r["mode"] for r in runs}):
        recs = [c for r in runs if r["mode"] == mode for c in r["chunks"]]
        entry = {"n_chunks": len(recs)}
        for leg in LEG_ORDER:
            if not all(leg in c["m_gt"] or leg in c.get("m_pred", {}) for c in recs):
                continue
            end_xy = [leg_metric(c, leg, "end_xy") for c in recs]
            end_z = [leg_metric(c, leg, "end_z") for c in recs]
            entry[f"pos_{leg}"] = {
                "end_xy_mean": nanmean(end_xy),
                "end_xy_p95": float(np.percentile(end_xy, 95)),
                "end_z_mean": nanmean(end_z),
                "end_z_p95": float(np.percentile(end_z, 95)),
                "step_mean": nanmean(leg_metric(c, leg, "step_mean") for c in recs),
                "max_xy_mean": nanmean(leg_metric(c, leg, "max_xy") for c in recs),
            }
            entry[f"pos_{leg}_lag"] = summarise_lag(recs, leg) if leg in LEGS_SCANNED else {}
            # Does the +-2.0 clip show up as underestimated displacement?
            with_clip = [c for c in recs if c["clip_frac"] > 0]
            without_clip = [c for c in recs if c["clip_frac"] == 0]
            entry[f"pos_{leg}_clip_split"] = {
                "n_with_clip": len(with_clip),
                "end_xy_with": nanmean(leg_metric(c, leg, "end_xy") for c in with_clip),
                "end_xy_without": nanmean(leg_metric(c, leg, "end_xy") for c in without_clip),
            }

        if mode == "policy":
            axis = ["mae", "rmse", "max", "corr", "gt_abs_mean", "gt_abs_max", "world_mae"]
            entry["lin_vel_per_axis"] = {
                k: [nanmean(c["linvel"][k][i] for c in recs) for i in range(3)] for k in axis
            }
            entry["lin_vel_mae_all"] = nanmean(c["linvel"]["mae_all"] for c in recs)
            entry["lin_vel_world_mae_all"] = nanmean(c["linvel"]["world_mae_all"] for c in recs)
            entry["lin_vel_world_step_mm"] = nanmean(c["linvel"]["world_step_mm"] for c in recs)
            entry["joint_mae"] = nanmean(c["joint_mae"] for c in recs)
            entry["joint_mae_arms"] = nanmean(c["joint_mae_arms"] for c in recs)
            entry["joint_mae_legs"] = nanmean(c["joint_mae_legs"] for c in recs)
            entry["joint_mae_waist_neck"] = nanmean(c["joint_mae_waist_neck"] for c in recs)
            entry["clip_frac"] = nanmean(c["clip_frac"] for c in recs)
            entry["err_clipped"] = nanmean(c["err_clipped"] for c in recs)
            entry["err_unclipped"] = nanmean(c["err_unclipped"] for c in recs)
            entry["infer_ms_mean"] = nanmean(
                r["infer_ms_mean"] for r in runs if r["mode"] == "policy" and r["infer_ms_mean"]
            )
            entry["horizon_mae"] = _horizon_mae(recs)
        entry["step_disp_mm"] = nanmean(c["step_disp_mm"] for c in recs)
        entry["max_abs_yaw_vel"] = float(np.max([c["max_abs_yaw_vel"] for c in recs]))
        entry["yaw_drift_deg_mean"] = nanmean(abs(np.degrees(c["yaw_B"] - c["yaw_gt"])) for c in recs)
        per_mode[mode] = entry

    summary["modes_detail"] = per_mode
    eps = [r["episode_metrics"] for r in runs]
    summary["episode_end_xy_mean"] = nanmean(m["end_xy"] for m in eps)
    summary["episode_end_xy_p95"] = float(np.percentile([m["end_xy"] for m in eps], 95))
    summary["episode_end_xy_max"] = float(np.max([m["end_xy"] for m in eps]))
    summary["episode_end_z_mean"] = nanmean(m["end_z"] for m in eps)
    summary["episode_end_z_p95"] = float(np.percentile([m["end_z"] for m in eps], 95))
    summary["episode_end_z_max"] = float(np.max([m["end_z"] for m in eps]))
    summary["episode_lag"] = {
        kind: {
            "best_shift": int(min(lag, key=lambda k: lag[k]["xy"])),
            "xy_best": lag[min(lag, key=lambda k: lag[k]["xy"])]["xy"],
            "xy_at_0": lag.get("0", {}).get("xy", float("nan")),
        }
        for kind, lag in (("published", _mean_lag([r["episode_lag"] for r in runs])),
                          ("true_quat", _mean_lag([r["episode_lag_true_quat"] for r in runs])))
    }
    summary["wire"] = {
        "n_runs": len(runs),
        "max_joint_err": max(r["wire"].get("joint_max_err", 0.0) for r in runs),
        "max_pos_err": max(r["wire"].get("pos_max_err", 0.0) for r in runs),
        "max_quat_err": max(r["wire"].get("quat_max_err", 0.0) for r in runs),
        "total_missing_frames": int(sum(r["wire"]["n_missing"] for r in runs)),
    }
    return summary


def _mean_lag(lags: list) -> dict:
    out = {}
    for s in SHIFTS:
        xy = [lag[str(s)]["xy"] for lag in lags if str(s) in lag]
        z = [lag[str(s)]["z"] for lag in lags if str(s) in lag]
        if xy:
            out[str(s)] = {"xy": nanmean(xy), "z": nanmean(z)}
    return out


def _horizon_mae(recs: list) -> dict:
    """MAE of the anchor velocity by row index inside the 50-row chunk."""
    n = max((len(c["horizon_pred"]) for c in recs if "horizon_pred" in c), default=0)
    if not n:
        return {}
    acc = [[] for _ in range(n)]
    for c in recs:
        if "horizon_pred" not in c:
            continue
        pred, gt = c["horizon_pred"], c["horizon_gt"]
        for j in range(len(pred)):
            acc[j].append(np.abs(pred[j] - gt[j]).mean())
    mae_by_j = [nanmean(v) for v in acc]
    return {"n_rows": n, "mae_by_row": mae_by_j,
            "mae_j0": mae_by_j[0], "mae_j1_3": nanmean(mae_by_j[1:4]),
            "mae_j24": mae_by_j[24] if n > 24 else float("nan"),
            "mae_j49": mae_by_j[-1]}


def print_report(summary: dict, runs: list, probes: list) -> None:
    print("\n" + "=" * 96)
    print(f"dataset replay: {summary['n_episodes']} episodes / {summary['n_frames']} frames "
          f"/ {summary['n_chunks']} chunks")
    print("=" * 96)

    print("\nconvention probe (is the recorded lin_vel a forward or backward difference, "
          "and does its dt match 1/30?)")
    for name in ("fwd", "bwd"):
        vals = [p[name] for p in probes if p.get(name, {}).get("n_valid", 0) > 0]
        if not vals:
            continue
        reading = "frame's velocity drives the NEXT interval" if name == "fwd" else \
                  "frame's velocity describes the interval just ENDED"
        print(f"   {name} ({reading}):")
        print(f"      dt_hat median {nanmean(v['dt_hat_median'] for v in vals) * 1000:7.2f} ms "
              f"(p25 {nanmean(v['dt_hat_p25'] for v in vals) * 1000:.2f}, "
              f"p75 {nanmean(v['dt_hat_p75'] for v in vals) * 1000:.2f})  -> implied fps "
              f"{nanmean(v['implied_fps'] for v in vals):6.2f}   "
              f"within 15% of 1/30: {nanmean(v['frac_dt_hat_near_1_30'] for v in vals) * 100:5.1f}%")
        print(f"      explained displacement ratio median "
              f"{nanmean(v['explained_median'] for v in vals):.3f}  "
              f"(1.0 = recorded dt matches 1/30 exactly)")
        rst = [v for v in vals if v.get("n_restricted", 0) >= 5]
        if rst:
            print(f"      -- restricted to |v|>{SPEED_FLOOR} m/s and |dp|>{DISP_FLOOR * 1000:.0f} mm/frame "
                  f"({nanmean(v['n_restricted'] for v in rst):.0f} frames/ep):")
            print(f"         dt_hat {nanmean(v['dt_hat_median_restricted'] for v in rst) * 1000:7.2f} ms   "
                  f"explained {nanmean(v['explained_median_restricted'] for v in rst):.3f}   "
                  f"proj_ratio {nanmean(v['proj_ratio'] for v in rst):.3f}  "
                  f"(displacement-weighted; 1.0 = magnitudes agree)")
            print(f"         -> on frames where the anchor actually moves, R(q).v*dt claims "
                  f"{nanmean(v['proj_ratio'] for v in rst):.3f}x the true displacement; "
                  f"equivalent divisor {nanmean(v['implied_divisor_ms'] for v in rst):.2f} ms "
                  f"({1000.0 / max(nanmean(v['implied_divisor_ms'] for v in rst), 1e-9):.1f} Hz), "
                  f"not the {DT * 1000:.2f} ms frame period")
    print(f"   clipped frames {sum(p['n_clipped'] for p in probes)} / "
          f"{sum(p['n_frames'] for p in probes)};  "
          f"stall frames {sum(p['stall_frames'] for p in probes)};  "
          f"duplicate frames {sum(p['dup_frames'] for p in probes)}")

    print("\n   recorded-row defects (why the open-loop position drifts; these are in the")
    print("   DATA, not in the integrator -- the trunk's velocity and position channels")
    print("   are not self-consistent row by row):")
    tot_true = sum(p["defect"]["true_path_mm"] for p in probes)
    print(f"      class      rows    GT path mm   velocity claims mm   ratio")
    for name in ("moving", "jitter", "dropout", "clipped"):
        n = sum(p["defect"][name]["n"] for p in probes)
        if not n:
            continue
        true_mm = sum(p["defect"][name]["true_mm"] for p in probes)
        claim_mm = sum(p["defect"][name]["claim_mm"] for p in probes)
        pct = true_mm / tot_true * 100 if tot_true else 0.0
        print(f"      {name:9s} {n:6d}   {true_mm:11.1f} {claim_mm:18.1f}   "
              f"{claim_mm / true_mm if true_mm > 1e-9 else float('nan'):6.2f}   ({pct:4.1f}% of GT path)")
    print(f"      {'ALL':9s} {sum(p['defect']['n_any'] for p in probes):6d}   "
          f"{sum(p['defect']['true_path_mm'] for p in probes):11.1f} "
          f"{sum(p['defect']['claim_path_mm'] for p in probes):18.1f}   "
          f"{sum(p['defect']['claim_path_mm'] for p in probes) / max(tot_true, 1e-9):6.2f}")
    print(f"   GT per-step displacement: mean {nanmean(p['step_disp_mean'] for p in probes) * 1000:.2f} mm, "
          f"p95 {nanmean(p['step_disp_p95'] for p in probes) * 1000:.2f} mm "
          f"(this is the floor any shift-0 comparison inherits if the trunk is backward-differenced)")

    labels = {"A": "GT vel + TRUE quat    (convention)",
              "A_eff": " as A, with the fitted dt (diagnostic)",
              "B": "GT vel + integrated yaw (shortcut)",
              "C": "pred vel + integrated yaw (DEPLOYED)",
              "C_eff": " as C, with the fitted dt (diagnostic)",
              "D": "pred vel + TRUE quat   (policy only)"}
    for mode, e in summary["modes_detail"].items():
        legs = [l for l in LEG_ORDER if f"pos_{l}" in e]
        print(f"\n[{mode}]  {e['n_chunks']} chunks"
              + (f"   infer {e['infer_ms_mean']:.0f} ms/chunk" if e.get("infer_ms_mean") else ""))
        print(f"   {'':6s} {'':39s} {'end_xy mm':>18s} {'end_z mm':>18s} {'step mm':>9s}")
        for leg in legs:
            m = e[f"pos_{leg}"]
            print(f"   {leg:6s} {labels[leg]:39s} "
                  f"{m['end_xy_mean'] * 1000:8.2f} (p95 {m['end_xy_p95'] * 1000:6.2f}) "
                  f"{m['end_z_mean'] * 1000:8.2f} (p95 {m['end_z_p95'] * 1000:6.2f}) "
                  f"{m['step_mean'] * 1000:8.3f}")
        print(f"   lag scan (mean error at frame shift; {e['step_disp_mm']:.2f} mm/step of GT motion):")
        for leg in LEGS_SCANNED:
            lg = e.get(f"pos_{leg}_lag") or {}
            if not lg:
                continue
            shiftstr = "  ".join(f"{s}:{lg['per_shift'][s]['xy'] * 1000:6.2f}"
                                 for s in map(str, SHIFTS) if s in lg["per_shift"])
            print(f"      {leg}  xy mm @ shift {shiftstr}   -> best {lg['best_shift']:+d} "
                  f"({lg['best_xy'] * 1000:.2f} mm vs {lg['xy_at_0'] * 1000:.2f} at 0)")
        for leg in legs:
            s = e.get(f"pos_{leg}_clip_split")
            if s and s["n_with_clip"]:
                print(f"      {leg}: clip-hit chunks {s['n_with_clip']:3d}  "
                      f"end_xy {s['end_xy_with'] * 1000:7.2f} mm  vs no-clip chunks "
                      f"{s['end_xy_without'] * 1000:7.2f} mm")
        if "lin_vel_per_axis" in e:
            la = e["lin_vel_per_axis"]
            print(f"   lin_vel MAE  x/y/z (m/s): "
                  f"{', '.join(f'{v:.4f}' for v in la['mae'])}  (mean {e['lin_vel_mae_all']:.4f})")
            print(f"   lin_vel RMSE x/y/z      : {', '.join(f'{v:.4f}' for v in la['rmse'])}")
            print(f"   lin_vel corr x/y/z      : {', '.join(f'{v:.3f}' for v in la['corr'])}"
                  f"   (|GT| mean {', '.join(f'{v:.3f}' for v in la['gt_abs_mean'])}"
                  f", max {', '.join(f'{v:.3f}' for v in la['gt_abs_max'])})")
            print(f"   world-frame MAE x/y/z   : {', '.join(f'{v:.4f}' for v in la['world_mae'])}"
                  f"   (= {e['lin_vel_world_step_mm']:.3f} mm/step of anchor displacement error)")
            print(f"   err on clip-hit rows {e['err_clipped']:.4f} vs other rows "
                  f"{e['err_unclipped']:.4f}  (GT rows at clip: {e['clip_frac'] * 100:.2f}%)")
            print(f"   joints MAE {e['joint_mae']:.4f} rad  (arms {e['joint_mae_arms']:.4f}, "
                  f"legs {e['joint_mae_legs']:.4f}, waist/neck {e['joint_mae_waist_neck']:.4f})")
            h = e.get("horizon_mae") or {}
            if h:
                print(f"   lin_vel MAE by row in chunk: j=0 {h['mae_j0']:.4f}  j=1-3 "
                      f"{h['mae_j1_3']:.4f}  j=24 {h['mae_j24']:.4f}  j=49 {h['mae_j49']:.4f}"
                      f"   (j=0 worse than j=1-3 would mean the 1:1 index map is off by one)")
        print(f"   yaw drift from integrating yaw_vel: {e['yaw_drift_deg_mean']:.2f} deg/chunk "
              f"(peak |yaw_vel| {e['max_abs_yaw_vel']:.2f} rad/s)")

    w = summary["wire"]
    print(f"\nwire check: {w['n_runs']} runs, missing frames {w['total_missing_frames']}, max err"
          f"  joint {w['max_joint_err']:.2e}  pos {w['max_pos_err']:.2e}  quat {w['max_quat_err']:.2e}")
    print("   (env publishes N-1 messages per N steps; frame k reaches the wire in step k+1's window)")
    print("episode-level drift, continuous (what deployment really publishes):")
    print(f"   end xy  mean {summary['episode_end_xy_mean'] * 1000:7.2f} mm  "
          f"p95 {summary['episode_end_xy_p95'] * 1000:7.2f}  max {summary['episode_end_xy_max'] * 1000:7.2f}")
    print(f"   end z   mean {summary['episode_end_z_mean'] * 1000:7.2f} mm  "
          f"p95 {summary['episode_end_z_p95'] * 1000:7.2f}  max {summary['episode_end_z_max'] * 1000:7.2f}")
    for kind, lg in summary["episode_lag"].items():
        print(f"   episode-level lag ({kind}): best shift {lg['best_shift']:+d} "
              f"({lg['xy_best'] * 1000:.2f} mm vs {lg['xy_at_0'] * 1000:.2f} at shift 0)")

    print("\nper (episode, mode):")
    for r in runs:
        m = r["episode_metrics"]
        extra = f"  infer {r['infer_ms_mean']:.0f} ms/chunk" if r["infer_ms_mean"] else ""
        print(f"   ep{r['episode']:<3d} {r['mode']:6s} {r['frames']:4d} frames  "
              f"end xy {m['end_xy'] * 1000:7.2f} mm  z {m['end_z'] * 1000:7.2f} mm  "
              f"step {m['step_mean'] * 1000:.3f} mm  "
              f"wire pos {r['wire'].get('pos_max_err', float('nan')):.2e}{extra}")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_episode(run: dict, out_dir: Path) -> None:
    ep, mode = run["episode"], run["mode"]
    ep_pos = run["episode_pos"]
    chunks = run["chunks"]

    if chunks and "lin_vel_pred" in chunks[0]:
        gt = np.concatenate([c["lin_vel_gt"] for c in chunks])
        pred = np.concatenate([c["lin_vel_pred"] for c in chunks])
        fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
        for i, ax in enumerate(axes):
            ax.plot(gt[:, i], label="GT (dataset)", lw=1.2)
            ax.plot(pred[:, i], label="policy", lw=1.2, alpha=0.8)
            for c in chunks:
                ax.axvline(c["t0"], color="0.85", lw=0.6, zorder=0)
            ax.axhline(CLIP, color="r", ls=":", lw=0.8)
            ax.axhline(-CLIP, color="r", ls=":", lw=0.8)
            ax.set_ylabel(f"lin_vel {'xyz'[i]} (m/s)")
            ax.grid(alpha=0.3)
        axes[0].legend(loc="upper right")
        axes[0].set_title(f"episode {ep} anchor_lin_vel (body frame), policy vs GT — "
                          f"grey: chunk boundaries, red: +-2.0 clip")
        axes[-1].set_xlabel("frame")
        fig.tight_layout()
        fig.savefig(out_dir / f"linvel_ep{ep}_{mode}.png", dpi=120)
        plt.close(fig)

    gt = np.asarray(ep_pos["gt"])
    envp = np.asarray(ep_pos["env"])
    tq = np.asarray(ep_pos["true_quat"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(gt[:, 0], gt[:, 1], label="GT actions[33:36]", lw=1.6)
    axes[0].plot(envp[:, 0], envp[:, 1], label="published (integrated yaw)", lw=1.3, alpha=0.85)
    axes[0].plot(tq[:, 0], tq[:, 1], label="integrated, TRUE quat", lw=1.1, alpha=0.7, ls="--")
    axes[0].set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")
    axes[0].set_title("anchor xy (pico world; origin is arbitrary per episode)")
    axes[0].axis("equal")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[1].plot(gt[:, 2], label="GT", lw=1.6)
    axes[1].plot(envp[:, 2], label="published", lw=1.3, alpha=0.85)
    axes[1].plot(tq[:, 2], label="TRUE quat", lw=1.1, alpha=0.7, ls="--")
    axes[1].set_xlabel("frame")
    axes[1].set_ylabel("anchor z (m, absolute)")
    axes[1].set_title("anchor z — absolute world height (root_z_mf)")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"pos_ep{ep}_{mode}.png", dpi=120)
    plt.close(fig)

    yaw_gt = np.asarray(ep_pos["yaw_gt"])
    yaw_env = np.asarray(ep_pos["yaw_env"])
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(np.degrees(np.unwrap(yaw_gt)), label="GT anchor yaw (from actions[36:40])", lw=1.6)
    ax.plot(np.degrees(np.unwrap(yaw_env)), label="integrated from yaw_vel", lw=1.3, alpha=0.85)
    ax.set_xlabel("frame")
    ax.set_ylabel("yaw (deg)")
    ax.set_title(f"episode {ep} [{mode}] — yaw integration drift (the mechanism behind leg B/C)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"yaw_ep{ep}_{mode}.png", dpi=120)
    plt.close(fig)


def plot_aggregate(runs: list, out_dir: Path, summary: dict) -> None:
    recs = [c for r in runs if r["mode"] == "policy" for c in r["chunks"]]
    if not recs:
        return
    label_of = {"A": "GT vel\nTRUE quat", "A_eff": "GT vel\nTRUE quat\nfitted dt",
                "B": "GT vel\nint. yaw", "C": "pred vel\nint. yaw",
                "C_eff": "pred vel\nint. yaw\nfitted dt", "D": "pred vel\nTRUE quat"}
    legs = [(l, label_of[l]) for l in LEG_ORDER
            if all(l in c["m_gt"] or l in c.get("m_pred", {}) for c in recs)]
    H = recs[0]["H"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, comp, name in ((axes[0], "end_xy", "xy"), (axes[1], "end_z", "z")):
        data, labels = [], []
        for leg, label in legs:
            data.append([leg_metric(c, leg, comp) * 1000 for c in recs])
            labels.append(f"{leg}\n{label}")
        ax.boxplot(data, showfliers=False)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel(f"per-chunk end drift {name} (mm)")
        ax.set_title(f"{name} error by error source ({H}-step chunk)")
        ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "chunk_drift_by_source.png", dpi=120)
    plt.close(fig)

    # Error vs frame shift: flat-at-0 means the convention is right; a clear minimum at +-1
    # means the reconstruction is offset by a frame rather than wrong.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, comp, name in ((axes[0], "xy", "xy"), (axes[1], "z", "z")):
        for leg, label in [(l, label_of[l]) for l, _ in legs if l in LEGS_SCANNED]:
            ys = []
            for s in SHIFTS:
                vals = [leg_lag(c, leg)[str(s)][comp] * 1000 for c in recs
                        if str(s) in leg_lag(c, leg)]
                ys.append(nanmean(vals))
            ax.plot(SHIFTS, ys, marker="o", label=f"{leg}: {label.replace(chr(10), ' ')}")
        ax.set_xticks(SHIFTS)
        ax.set_xlabel("frame shift applied to the truth")
        ax.set_ylabel(f"mean {name} error (mm)")
        ax.set_title(f"{name} error vs frame shift (min at 0 = convention confirmed)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "lag_scan.png", dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for i, ax in enumerate(axes):
        gt = np.concatenate([c["lin_vel_gt"][:, i] for c in recs])
        pr = np.concatenate([c["lin_vel_pred"][:, i] for c in recs])
        ax.scatter(np.abs(gt), np.abs(pr - gt), s=4, alpha=0.3)
        ax.axvline(CLIP, color="r", ls="--", lw=1, label="clip (+-2.0)")
        ax.set_xlabel(f"|GT lin_vel {'xyz'[i]}| (m/s)")
        ax.set_ylabel("|error| (m/s)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("lin_vel error vs GT magnitude (clip boundary marked)")
    fig.tight_layout()
    fig.savefig(out_dir / "linvel_err_vs_magnitude.png", dpi=120)
    plt.close(fig)

    h = (summary["modes_detail"].get("policy") or {}).get("horizon_mae") or {}
    if h.get("mae_by_row"):
        fig, ax = plt.subplots(figsize=(9, 4))
        rows = np.arange(len(h["mae_by_row"]))
        ax.plot(rows, h["mae_by_row"], marker=".", lw=1.2)
        ax.axvline(H, color="r", ls="--", lw=1, label=f"executed horizon ({H})")
        ax.set_xlabel("row index inside the 50-row chunk")
        ax.set_ylabel("mean |lin_vel error| (m/s)")
        ax.set_title("lin_vel error vs position in the policy's chunk")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "linvel_horizon.png", dpi=120)
        plt.close(fig)


def save_arrays(runs: list, path: Path) -> None:
    npz = {}
    for r in runs:
        ch = r["chunks"]
        tag = f"ep{r['episode']}_{r['mode']}"
        npz[f"{tag}_t0"] = np.array([c["t0"] for c in ch], dtype=np.int64)
        for key in ("pos_gt", "pos_A", "pos_A_eff", "pos_B", "pos_C", "pos_C_eff", "pos_D",
                    "lin_vel_gt", "lin_vel_pred"):
            vals = [c[key] for c in ch if key in c]
            if vals:
                npz[f"{tag}_{key}"] = np.concatenate(vals)
        if ch and "pred_actions" in ch[0]:
            # The policy's executed chunk rows (H,33) per chunk, concatenated.
            npz[f"{tag}_pred_actions"] = np.concatenate([c["pred_actions"] for c in ch])
        npz[f"{tag}_pos_env"] = np.asarray(r["episode_pos"]["env"])
        npz[f"{tag}_pos_true_quat"] = np.asarray(r["episode_pos"]["true_quat"])
        npz[f"{tag}_yaw_gt"] = np.asarray(r["episode_pos"]["yaw_gt"])
        npz[f"{tag}_yaw_env"] = np.asarray(r["episode_pos"]["yaw_env"])
    np.savez_compressed(path, **npz)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Args:
    mode: Literal["gt", "policy", "both"] = "both"
    dataset_root: str = DEFAULT_DATASET_ROOT
    episodes: tuple[int, ...] = ()       # empty = all 47
    max_chunks: int = 0                  # >0 = stop after N chunks per episode (smoke test)
    open_loop_horizon: int = 25
    prompt: str = ""                     # empty = the dataset's own task string
    remote_host: str = "127.0.0.1"
    remote_port: int = 8000
    motion_port: int = 28701             # the real port; use 28799 if a receiver is live
    out_dir: str = ""                    # empty = visualization/dataset_replay/<timestamp>
    plot_episode: int = -1               # -1 = the first episode with a policy run
    no_plots: bool = False


def main(args: Args):
    import zmq

    root = os.path.abspath(args.dataset_root)
    if not os.path.isfile(os.path.join(root, "meta", "info.json")):
        checked = "\n".join(f"    {c}" for c in DATASET_ROOT_CANDIDATES)
        raise SystemExit(
            f"dataset not found at: {root}\n"
            f"  (looked for meta/info.json -- an existing directory without one is still wrong)\n"
            f"auto-detected candidates that were tried:\n{checked}\n"
            f"pass --dataset-root <path to the level-0 dir> if it lives elsewhere"
        )
    check_meta(root)

    lengths = episode_lengths(root)
    episodes = list(args.episodes) if args.episodes else sorted(lengths)
    for ep in episodes:
        if ep not in lengths:
            raise SystemExit(f"episode {ep} not in {root} (max {max(lengths)})")

    trained_prompt = dataset_task(root)
    prompt = args.prompt or trained_prompt
    modes = ["gt", "policy"] if args.mode == "both" else [args.mode]

    out_dir = Path(args.out_dir) if args.out_dir else (
        Path(_HERE) / "visualization" / "dataset_replay" / datetime.now().strftime("%Y%m%d_%H%M")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"dataset : {root}")
    print(f"episodes: {episodes}  ({sum(lengths[e] for e in episodes)} frames)")
    print(f"prompt  : {prompt!r}"
          + ("   <- the dataset's training prompt" if not args.prompt else "   <- OVERRIDE"))
    if args.prompt and args.prompt != trained_prompt:
        print(f"   [warn] the policy was TRAINED with {trained_prompt!r} "
              f"(prompt_from_task=True); a different prompt at deploy time is a real "
              f"distribution shift, so compare both before blaming the weights")
    print(f"modes   : {modes}   horizon {args.open_loop_horizon}"
          + (f"   max_chunks {args.max_chunks}" if args.max_chunks else ""))
    print(f"out_dir : {out_dir}")

    client = None
    if "policy" in modes:
        print(f"\nconnecting to policy server at {args.remote_host}:{args.remote_port} ...")
        client = PolicyClient(args.remote_host, args.remote_port)
        print("connected (the first infer can take ~20 s while the server warms up)")

    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.SUBSCRIBE, b"")
    sub.connect(f"tcp://127.0.0.1:{args.motion_port}")
    time.sleep(0.3)

    runs = []
    probes = []
    try:
        for ep in episodes:
            data = load_episode(root, ep)
            probe = convention_probe(data["actions"])
            probes.append(probe)
            print(f"\nepisode {ep}: {data['actions'].shape[0]} frames, "
                  f"anchor z {data['actions'][0, 35]:.3f} -> {data['actions'][-1, 35]:.3f}, "
                  f"yaw {np.degrees(yaw_of_quat(data['actions'][0, 36:40])):.1f} -> "
                  f"{np.degrees(yaw_of_quat(data['actions'][-1, 36:40])):.1f} deg, "
                  f"dt_hat fwd {probe.get('fwd', {}).get('dt_hat_median_restricted', float('nan')) * 1000:.2f} ms "
                  f"bwd {probe.get('bwd', {}).get('dt_hat_median_restricted', float('nan')) * 1000:.2f} ms "
                  f"proj_ratio {probe.get('fwd', {}).get('proj_ratio', float('nan')):.3f}")
            for mode in modes:
                run = run_mode(data, mode, args, client, prompt, sub, probe.get("dt_eff"))
                runs.append(run)
                m = run["episode_metrics"]
                extra = f"  infer {run['infer_ms_mean']:.0f} ms/chunk" if run["infer_ms_mean"] else ""
                print(f"   [{mode:6s}] end drift xy {m['end_xy'] * 1000:7.2f} mm  "
                      f"z {m['end_z'] * 1000:7.2f} mm  step {m['step_mean'] * 1000:.3f} mm  "
                      f"wire pos err {run['wire'].get('pos_max_err', float('nan')):.2e}{extra}")
    finally:
        sub.close()
        ctx.term()

    summary = aggregate(runs)
    summary["convention_probe"] = {
        name: {k: nanmean(p[name][k] for p in probes if name in p and k in p[name])
               for k in ("dt_hat_median", "implied_fps", "frac_dt_hat_near_1_30",
                         "explained_median", "dt_hat_median_restricted",
                         "explained_median_restricted", "proj_ratio", "n_restricted",
                         "implied_divisor_ms")}
        for name in ("fwd", "bwd") if any(name in p for p in probes)
    }
    summary["convention_probe"]["defect_mm"] = {
        name: {"n": sum(p["defect"][name]["n"] for p in probes),
               "true_mm": sum(p["defect"][name]["true_mm"] for p in probes),
               "claim_mm": sum(p["defect"][name]["claim_mm"] for p in probes)}
        for name in ("moving", "jitter", "dropout", "clipped")
    }
    summary["convention_probe"]["defect_mm"]["true_path_total_mm"] = sum(
        p["defect"]["true_path_mm"] for p in probes)
    summary["convention_probe"]["defect_mm"]["claim_path_total_mm"] = sum(
        p["defect"]["claim_path_mm"] for p in probes)
    summary["convention_probe"]["floors"] = {"speed_m_s": SPEED_FLOOR, "disp_m": DISP_FLOOR}
    summary["convention_probe"]["counts"] = {
        "n_frames": sum(p["n_frames"] for p in probes),
        "n_clipped": sum(p["n_clipped"] for p in probes),
        "stall_frames": sum(p["stall_frames"] for p in probes),
        "dup_frames": sum(p["dup_frames"] for p in probes),
        "step_disp_mean_m": nanmean(p["step_disp_mean"] for p in probes),
    }
    summary["config"] = {
        "dataset_root": root,
        "episodes": episodes,
        "prompt": prompt,
        "prompt_is_dataset_default": not args.prompt,
        "trained_prompt": trained_prompt,
        "modes": modes,
        "open_loop_horizon": args.open_loop_horizon,
        "max_chunks": args.max_chunks,
        "motion_port": args.motion_port,
        "remote": f"{args.remote_host}:{args.remote_port}",
        "dt": DT,
        "clip": CLIP,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "caveat": "open loop: the policy's observations are always the dataset's own "
                  "frames/state, so policy drift never feeds back into its input",
    }
    print_report(summary, runs, probes)

    with open(out_dir / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    save_arrays(runs, out_dir / "per_chunk.npz")
    print(f"\nwrote {out_dir}/summary.json and per_chunk.npz")

    if args.no_plots or not runs:
        pass
    elif not HAVE_MPL:
        print("matplotlib not installed in this venv -- skipping plots "
              "(summary.json and per_chunk.npz are complete; install matplotlib for the PNGs)")
    else:
        plot_run = next((r for r in runs if r["mode"] == "policy"), runs[0])
        if args.plot_episode >= 0:
            plot_run = next((r for r in runs if r["episode"] == args.plot_episode), plot_run)
        plot_episode(plot_run, out_dir)
        plot_aggregate(runs, out_dir, summary)
        print(f"wrote plots to {out_dir}/ (episode {plot_run['episode']} [{plot_run['mode']}] "
              f"+ aggregates)")


if __name__ == "__main__":
    main(tyro.cli(Args))
