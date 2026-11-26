"""Self-contained record helpers for ``record_jaka_zmq_real.py``.

Migrated from the SIMPLE repo WITHOUT pulling in any ``simple.*`` module:

  * constants (JAKA joint/body names, openhlm dims, key names) — from
    ``simple/jaka_rl/config.py`` + ``simple/datasets/jaka_lerobot.py``;
  * ``LatestMotionFrame`` — from ``simple/jaka_rl/motion_buffer.py``;
  * ``quat_rotate_inverse_numpy`` — from ``simple/jaka_rl/math.py``;
  * ``OpenHLMRootVel`` / ``build_openhlm_features`` / ``build_openhlm_frame`` /
    ``reference_action_live_latest`` / ``init_openhlm_exporter`` /
    ``finalize_openhlm_episode`` / lerobot offline patches — verbatim ports from
    ``simple/datasets/jaka_lerobot.py``.

Runtime third-party deps (pip packages, NOT migrated — must be installed in the
environment that runs the recorder):
  numpy, scipy, pyzmq, typer, PyYAML, lerobot (imported lazily; ``packaging`` is
  used by the lerobot offline patch and ships with most envs).
No mujoco / simple / decoupled_wbc needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Constants (values identical to the SIMPLE side)
# ---------------------------------------------------------------------------

# 27 joints in MuJoCo order (simple.jaka_rl.config.JOINT_NAMES / JAKA_JOINT_NAMES)
JAKA_JOINT_NAMES: list[str] = [
    "Left_hip_pitch_joint",
    "Left_hip_roll_joint",
    "Left_hip_yaw_joint",
    "Left_knee_joint",
    "Left_ankle_pitch_joint",
    "Left_ankle_roll_joint",
    "Right_hip_pitch_joint",
    "Right_hip_roll_joint",
    "Right_hip_yaw_joint",
    "Right_knee_joint",
    "Right_ankle_pitch_joint",
    "Right_ankle_roll_joint",
    "waist_yaw_joint",
    "Left_shoulder_pitch_joint",
    "Left_shoulder_roll_joint",
    "Left_shoulder_yaw_joint",
    "Left_elbow_joint",
    "Left_wrist_roll_joint",
    "Left_wrist_yaw_joint",
    "Right_shoulder_pitch_joint",
    "Right_shoulder_roll_joint",
    "Right_shoulder_yaw_joint",
    "Right_elbow_joint",
    "Right_wrist_roll_joint",
    "Right_wrist_yaw_joint",
    "Neck_yaw_joint",
    "Neck_pitch_joint",
]

# 28 bodies in npz_body_names order (NOTE: Right_wrist_yaw__Link has two underscores)
JAKA_BODY_NAMES: list[str] = [
    "base_link", "Left_hip_pitch_Link", "Right_hip_pitch_Link", "waist_yaw_Link",
    "Left_hip_roll_Link", "Right_hip_roll_Link", "Left_shoulder_pitch_Link",
    "Right_shoulder_pitch_Link", "Neck_yaw_Link", "Left_hip_yaw_Link",
    "Right_hip_yaw_Link", "Left_shoulder_roll_Link", "Right_shoulder_roll_Link",
    "Neck_pitch_Link", "Left_knee_Link", "Right_knee_Link",
    "Left_shoulder_yaw_Link", "Right_shoulder_yaw_Link", "Left_ankle_pitch_Link",
    "Right_ankle_pitch_Link", "Left_elbow_Link", "Right_elbow_Link",
    "Left_ankle_roll_Link", "Right_ankle_roll_Link", "Left_wrist_roll_Link",
    "Right_wrist_roll_Link", "Left_wrist_yaw_Link", "Right_wrist_yaw__Link",
]

VIDEO_SHAPE: list[int] = [360, 640, 3]        # camera image shape (H, W, C)
OPENHLM_HEAD_IMAGE_SHAPE = [224, 224, 3]

OPENHLM_ROOT_DIM = 3          # roll, pitch, yaw_vel
OPENHLM_LIN_VEL_DIM = 3       # reference anchor linear velocity xyz
OPENHLM_ANCHOR_POSE_DIM = 7   # raw world anchor pose: pos xyz + quat wxyz
OPENHLM_STATE_DIM = len(JAKA_JOINT_NAMES) + OPENHLM_ROOT_DIM        # 30
OPENHLM_ACTION_DIM = (
    len(JAKA_JOINT_NAMES) + OPENHLM_ROOT_DIM + OPENHLM_LIN_VEL_DIM + OPENHLM_ANCHOR_POSE_DIM
)  # 40

# ZMQ payload key names (simple.jaka_rl.config)
PUBLISH_T_NS_KEY = "publish_t_ns"
SMPLX_T_NS_KEY = "smplx_t_ns"
SEQ_KEY = "seq"
JOINT_POS_KEY = "joint_pos"
JOINT_VEL_KEY = "joint_vel"
BODY_POS_W_KEY = "body_pos_w"
BODY_QUAT_W_KEY = "body_quat_w"
TOGGLE_DATA_COLLECTION_KEY = "toggle_data_collection"


# ---------------------------------------------------------------------------
# LatestMotionFrame (from simple.jaka_rl.motion_buffer)
# ---------------------------------------------------------------------------
@dataclass
class LatestMotionFrame:
    """The most recent reference motion frame (no control-loop look-back)."""

    joint_pos: np.ndarray  # (num_joints,), JAKA/robot order
    body_pos_w: np.ndarray  # (num_bodies, 3)
    body_quat_w: np.ndarray  # (num_bodies, 4), wxyz
    timestamp_ns: int
    prev_body_pos_w: Optional[np.ndarray]  # (num_bodies, 3) or None
    prev_timestamp_ns: Optional[int]  # or None
    seq: Optional[int] = None
    publish_t_ns: Optional[int] = None
    smplx_t_ns: Optional[int] = None
    receive_t_ns: Optional[int] = None
    receive_monotonic_ns: Optional[int] = None
    paused: bool = False
    prev_body_quat_w: Optional[np.ndarray] = None
    reference_monotonic_ns: Optional[int] = None
    reference_session_id: Optional[str] = None
    xr_sample: Optional[dict] = None  # raw SDK timestamps/provenance; not a verified capture clock
    xr_metadata: Optional[dict] = None  # static XR descriptor; persisted once per episode/content


def decode_recording_level(value: object) -> bool:
    """Decode a persistent level, including the legacy one-element bool list.

    This is a level, not an edge event. In particular ``[False]`` must never
    become True merely because the container is non-empty.
    """
    if isinstance(value, list) and len(value) == 1:
        value = value[0]
    if not isinstance(value, bool):
        raise ValueError("recording level must be a JSON boolean or [boolean]")
    return value


# ---------------------------------------------------------------------------
# quat_rotate_inverse_numpy (from simple.jaka_rl.math)
# ---------------------------------------------------------------------------
def quat_rotate_inverse_numpy(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector(s) by the inverse of quaternion(s).

    ``q`` is (..., 4) wxyz, ``v`` is (..., 3).
    """
    shape = v.shape
    q = q.reshape(-1, 4)
    v = v.reshape(-1, 3)
    q_w = q[:, 0]
    q_vec = q[:, 1:]
    a = v * (2.0 * q_w ** 2 - 1.0)[:, np.newaxis]
    b = np.cross(q_vec, v) * q_w[:, np.newaxis] * 2.0
    dot_product = np.sum(q_vec * v, axis=1, keepdims=True)
    c = q_vec * dot_product * 2.0
    return (a - b + c).reshape(shape)


# ---------------------------------------------------------------------------
# OpenHLMRootVel (from simple.datasets.jaka_lerobot)
# ---------------------------------------------------------------------------
class OpenHLMRootVel:
    """Legacy roll/pitch and wrapped yaw difference on the caller's clock.

    Kept for the legacy online backup; the alpha argument does not apply EMA.
    """

    def __init__(self, yaw_vel_ema_alpha: float = 0.1):
        from scipy.spatial.transform import Rotation as _R

        self._R = _R
        self._alpha = float(yaw_vel_ema_alpha)
        self.reset()

    def reset(self) -> None:
        self._prev_raw_yaw: float | None = None
        self._prev_ts: int | None = None
        self._yaw_vel_ema: float | None = None

    def __call__(self, quat_wxyz: np.ndarray, timestamp_ns: int) -> np.ndarray:
        w, x, y, z = (float(quat_wxyz[i]) for i in range(4))
        roll, pitch, raw_yaw = self._R.from_quat([x, y, z, w]).as_euler("xyz", degrees=False)
        if self._prev_raw_yaw is None:
            self._prev_ts = timestamp_ns
            self._prev_raw_yaw = raw_yaw
            self._yaw_vel_ema = 0.0
        else:
            dt = (timestamp_ns - self._prev_ts) * 1e-9 if self._prev_ts is not None else 0.0
            dyaw = (raw_yaw - self._prev_raw_yaw + np.pi) % (2 * np.pi) - np.pi  # 去掉 ±π 跳变
            self._yaw_vel_ema = dyaw / dt if dt > 1e-6 else 0.0
            self._prev_ts = timestamp_ns
            self._prev_raw_yaw = raw_yaw
        return np.array([roll, pitch, self._yaw_vel_ema], dtype=np.float32)


class RecordedPoseDifferentiator:
    """Causal derivatives of adjacent selected recording rows on one clock.

    Linear velocity is expressed in the ending row's reference body frame.
    This estimates the recorded latest/hold trajectory, not sensor-time motion.
    Source timestamps (including zero, regressions and clock switches) do not
    participate. No filtering or clipping hides the observed pose increments.
    """

    def __init__(self):
        from scipy.spatial.transform import Rotation
        self._rotation = Rotation
        self.reset()

    def reset(self):
        self._previous = None
        self.last_dt_s = None

    def _rpy(self, quat):
        q = np.asarray(quat, dtype=np.float64)
        if q.shape != (4,) or not np.isfinite(q).all():
            raise ValueError("Expected finite wxyz quaternion")
        norm = np.linalg.norm(q)
        if norm < 1e-8 or not np.isfinite(norm):
            raise ValueError("Invalid quaternion norm")
        q = q / norm
        return q, self._rotation.from_quat(q[[1, 2, 3, 0]]).as_euler("xyz")

    def update(self, state_quat_wxyz, anchor_pos_w, anchor_quat_wxyz, timestamp_ns):
        if isinstance(timestamp_ns, bool) or not isinstance(timestamp_ns, (int, np.integer)):
            raise ValueError("Recording monotonic timestamp must be an integer")
        stamp = int(timestamp_ns)
        pos = np.asarray(anchor_pos_w, dtype=np.float64)
        if pos.shape != (3,) or not np.isfinite(pos).all():
            raise ValueError("Expected finite reference position")
        quat, action_rpy = self._rpy(anchor_quat_wxyz)
        state_rpy = None if state_quat_wxyz is None else self._rpy(state_quat_wxyz)[1]
        state_root = np.zeros(3, dtype=np.float64)
        if state_rpy is not None:
            state_root[:2] = state_rpy[:2]
        action_root = np.r_[action_rpy[:2], 0.]
        linear = np.zeros(3, dtype=np.float64)
        self.last_dt_s = None
        if self._previous is not None:
            prev_stamp, prev_pos, prev_action_yaw, prev_state_yaw = self._previous
            dt = (stamp - prev_stamp) * 1e-9
            if dt <= 0:
                raise ValueError("Recording monotonic timestamps must strictly increase")
            self.last_dt_s = dt
            action_root[2] = (action_rpy[2] - prev_action_yaw + np.pi) % (2 * np.pi) - np.pi
            action_root[2] /= dt
            if state_rpy is not None and prev_state_yaw is not None:
                state_root[2] = ((state_rpy[2] - prev_state_yaw + np.pi) % (2 * np.pi) - np.pi) / dt
            linear = quat_rotate_inverse_numpy(quat, pos - prev_pos) / dt
        values = (state_root, action_root, linear)
        if any(not np.isfinite(v).all() or np.any(np.abs(v) > np.finfo(np.float32).max) for v in values):
            raise ValueError("Recorded row derivative cannot be represented as finite float32")
        self._previous = (stamp, pos.copy(), action_rpy[2], None if state_rpy is None else state_rpy[2])
        return tuple(v.astype(np.float32) for v in values)


# ---------------------------------------------------------------------------
# openhlm frame / features / reference action (from simple.datasets.jaka_lerobot)
# ---------------------------------------------------------------------------
_OPENHLM_STATE_NAMES = list(JAKA_JOINT_NAMES) + ["root_roll", "root_pitch", "yaw_vel"]
_OPENHLM_ACTION_NAMES = (
    list(JAKA_JOINT_NAMES)
    + ["root_roll", "root_pitch", "yaw_vel"]
    + ["anchor_lin_vel_x", "anchor_lin_vel_y", "anchor_lin_vel_z"]
    + ["anchor_pos_w_x", "anchor_pos_w_y", "anchor_pos_w_z"]  # DEBUG: 原始世界 pos
    + ["anchor_quat_w_w", "anchor_quat_w_x", "anchor_quat_w_y", "anchor_quat_w_z"]  # DEBUG: 原始世界 quat
)


def build_openhlm_features(head_image_shape: Optional[Sequence[int]] = None) -> dict[str, dict]:
    """Feature spec the OpenHLM data loader expects (head image + 30 state + 40 actions)."""
    shape = list(head_image_shape) if head_image_shape is not None else list(OPENHLM_HEAD_IMAGE_SHAPE)
    return {
        "head_image_left": {
            "dtype": "image",
            "shape": shape,
            "names": ["height", "width", "channel"],
        },
        "state": {"dtype": "float32", "shape": (OPENHLM_STATE_DIM,), "names": _OPENHLM_STATE_NAMES},
        "actions": {"dtype": "float32", "shape": (OPENHLM_ACTION_DIM,), "names": _OPENHLM_ACTION_NAMES},
    }


def build_openhlm_frame(head_image: np.ndarray, state, actions) -> dict[str, np.ndarray]:
    """One OpenHLM frame: RAW head image (HWC uint8) + float32 state/actions."""
    return {
        "head_image_left": np.ascontiguousarray(np.asarray(head_image, dtype=np.uint8)),
        "state": np.asarray(state, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32),
    }


def reference_action_live_latest(
    motion_buffer,
    *,
    anchor_name: str = "base_link",
    neutral: Optional[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Live record-time reference action from a latest-frame motion source.

    ``motion_buffer`` exposes ``joint_names`` / ``body_names`` / ``get_latest_frame``.
    Returns ``(ref_joints[27], anchor_quat_wxyz[4], anchor_lin_vel[3], anchor_pos_w[3])``.
    """
    return reference_action_from_frame(
        motion_buffer.get_latest_frame(),
        joint_names=motion_buffer.joint_names,
        body_names=motion_buffer.body_names,
        anchor_name=anchor_name,
        neutral=neutral,
    )


def reference_action_from_frame(
    latest: Optional[LatestMotionFrame],
    *,
    joint_names: Sequence[str],
    body_names: Sequence[str],
    anchor_name: str = "base_link",
    neutral: Optional[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None,
    compute_velocity: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract a reference pose, optionally computing the legacy source-time rate.

    The current recorder passes compute_velocity=False and differentiates
    selected rows with RecordedPoseDifferentiator. The default remains for
    legacy diagnostics/callers. One atomic snapshot avoids action/raw races.
    """
    if neutral is None:
        neutral = (
            np.zeros(len(JAKA_JOINT_NAMES), dtype=np.float32),
            np.array([1, 0, 0, 0], np.float32),
            np.zeros(3, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
        )

    body_names = list(body_names)
    if anchor_name not in body_names:
        raise ValueError(f"Motion schema does not contain anchor {anchor_name!r}")
    anchor_idx = body_names.index(anchor_name)

    if latest is None:
        return neutral

    ref = np.asarray(latest.joint_pos, dtype=np.float32)
    mj_names = list(joint_names)
    if list(mj_names) == list(JAKA_JOINT_NAMES):
        ref_joints = ref.copy().astype(np.float32)
    else:
        ref_joints = ref[[mj_names.index(n) for n in JAKA_JOINT_NAMES]].astype(np.float32)

    ref_anchor_pos_w = np.asarray(latest.body_pos_w[anchor_idx], dtype=np.float32)
    ref_anchor_quat = np.asarray(latest.body_quat_w[anchor_idx], dtype=np.float32)

    ref_anchor_lin_vel = np.zeros(3, dtype=np.float32)
    if compute_velocity and latest.prev_body_pos_w is not None and latest.prev_timestamp_ns is not None:
        pos_prev = np.asarray(latest.prev_body_pos_w[anchor_idx], dtype=np.float32)
        dt = (latest.timestamp_ns - latest.prev_timestamp_ns) * 1e-9
        if dt > 1e-6:
            vel_body = quat_rotate_inverse_numpy(
                ref_anchor_quat[None, :], (ref_anchor_pos_w - pos_prev)[None, :]
            )[0]
            ref_anchor_lin_vel = np.clip(vel_body / np.float32(dt), -2.0, 2.0).astype(np.float32)
    return ref_joints, ref_anchor_quat, ref_anchor_lin_vel, ref_anchor_pos_w


# ---------------------------------------------------------------------------
# lerobot exporter (from simple.datasets.jaka_lerobot; lerobot 惰性导入)
# ---------------------------------------------------------------------------
def _patch_lerobot_offline() -> None:
    """Make lerobot's version check work without a HuggingFace Hub connection."""
    import lerobot.datasets.utils as lerobot_utils

    if getattr(lerobot_utils, "_simple_offline_patched", False):
        return

    def _offline_safe_version(repo_id: str, version) -> str:
        import packaging.version

        v = (
            packaging.version.parse(version)
            if not isinstance(version, packaging.version.Version)
            else version
        )
        return f"v{v}"

    lerobot_utils.get_safe_version = _offline_safe_version
    lerobot_utils._simple_offline_patched = True


def _patch_lerobot_save_episode_video_assert() -> None:
    """Tolerate LeRobot's post-save ``*.mp4`` count assert when there are no video keys.

    With ``use_videos=False`` the head image bytes live in the parquet; the browsable
    mp4 under ``videos/`` (written by finalize_openhlm_episode) trips lerobot's own
    count assert from the second episode on — swallow exactly that case and reset the
    episode buffer as the original would, re-raising everything else.
    """
    import logging

    import lerobot.datasets.lerobot_dataset as lerobot_dataset

    if getattr(lerobot_dataset, "_simple_video_assert_patched", False):
        return

    original_save_episode = lerobot_dataset.LeRobotDataset.save_episode
    warned = False

    def save_episode(self, episode_data: dict | None = None) -> None:
        nonlocal warned
        try:
            return original_save_episode(self, episode_data)
        except AssertionError:
            if list(self.meta.video_keys) or len(list(self.root.rglob("*.parquet"))) != self.num_episodes:
                raise  # a real problem: keep LeRobot's own error
            if not warned:
                warned = True
                logging.getLogger(__name__).warning(
                    "save_episode: tolerating LeRobot's '*.mp4'-count sanity assert for the "
                    "rest of this session (use_videos=False, but a browsable mp4 per episode "
                    "lives under videos/); episodes are still fully written (parquet + meta + "
                    "episodes.jsonl + videos/*.mp4)."
                )
            if not episode_data:  # what the original does right after those asserts
                self.episode_buffer = self.create_episode_buffer()
            return None

    lerobot_dataset.LeRobotDataset.save_episode = save_episode
    lerobot_dataset._simple_video_assert_patched = True


def init_openhlm_exporter(
    save_root: str,
    fps: int = 50,
    head_image_shape: Optional[Sequence[int]] = None,
):
    """Create a lerobot LeRobotDataset with use_videos=False (image bytes in parquet)."""
    import os

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    _patch_lerobot_offline()
    _patch_lerobot_save_episode_video_assert()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset.create(
        repo_id="tmp/tmp_dataset",
        fps=fps,
        features=build_openhlm_features(head_image_shape),
        root=save_root,
        use_videos=False,
    )


def finalize_openhlm_episode(exporter) -> Optional[str]:
    """Encode the episode's PNGs to videos/...mp4 and delete them (parquet stays)."""
    import shutil
    from pathlib import Path

    from lerobot.datasets.utils import DEFAULT_VIDEO_PATH
    from lerobot.datasets.video_utils import encode_video_frames

    root = Path(exporter.root)
    episode_index = int(exporter.meta.total_episodes) - 1  # the episode just saved
    chunk = episode_index // int(exporter.meta.chunks_size)
    written = None
    for key in exporter.meta.camera_keys:
        img_dir = exporter._get_image_file_path(episode_index, key, 0).parent
        if not img_dir.is_dir():
            continue
        video_path = root / DEFAULT_VIDEO_PATH.format(
            episode_chunk=chunk, video_key=key, episode_index=episode_index
        )
        video_path.parent.mkdir(parents=True, exist_ok=True)
        encode_video_frames(img_dir, video_path, int(exporter.fps), overwrite=True)
        shutil.rmtree(img_dir)
        parent = img_dir.parent
        while parent != root and parent.is_dir() and not any(parent.iterdir()):
            parent.rmdir()
            parent = parent.parent
        written = str(video_path)
    return written
