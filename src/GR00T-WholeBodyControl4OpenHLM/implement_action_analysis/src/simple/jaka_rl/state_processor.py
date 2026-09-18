"""State processor for the Jaka MF policy (in-process).

Ported from sim2real-jaka ``rl_policy/utils/state_processor.py`` and adapted
so the low-level state no longer arrives over ZMQ: a caller-supplied
``state_getter`` returns ``(root_quat_w, root_ang_vel_b, joint_pos, joint_vel)``
each step (in practice read straight from the SIMPLE bridge / ``mjData``).

Keeps the same motion-backend management (``npz`` / ``raw_npz`` / ``zmq``) as
the original, so the policy can run live from the pico hub (``zmq``) or
offline from a recorded ``.npz`` for validation/replay.
"""

from __future__ import annotations

import numpy as np
from typing import Any, Dict, Optional

from simple.jaka_rl.motion import MotionData
from simple.jaka_rl.motion_buffer import RealtimeMotionBuffer, RealtimeMotionBufferVla
from simple.jaka_rl.npz_motion import NpzMotionDataset


class StateProcessor:
    """Holds the robot state + the active motion source for the policy.

    Assumes the incoming state follows ``joint_names`` (MuJoCo) order.
    """

    def __init__(
        self,
        joint_names,
        policy_config,
        *,
        body_names=None,
        state_getter=None,
        mj_model=None,
        mj_data=None,
        default_qpos=None,
    ):
        self.joint_names = list(joint_names)
        self.num_dof = len(self.joint_names)

        self.qpos = np.zeros(3 + 4 + self.num_dof)
        self.qvel = np.zeros(3 + 3 + self.num_dof)

        self.root_pos_w = self.qpos[0:3]
        self.root_lin_vel_w = self.qvel[0:3]
        self.root_quat_w = self.qpos[3:7]
        self.root_ang_vel_b = self.qvel[3:6]
        self.joint_pos = self.qpos[7:]
        self.joint_vel = self.qvel[6:]

        # state_getter returns (root_quat_w, root_ang_vel_b, joint_pos, joint_vel)
        # in joint_names order; used by the in-process mode.
        self._state_getter = state_getter
        self._mj_model = mj_model
        self._mj_data = mj_data
        self._default_qpos = default_qpos
        self._body_names = list(body_names) if body_names is not None else None

        self.motion_config: Dict[str, Any] = dict(policy_config.get("motion", {}))
        self.motion_data: Optional[MotionData] = None
        self.motion_backend = "npz"
        self.motion_joint_names: list[str] = []
        self.motion_body_names: list[str] = []
        self.motion_future_steps = np.array([0], dtype=int)
        self.motion_dataset = None
        self.motion_buffer: RealtimeMotionBuffer | RealtimeMotionBufferVla | None = None
        self.motion_t = np.array([0], dtype=int)
        self.motion_ids = np.array([0], dtype=int)
        self.motion_length = 0

        self._init_motion_backend()
        self._load_motion_frame0()

    # ------------------------------------------------------------------
    # Motion backend
    # ------------------------------------------------------------------
    def _init_motion_backend(self) -> None:
        self.motion_future_steps = np.array(
            self.motion_config.get("future_steps", []), dtype=int
        )
        if self.motion_future_steps.ndim != 1:
            raise ValueError(
                f"motion.future_steps must be 1D, got shape={self.motion_future_steps.shape}"
            )

        motion_backend = str(
            self.motion_config.get("motion_backend", "npz")
        ).lower().strip()
        self.motion_config["motion_backend"] = motion_backend
        self.motion_backend = motion_backend

        if motion_backend == "npz":
            self._init_npz_backend()
        elif motion_backend == "raw_npz":
            self._init_raw_npz_backend()
        elif motion_backend == "zmq":
            self.motion_buffer = RealtimeMotionBuffer(
                joint_names=self.joint_names,
                body_names=self._body_names or [],
                future_steps=self.motion_future_steps,
                mj_model=self._mj_model,
                mj_data=self._mj_data,
                default_qpos=self._default_qpos,
                motion_zmq_connect=self.motion_config.get(
                    "motion_zmq_connect", "tcp://127.0.0.1:28701"
                ),
                motion_zmq_hwm=int(self.motion_config.get("motion_zmq_hwm", 1)),
                dt_s=float(self.motion_config.get("motion_dt_s", 0.02)),
                tolerance_s=float(self.motion_config.get("motion_tolerance_s", 0.04)),
            )
            self.motion_joint_names = list(self.motion_buffer.joint_names)
            self.motion_body_names = list(self.motion_buffer.body_names)
            # Fallback: replay the local motion npz (reordered to JAKA order,
            # same as the hub publishes) when the live buffer has no data — so
            # `[`-align + `]`-track stands on the real motion even without a hub.
            self._init_zmq_npz_replay()
        elif motion_backend == "zmq_vla":
            # VLA 部署: openpi-eval 客户端 30Hz 二进制流 → RealtimeMotionBufferVla
            # (双协议入口 + frame_index 去重 + 重锚定 + 数据驱动播放时钟;
            #  无 npz replay 回退 — 空流时参考为默认 FK 站姿)
            self.motion_buffer = RealtimeMotionBufferVla(
                joint_names=self.joint_names,
                body_names=self._body_names or [],
                future_steps=self.motion_future_steps,
                mj_model=self._mj_model,
                mj_data=self._mj_data,
                default_qpos=self._default_qpos,
                motion_zmq_connect=self.motion_config.get(
                    "motion_zmq_connect", "tcp://127.0.0.1:28701"
                ),
                motion_zmq_hwm=int(self.motion_config.get("motion_zmq_hwm", 1)),
                dt_s=float(self.motion_config.get("motion_dt_s", 0.02)),
                tolerance_s=float(self.motion_config.get("motion_tolerance_s", 0.04)),
                nominal_frame_s=float(
                    self.motion_config.get("motion_nominal_frame_s", 1.0 / 30.0)
                ),
                gap_threshold_s=float(
                    self.motion_config.get("motion_gap_threshold_s", 0.05)
                ),
            )
            self.motion_joint_names = list(self.motion_buffer.joint_names)
            self.motion_body_names = list(self.motion_buffer.body_names)
        else:
            raise ValueError(f"Unsupported motion_backend: {motion_backend}")

    def _init_npz_backend(self) -> None:
        raise NotImplementedError(
            "npz backend (any4hdmi) not ported; use 'raw_npz' instead."
        )

    def _load_motion_frame0(self) -> None:
        """Load motion frame 0 (in MuJoCo/JAKA order) from the motion npz.

        Gives ``[`` (align) a *deterministic* reference — the motion's first
        frame — regardless of how far the live zmq buffer has advanced. Falls
        back to None for live streams (no npz) so align uses the current frame.
        """
        # `[` (align) target = the static ZMQ_DEFAULT_QPOS stand pose, so align
        # matches the no-data reference (keeps align+track consistent).
        from simple.jaka_rl.config import ZMQ_DEFAULT_QPOS

        q = np.asarray(ZMQ_DEFAULT_QPOS, dtype=np.float32)
        self.motion_frame0 = {
            "joint_pos": q[7:].copy(),   # [27] JAKA order
            "base_pos": q[0:3].copy(),   # [3]
            "base_quat": q[3:7].copy(),  # [4]
        }

    def _init_zmq_npz_replay(self) -> None:
        """Load the local motion npz (JAKA order) as a live-buffer fallback."""
        self._npz_replay: NpzMotionDataset | None = None
        motion_path = self.motion_config.get("motion_path")
        if not motion_path:
            return
        resolved = _resolve_path(motion_path)
        if not resolved.exists() or resolved.suffix.lower() != ".npz":
            return
        try:
            d = np.load(resolved)
            from simple.jaka_rl.config import BODY_NAMES, JOINT_NAMES, NPZ_BODY_NAMES, NPZ_JOINT_NAMES

            jid = np.array([NPZ_JOINT_NAMES.index(n) for n in JOINT_NAMES], dtype=int)
            bid = np.array([NPZ_BODY_NAMES.index(n) for n in BODY_NAMES], dtype=int)
            # Store JAKA-order arrays so motion_data matches the zmq buffer layout.
            self._npz_replay_data = {
                "joint_pos": d["joint_pos"][:, jid].astype(np.float32),
                "joint_vel": d["joint_vel"][:, jid].astype(np.float32),
                "body_pos_w": d["body_pos_w"][:, bid].astype(np.float32),
                "body_quat_w": d["body_quat_w"][:, bid].astype(np.float32),
                "body_lin_vel_w": d["body_lin_vel_w"][:, bid].astype(np.float32),
                "body_ang_vel_w": d["body_ang_vel_w"][:, bid].astype(np.float32),
            }
            self._npz_replay_len = self._npz_replay_data["joint_pos"].shape[0]
            self._npz_replay_t = np.array([0], dtype=int)
        except Exception as exc:  # noqa: BLE001
            import logging

            logging.getLogger(__name__).warning(f"Failed to load zmq npz replay: {exc}")
            self._npz_replay_len = 0

    def _npz_replay_slice(self):
        t = int(self._npz_replay_t[0])
        idx = np.clip(t + self.motion_future_steps, 0, self._npz_replay_len - 1)
        key = self._npz_replay_data
        return MotionData(
            joint_pos=key["joint_pos"][idx][None],
            joint_vel=key["joint_vel"][idx][None],
            body_pos_w=key["body_pos_w"][idx][None],
            body_lin_vel_w=key["body_lin_vel_w"][idx][None],
            body_quat_w=key["body_quat_w"][idx][None],
            body_ang_vel_w=key["body_ang_vel_w"][idx][None],
        )

    def _init_raw_npz_backend(self) -> None:
        motion_path = self.motion_config.get("motion_path")
        if motion_path is None:
            raise ValueError("motion_path is required for raw_npz motion backend")
        npz_joint_names = self.motion_config.get("npz_joint_names")
        npz_body_names = self.motion_config.get("npz_body_names")
        if npz_joint_names is None or npz_body_names is None:
            raise ValueError(
                "motion.npz_joint_names and motion.npz_body_names are required "
                "for raw_npz backend"
            )
        resolved_path = _resolve_path(motion_path)
        self.npz_dataset = NpzMotionDataset(
            str(resolved_path),
            joint_names=npz_joint_names,
            body_names=npz_body_names,
        )
        self.motion_ids = np.array([0], dtype=int)
        self.motion_t = np.array([0], dtype=int)
        self.motion_length = self.npz_dataset.num_steps
        self.motion_joint_names = list(self.npz_dataset.joint_names)
        self.motion_body_names = list(self.npz_dataset.body_names)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def _using_zmq_replay(self) -> bool:
        """True if we fall back to the local npz replay on an empty live buffer.

        DISABLED: when the zmq stream has no data the reference is the static
        ZMQ_DEFAULT_QPOS (straight-leg stand) from RealtimeMotionBuffer's FK
        default posture, not the dance npz replay.
        """
        return False

    def reset(self):
        if self.motion_backend in ("npz", "raw_npz"):
            self.motion_t[:] = 0
        if self._using_zmq_replay():
            self._npz_replay_t[:] = 0
        self._update_motion_data()

    def update(self, data: Optional[Dict] = None):
        data = data or {}
        paused = data.get("paused", False)
        if not paused and self.motion_backend in ("npz", "raw_npz"):
            self.motion_t += 1
            if self.motion_length > 0:
                if self.motion_t[0] >= self.motion_length:
                    self.motion_t[:] = 0
                    data["paused"] = True
        if not paused and self._using_zmq_replay() and self._npz_replay_len > 0:
            self._npz_replay_t += 1
            if self._npz_replay_t[0] >= self._npz_replay_len:
                self._npz_replay_t[:] = 0
        self._update_motion_data()

    def _update_motion_data(self):
        if self.motion_backend == "raw_npz":
            self.motion_data = self.npz_dataset.get_slice(
                self.motion_ids, self.motion_t, self.motion_future_steps
            )
        elif self.motion_backend in ("zmq", "zmq_vla"):
            if self._using_zmq_replay():
                self.motion_data = self._npz_replay_slice()
            else:
                self.motion_data = self.motion_buffer.get_obs()

    # ------------------------------------------------------------------
    # Low state
    # ------------------------------------------------------------------
    def _prepare_low_state(self) -> bool:
        if self._state_getter is None:
            return False
        root_quat_w, root_ang_vel_b, joint_pos, joint_vel = self._state_getter()
        self.root_quat_w[:] = root_quat_w
        self.root_ang_vel_b[:] = root_ang_vel_b
        self.joint_pos[:] = joint_pos
        self.joint_vel[:] = joint_vel
        return True


def _resolve_path(p: str):
    """Resolve a possibly-project-relative motion path to an absolute path.

    The sim2real policy yaml stores ``motion_path`` relative to the
    *sim2real-jaka* repo root (e.g. ``jaka_data/...``). SIMPLE and
    sim2real-jaka are sibling checkouts, so try, in order: cwd, ``data/motion``
    (by basename), the sibling ``sim2real-jaka``, and ``~/code/sim2real-jaka``.
    """
    from pathlib import Path

    p = Path(p)
    if p.is_absolute():
        return p.expanduser().resolve()
    candidates = [Path.cwd() / p, Path.cwd() / "data" / "motion" / p.name]
    for sibling in (
        Path.cwd().parent / "sim2real-jaka",
        Path.home() / "code" / "sim2real-jaka",
    ):
        candidates.append(sibling / p)
    for c in candidates:
        if c.exists():
            return c.resolve()
    return candidates[0].resolve()


__all__ = ["StateProcessor"]
