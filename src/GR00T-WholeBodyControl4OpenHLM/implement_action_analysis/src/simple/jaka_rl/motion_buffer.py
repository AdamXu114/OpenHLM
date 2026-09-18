"""Realtime motion buffer that subscribes to the pico retarget motion stream.

Ported from sim2real-jaka ``rl_policy/utils/motion_buffer.py``. Subscribes to
a ZMQ PUB socket (``tcp://127.0.0.1:28701``) carrying JSON payloads from
``pico_retarget_pub``, buffers them by timestamp, and interpolates the
requested ``future_steps`` window on ``get_obs()``.

The only change vs sim2real is the default-posture source: instead of
resolving the MJCF path through a ``RobotCfg``, the buffer accepts explicit
``joint_names`` / ``body_names`` and an optional pre-built ``mj_model`` /
``mj_data`` (when provided, no FK is repeated).
"""

from __future__ import annotations

import json
import threading
import time
from bisect import bisect_right
from dataclasses import dataclass
from typing import Any, Iterable, Optional

import numpy as np
import mujoco
import zmq

from loguru import logger

from simple.jaka_rl.config import ANCHOR_BODY, TOGGLE_DATA_COLLECTION_KEY
from simple.jaka_rl.math import (
    quat_conjugate,
    quat_mul,
    quat_normalize,
    quat_slerp_batch,
    yaw_from_quat,
    yaw_quat,
)
from simple.jaka_rl.motion import MotionData

_normalize_quat_batch = quat_normalize
_quat_slerp_batch = quat_slerp_batch


def _ensure_np(value: Any, ndim: int, dtype=np.float32) -> np.ndarray:
    arr = np.asarray(value, dtype=dtype)
    if arr.ndim != ndim:
        raise ValueError(f"Expected ndim={ndim}, got shape={arr.shape}")
    return arr


@dataclass
class LatestMotionFrame:
    """The most recent reference motion frame, for record-time ("now") sampling.

    Unlike :meth:`RealtimeMotionBuffer.get_obs`, this carries **no** control-loop
    look-back (``_delay_ns``): it is the reference *as it is right now*, so it
    sits on the same wall-clock basis as the live robot state/camera it is
    recorded against. This mirrors OpenHLM's "latest pico frame" semantics
    (``joint_pos[-1]``). ``prev_body_pos_w`` / ``prev_timestamp_ns`` expose the
    frame immediately before the latest so a reference linear velocity can be
    computed from the raw stream.

    ``body_quat_w`` is world-frame **wxyz** (MuJoCo ``xquat`` order).
    """

    joint_pos: np.ndarray  # (num_joints,), JAKA/robot order
    body_pos_w: np.ndarray  # (num_bodies, 3)
    body_quat_w: np.ndarray  # (num_bodies, 4), wxyz
    timestamp_ns: int
    prev_body_pos_w: Optional[np.ndarray]  # (num_bodies, 3) or None
    prev_timestamp_ns: Optional[int]  # or None


class RealtimeMotionBuffer:
    """Timestamped motion buffer with interpolation over a future window."""

    def __init__(
        self,
        joint_names: Iterable[str],
        body_names: Iterable[str],
        future_steps: Iterable[int],
        *,
        mj_model: mujoco.MjModel | None = None,
        mj_data: mujoco.MjData | None = None,
        default_qpos: np.ndarray | None = None,
        motion_zmq_connect: str | None = "tcp://127.0.0.1:28701",
        motion_zmq_hwm: int = 1,
        dt_s: float = 0.02,
        tolerance_s: float = 0.04,
        align_first_frame_yaw: bool = True,
    ):
        if dt_s <= 0.0:
            raise ValueError("dt_s must be positive")
        self.joint_names: list[str] = list(joint_names)
        self.body_names: list[str] = list(body_names)
        self._num_joints = len(self.joint_names)
        self._num_bodies = len(self.body_names)
        self.future_steps = np.asarray(list(future_steps), dtype=int)
        if self.future_steps.ndim != 1:
            raise ValueError(f"future_steps must be 1D, got {self.future_steps.shape}")
        self.dt_s = float(dt_s)
        self.tolerance_s = float(tolerance_s)
        self.min_future_step = (
            int(np.min(self.future_steps)) if self.future_steps.size else 0
        )
        self.max_future_step = (
            int(np.max(self.future_steps)) if self.future_steps.size else 0
        )
        self._dt_ns = int(self.dt_s * 1e9)
        self._tolerance_ns = int(self.tolerance_s * 1e9)
        self._future_steps_ns = self.future_steps.astype(np.int64, copy=False) * self._dt_ns
        self._delay_ns = self.max_future_step * self._dt_ns + self._tolerance_ns
        self._history_ns = self._delay_ns + abs(self.min_future_step) * self._dt_ns
        self.delay_s = float(self._delay_ns / 1e9)
        self._identity_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        self._lock = threading.Lock()
        self._timestamps_ns: list[int] = []
        self._joint_pos_frames: list[np.ndarray] = []
        self._body_pos_w_frames: list[np.ndarray] = []
        self._body_quat_w_frames: list[np.ndarray] = []
        # Latest ``toggle_data_collection`` level seen on the stream (see
        # :attr:`latest_toggle_data_collection`); None until the first payload.
        self._latest_toggle_data_collection: bool | None = None

        # ── Stream diagnostics (see :attr:`diagnostics`) ──────────────────
        # All are cheap counters refreshed by the ingest thread / get_obs; they
        # exist to tell a *live* pico stream apart from a replay-like one, since
        # the policy only sees this buffer.
        self._diag_payloads = 0            # payloads accepted
        self._diag_ts_regressions = 0      # payload ts older than the newest -> bisect insert
        self._diag_last_payload_ts_ns: int | None = None
        self._diag_last_recv_ns: int | None = None   # local wall clock at receipt
        self._diag_last_smplx_ts_ns: int | None = None  # the payload's raw smplx_t_ns
        self._diag_last_dt_ns: int | None = None     # ts gap vs the previous payload
        self._diag_max_jump_m = 0.0        # largest per-frame body translation jump (all time)
        # Rolling (2 s bucket) max jump, so a one-off transition (e.g. the idle
        # stand -> live motion frame) does not mask jumps that recur while tracking.
        self._jump_bucket_m = 0.0
        self._jump_bucket_prev_m = 0.0
        self._jump_bucket_t0 = time.monotonic()
        self._diag_clamped_low = False     # get_obs: window starts before the oldest frame
        self._diag_clamped_high = False    # get_obs: window ends after the newest frame
        self._diag_window_frames = 0       # distinct buffered frames the window touched
        self._diag_buffered = 0
        self._collapsed_warned = False
        self._collapsed_warned_at = 0.0

        # ── First-frame yaw alignment (see _compute_align_quat) ───────────
        # Mirrors the C++ deployment (doc/RealtimeMotionBuffer.cpp +
        # doc/FSMMimicJakaMiniZmq.cpp): ``align_quat`` is (re)computed whenever the
        # buffer goes empty -> non-empty, and handed to the observation through
        # MotionData. The live robot model/data the caller already passes is what
        # lets us read the robot's own heading here — read-only, never stepped.
        self._mj_model = mj_model
        self._mj_data = mj_data
        self.align_first_frame_yaw = bool(align_first_frame_yaw)
        self._align_quat: np.ndarray | None = None   # (4,) wxyz, identity until first frame
        self._align_yaw_deg: float | None = None
        self._ready_prev = False                     # buffer non-empty on the previous get_obs
        self._align_warned = False

        self._init_default_posture(
            mj_model=mj_model,
            mj_data=mj_data,
            default_qpos=default_qpos,
        )

        self._motion_id_template = np.zeros((1, self.future_steps.shape[0]), dtype=np.int64)
        self._step_template = self.future_steps.reshape(1, -1)
        self._zmq_context = zmq.Context.instance()
        self._motion_zmq_connect = motion_zmq_connect
        self._motion_zmq_hwm = int(motion_zmq_hwm)
        self._motion_stream_socket = None
        self._motion_stream_thread: threading.Thread | None = None
        self._motion_stream_stop = threading.Event()
        if self._motion_zmq_connect:
            self._start_motion_stream()

    def _init_default_posture(self, *, mj_model, mj_data, default_qpos):
        """Resolve the default posture used when the buffer holds no frames.

        Mirrors sim2real-jaka, which FKs the robot MJCF at the default qpos. The
        caller here hands us its already-compiled ``mj_model`` (instead of a
        ``robot_cfg`` MJCF path), so the same FK runs on a SCRATCH ``MjData``.
        ``mj_data`` is accepted for API compatibility only and is deliberately
        unused: writing ``default_qpos`` into the LIVE ``mj_data`` would re-pose
        the robot from its task spawn (e.g. x=-1.45 in front of a table) to the
        neutral-stand origin (x=0), dropping it into the table.

        A model is required — there is no synthetic-model fallback. The old
        ``mujoco.MjSpec()``-built placeholder model raised on mujoco>=3.3
        (``MjSpec.add_joint`` does not exist), and the ``except`` below turned
        that into a silent all-zero posture.
        """
        if mj_model is None:
            logger.error(
                "RealtimeMotionBuffer was given no mj_model, so the default posture "
                "cannot be FK'd. The empty-buffer fallback (before the first stream "
                "frame, and right after clear()) will be an ALL-ZERO reference "
                "posture — anchor at z=0 instead of the default stand (~0.874)."
            )
            self._set_zero_default_posture()
            return

        try:
            m = mj_model
            d = mujoco.MjData(m)
            # Overwrite the robot slice with the fallback default pose BEFORE FK,
            # so the default posture reflects `default_qpos` (e.g. the static
            # ZMQ_DEFAULT_QPOS stand), not whatever the robot is posed at now
            # (which may be the spawn DEFAULT_QPOS / motion frame0).
            if default_qpos is not None:
                d.qpos[: len(default_qpos)] = np.asarray(default_qpos, dtype=float)
            mujoco.mj_forward(m, d)

            joint_qpos_indices = []
            for joint_name in self.joint_names:
                joint_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                if joint_id < 0:
                    raise ValueError(f"Failed to resolve joint name in MJCF: {joint_name}")
                joint_qpos_indices.append(int(m.jnt_qposadr[joint_id]))

            body_ids = []
            for body_name in self.body_names:
                body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body_name)
                if body_id < 0:
                    raise ValueError(f"Failed to resolve body name in MJCF: {body_name}")
                body_ids.append(int(body_id))

            self._default_joint_pos = np.asarray(
                d.qpos[joint_qpos_indices], dtype=np.float32
            )
            self._default_body_pos_w = np.asarray(d.xpos[body_ids], dtype=np.float32)
            self._default_body_quat_w = np.asarray(d.xquat[body_ids], dtype=np.float32)
            logger.info("Initialized default standing posture via MuJoCo FK successfully.")
        except Exception as exc:
            logger.error(
                f"Failed to initialize the default posture via MuJoCo FK: {exc}. "
                "Falling back to ALL ZEROS — the empty-buffer reference (before the "
                "first stream frame, and right after clear()) will be wrong: anchor "
                "at z=0 instead of the default stand."
            )
            self._set_zero_default_posture()

    def _set_zero_default_posture(self) -> None:
        """Last-resort default posture: zero joints, all-identity body quats.

        Only correct if a zero posture is genuinely what the reference should be;
        normally this means the FK route failed and the empty-buffer fallback is
        wrong (see :meth:`_init_default_posture`).
        """
        self._default_joint_pos = np.zeros(self._num_joints, dtype=np.float32)
        self._default_body_pos_w = np.zeros((self._num_bodies, 3), dtype=np.float32)
        self._default_body_quat_w = np.tile(
            self._identity_quat[None, :], (self._num_bodies, 1)
        )

    def _start_motion_stream(self) -> None:
        import zmq

        if self._motion_stream_thread is not None:
            return

        sock = self._zmq_context.socket(zmq.SUB)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.RCVHWM, self._motion_zmq_hwm)
        sock.setsockopt(zmq.SUBSCRIBE, b"")
        sock.connect(self._motion_zmq_connect)
        self._motion_stream_socket = sock

        def _stream_loop() -> None:
            while not self._motion_stream_stop.is_set():
                try:
                    raw = sock.recv_string(flags=zmq.NOBLOCK)
                except zmq.Again:
                    time.sleep(0.001)
                    continue
                except Exception as exc:
                    logger.warning(f"Motion subscriber error: {exc}")
                    time.sleep(0.01)
                    continue

                try:
                    self.__append_payload(raw)
                except Exception as exc:
                    logger.warning(f"Failed to decode motion payload: {exc}")

        self._motion_stream_thread = threading.Thread(target=_stream_loop, daemon=True)
        self._motion_stream_thread.start()

    # ------------------------------------------------------------------
    # First-frame yaw alignment (C++ deployment parity)
    # ------------------------------------------------------------------
    def _anchor_index(self) -> int | None:
        """Index of the reference anchor (``waist_yaw_Link``) in the served body list."""
        try:
            return self.body_names.index(ANCHOR_BODY)
        except ValueError:
            return None

    def _robot_anchor_quat(self) -> np.ndarray | None:
        """The robot's CURRENT anchor-body world quat, read from the live MjData.

        Equivalent of the C++ ``zmq_robot_quat_from_imu()``: ``waist_yaw_Link``
        carries the ``waist_imu`` site at its origin with identity orientation, so
        the body quat *is* the IMU framequat the observation uses for the robot too.
        Read-only — this MjData belongs to the running simulation, so we never run
        FK on it. Returns None (alignment is then skipped and logged) when the model
        is missing or the values look invalid.
        """
        if self._mj_model is None or self._mj_data is None:
            return None
        try:
            bid = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_BODY, ANCHOR_BODY)
            if bid < 0:
                return None
            quat = np.asarray(self._mj_data.xquat[bid], dtype=np.float64).copy()
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Could not read the robot anchor quat for yaw alignment: {exc}")
            return None
        norm = float(np.linalg.norm(quat))
        if not np.all(np.isfinite(quat)) or norm < 1e-6:
            return None
        return (quat / norm).astype(np.float32)

    def _compute_align_quat(self, ref_quat: np.ndarray | None) -> np.ndarray | None:
        """``align_quat = yaw(robot) * yaw(reference)^-1`` — C++ FSMMimicJakaMiniZmq.cpp:181-186.

        Only the yaw of each side is kept (roll/pitch of the reference are preserved
        untouched, as the C++ does), so this is a pure world-z rotation mapping the
        reference heading onto the robot's heading at the moment it is computed.
        """
        robot_quat = self._robot_anchor_quat()
        if robot_quat is None or ref_quat is None:
            if not self._align_warned:
                self._align_warned = True
                logger.warning(
                    "First-frame yaw alignment unavailable: no readable live robot "
                    "model/data (or no reference anchor) — align_quat stays identity."
                )
            return None
        rob_yaw = yaw_quat(np.asarray(robot_quat, dtype=np.float64))
        ref_yaw = yaw_quat(np.asarray(ref_quat, dtype=np.float64))
        align = quat_mul(
            np.asarray(rob_yaw, dtype=np.float32).reshape(1, 4),
            np.asarray(quat_conjugate(ref_yaw), dtype=np.float32).reshape(1, 4),
        ).reshape(4)
        align = quat_normalize(align.reshape(1, 4)).reshape(4)
        self._align_yaw_deg = float(
            np.rad2deg(np.asarray(yaw_from_quat(align)).reshape(-1)[0])
        )
        return align.astype(np.float32)

    def _update_align_quat(self, first_frame_quat: np.ndarray | None) -> None:
        """(Re)compute ``align_quat`` on the buffer's empty -> non-empty edge.

        Mirrors the two triggers in the C++ FSM (:408-416 on entry, :418-427 on the
        first package arrival): the first call (nothing buffered yet, so the window
        is the FK default posture — same as the C++ entry case) and every transition
        from empty to non-empty, which includes after ``clear()``.
        """
        if not self.align_first_frame_yaw:
            return
        ready = bool(self._timestamps_ns)
        first_call = self._align_quat is None
        if (ready and not self._ready_prev) or first_call:
            print("//////////////align quat//////////////////")
            align = self._compute_align_quat(first_frame_quat)
            if align is not None:
                self._align_quat = align
                logger.info(
                    "align_quat {:+.1f} deg (robot yaw - reference yaw) — computed {}.",
                    self._align_yaw_deg,
                    "on the empty -> non-empty edge"
                    if not first_call
                    else "on the first get_obs",
                )
        self._ready_prev = ready

    def __append_payload(
        self,
        payload: dict[str, Any] | str | bytes,
        recv_time_ns: int | None = None,
    ) -> None:
        import json as _json

        if isinstance(payload, bytes):
            payload = payload.decode("utf-8")
        if isinstance(payload, str):
            payload = _json.loads(payload.strip())
        if not isinstance(payload, dict):
            raise TypeError(f"Unsupported payload type: {type(payload)}")

        # Which clock to buffer on. ``get_obs`` places its window with the LOCAL
        # ``time.time_ns()``, so the stored timestamps must be on that same clock or
        # ``cleanup``/``np.clip`` silently collapse the look-ahead (see
        # :meth:`diagnostics`). ``publish_t_ns`` is the publisher's own wall clock —
        # pico_retarget_pub sets it from ``time.time_ns()`` on this same machine, so
        # it is directly comparable. ``smplx_t_ns`` is the HEADSET's timestamp (and
        # the controller stream's while paused), i.e. a different clock; using it
        # (as the sim2real reference does) is what breaks a live pico stream while
        # a replay-style publisher — which stamps both fields with the wall clock —
        # works fine. Falls back to smplx / arrival time for payloads without it.
        smplx_ts_ns = payload.get("smplx_t_ns")
        timestamp_ns = (
            payload.get("publish_t_ns") or smplx_ts_ns or recv_time_ns or time.time_ns()
        )
        timestamp_ns = int(timestamp_ns)

        # Record-control toggle (level, not edge) — read before the frame
        # validation below so a malformed motion frame still updates it.
        self._update_toggle_data_collection(payload)

        joint_pos = payload.get("joint_pos", payload.get("dof_pos", payload.get("qpos", None)))
        if joint_pos is None:
            raise ValueError("Payload missing joint_pos/dof_pos/qpos")
        joint_pos = _ensure_np(joint_pos, 1)
        if joint_pos.shape[0] >= 7 + self._num_joints and payload.get("joint_pos") is None:
            joint_pos = joint_pos[7 : 7 + self._num_joints]
        if joint_pos.shape[0] != self._num_joints:
            raise ValueError(
                f"Expected {self._num_joints} joint positions, got {joint_pos.shape[0]}"
            )

        body_pos_w = payload.get("body_pos_w", None)
        body_quat_w = payload.get("body_quat_w", None)


        # print("//////////////body_pos_w//////////////////", body_pos_w)
        # print("//////////////body_quat_w//////////////////", body_quat_w)
        if body_pos_w is None or body_quat_w is None:
            raise ValueError("Payload missing body_pos_w/body_quat_w")

        body_pos_w = _ensure_np(body_pos_w, 2)
        body_quat_w = _ensure_np(body_quat_w, 2)
        if body_pos_w.shape[-1] != 3:
            raise ValueError(f"Expected body_pos_w[..., 3], got {body_pos_w.shape}")
        if body_quat_w.shape[-1] != 4:
            raise ValueError(f"Expected body_quat_w[..., 4], got {body_quat_w.shape}")
        if body_pos_w.shape[-2] != self._num_bodies:
            raise ValueError(
                f"Expected {self._num_bodies} body positions, got {body_pos_w.shape[-2]}"
            )
        if body_quat_w.shape[-2] != self._num_bodies:
            raise ValueError(
                f"Expected {self._num_bodies} body quaternions, got {body_quat_w.shape[-2]}"
            )

        joint_pos_frame = joint_pos.astype(np.float32, copy=True)
        body_pos_w_frame = body_pos_w.astype(np.float32, copy=True)
        body_quat_w_frame = _normalize_quat_batch(
            body_quat_w.astype(np.float32, copy=False),
            eps=1e-8,
        ).astype(np.float32, copy=True)

        with self._lock:
            self._diag_payloads += 1
            prev_ts = self._timestamps_ns[-1] if self._timestamps_ns else None
            if prev_ts is not None:
                self._diag_last_dt_ns = timestamp_ns - prev_ts
                if timestamp_ns < prev_ts:
                    # Older than the newest frame: this is the bisect-insert path,
                    # i.e. the stream's clock went backwards (mixed clock domains,
                    # or a paused publisher stamping a foreign clock). The timeline
                    # then interleaves two epochs, which corrupts the window.
                    self._diag_ts_regressions += 1
                if self._body_pos_w_frames:
                    jump = float(
                        np.abs(body_pos_w_frame - self._body_pos_w_frames[-1]).max()
                    )
                    if np.isfinite(jump):
                        self._diag_max_jump_m = max(self._diag_max_jump_m, jump)
                        mono = time.monotonic()
                        if mono - self._jump_bucket_t0 >= 2.0:
                            self._jump_bucket_prev_m = self._jump_bucket_m
                            self._jump_bucket_m = 0.0
                            self._jump_bucket_t0 = mono
                        self._jump_bucket_m = max(self._jump_bucket_m, jump)
            self._diag_last_payload_ts_ns = timestamp_ns
            self._diag_last_recv_ns = time.time_ns()
            self._diag_last_smplx_ts_ns = None if smplx_ts_ns is None else int(smplx_ts_ns)

            if not self._timestamps_ns or timestamp_ns >= self._timestamps_ns[-1]:
                self._timestamps_ns.append(timestamp_ns)
                self._joint_pos_frames.append(joint_pos_frame)
                self._body_pos_w_frames.append(body_pos_w_frame)
                self._body_quat_w_frames.append(body_quat_w_frame)
            else:
                insert_idx = bisect_right(self._timestamps_ns, timestamp_ns)
                self._timestamps_ns.insert(insert_idx, timestamp_ns)
                self._joint_pos_frames.insert(insert_idx, joint_pos_frame)
                self._body_pos_w_frames.insert(insert_idx, body_pos_w_frame)
                self._body_quat_w_frames.insert(insert_idx, body_quat_w_frame)

    @property
    def latest_timestamp_ns(self) -> int | None:
        with self._lock:
            return self._timestamps_ns[-1] if self._timestamps_ns else None

    def diagnostics(self) -> dict[str, Any]:
        """Health of the incoming reference stream — for telling a live pico stream
        apart from a replay-like one. All values are cheap counters, no behaviour change.

        Read as a group:

        ``ts_offset_ms``  buffered timestamp minus the LOCAL wall clock at receipt.
                          Should stay near 0: the buffer must run on this process's
                          clock (it places the window with ``time.time_ns()``).
        ``smplx_offset_ms`` the payload's raw ``smplx_t_ns`` minus that same local
                          clock. A live pico hub stamps the HEADSET's clock here
                          (``body["timeStampNs"]`` via the XRoboToolkit service), so
                          a huge value is expected and harmless *as long as*
                          ``ts_offset_ms`` is ~0 — it just means the two clocks are
                          different domains, which is exactly why ``smplx_t_ns`` must
                          not be used as the buffer's timeline.
        ``clamped``       the requested window extended past the buffered range, i.e.
                          the look-ahead was silently cut.
        ``window_frames`` distinct buffered frames the last window touched. 1 means
                          every future step was the same pose → ``root_pos_diff_b``
                          is 0 → the policy sees no feed-forward at all.
        ``buffered``      frames currently in the buffer (~``_history_ns`` worth).
        ``ts_regressions``payloads that arrived with a timestamp older than the newest
                          one. >0 means the stream's clock went backwards (mixed
                          domains, or a paused publisher stamping another clock) and
                          those frames were inserted mid-timeline.
        ``jump_recent_m`` largest per-frame body translation jump within the last ~2 s
                          (``max_jump_m`` is the all-time one, which a single
                          transition can pin, hiding jumps that recur while tracking).
        """
        with self._lock:
            offset = (
                None
                if self._diag_last_payload_ts_ns is None or self._diag_last_recv_ns is None
                else (self._diag_last_payload_ts_ns - self._diag_last_recv_ns) / 1e6
            )
            smplx_offset = (
                None
                if self._diag_last_smplx_ts_ns is None or self._diag_last_recv_ns is None
                else (self._diag_last_smplx_ts_ns - self._diag_last_recv_ns) / 1e6
            )
            return {
                "payloads": self._diag_payloads,
                "buffered": self._diag_buffered,
                "window_frames": self._diag_window_frames,
                "clamped": self._diag_clamped_low or self._diag_clamped_high,
                "clamped_lo": self._diag_clamped_low,
                "clamped_hi": self._diag_clamped_high,
                "ts_offset_ms": offset,
                "smplx_offset_ms": smplx_offset,
                "last_dt_ms": (
                    None if self._diag_last_dt_ns is None else self._diag_last_dt_ns / 1e6
                ),
                "ts_regressions": self._diag_ts_regressions,
                "max_jump_m": self._diag_max_jump_m,
                "jump_recent_m": max(self._jump_bucket_m, self._jump_bucket_prev_m),
                "delay_ms": self._delay_ns / 1e6,
                "history_ms": self._history_ns / 1e6,
                "aligned": self._align_quat is not None,
                "align_yaw_deg": self._align_yaw_deg,
            }

    def _update_toggle_data_collection(self, payload: dict[str, Any]) -> None:
        """Store the payload's ``toggle_data_collection`` level (if present).

        The publisher (``pico_retarget_pub``) sends ``[bool]`` on every frame: a
        persistent level it flips on the PICO A button. Accepts a list/array/bool.
        """
        raw = payload.get(TOGGLE_DATA_COLLECTION_KEY, None)
        if raw is None:
            return
        if isinstance(raw, (list, tuple, np.ndarray)):
            if len(raw) == 0:
                return
            raw = raw[0]
        with self._lock:
            self._latest_toggle_data_collection = bool(raw)

    @property
    def latest_toggle_data_collection(self) -> bool | None:
        """Latest record-control level from the stream, or ``None`` before the first frame."""
        with self._lock:
            return self._latest_toggle_data_collection

    # ------------------------------------------------------------------
    # Live sessions (pico paused → live)
    # ------------------------------------------------------------------
    def _fill_sample_frames_locked(
        self,
        target_times_ns: np.ndarray,
        joint_pos_out: np.ndarray,
        body_pos_w_out: np.ndarray,
        body_quat_w_out: np.ndarray,
    ) -> None:
        self._diag_buffered = len(self._timestamps_ns)
        if not self._timestamps_ns:
            self._diag_clamped_low = self._diag_clamped_high = True
            self._diag_window_frames = 0
            joint_pos_out[:] = self._default_joint_pos[None, :]
            body_pos_w_out[:] = self._default_body_pos_w[None, :, :]
            body_quat_w_out[:] = self._default_body_quat_w[None, :, :]
            return

        if len(self._timestamps_ns) == 1:
            # One frame replicated across the whole window: the policy gets NO
            # look-ahead (root_pos_diff_b == 0), so it cannot anticipate motion.
            self._diag_clamped_low = self._diag_clamped_high = True
            self._diag_window_frames = 1
            joint_pos_out[:] = self._joint_pos_frames[0]
            body_pos_w_out[:] = self._body_pos_w_frames[0]
            body_quat_w_out[:] = self._body_quat_w_frames[0]
            return

        timestamps_ns = np.asarray(self._timestamps_ns, dtype=np.int64)
        # The window is expressed on the CONSUMER's wall clock while the buffer is
        # keyed on the publisher's timestamp: if those are different clock domains
        # (or the stream stalled), the clip below silently collapses the look-ahead.
        self._diag_clamped_low = bool(target_times_ns.min() < timestamps_ns[0])
        self._diag_clamped_high = bool(target_times_ns.max() > timestamps_ns[-1])
        clamped_times_ns = np.clip(target_times_ns, timestamps_ns[0], timestamps_ns[-1])
        right = np.searchsorted(timestamps_ns, clamped_times_ns, side="right")
        right = np.clip(right, 1, timestamps_ns.shape[0] - 1)
        left = right - 1
        self._diag_window_frames = int(len(np.unique(np.stack([left, right]))))
        t0 = timestamps_ns[left]
        t1 = timestamps_ns[right]
        alpha = np.divide(
            clamped_times_ns - t0,
            t1 - t0,
            out=np.zeros_like(clamped_times_ns, dtype=np.float32),
            where=t1 > t0,
        ).astype(np.float32, copy=False)

        joint_pos_left = np.stack([self._joint_pos_frames[idx] for idx in left], axis=0)
        joint_pos_right = np.stack([self._joint_pos_frames[idx] for idx in right], axis=0)
        alpha_joint = alpha[:, None]
        joint_pos_out[:] = joint_pos_left + alpha_joint * (joint_pos_right - joint_pos_left)

        body_pos_left = np.stack([self._body_pos_w_frames[idx] for idx in left], axis=0)
        body_pos_right = np.stack([self._body_pos_w_frames[idx] for idx in right], axis=0)
        alpha_body = alpha[:, None, None]
        body_pos_w_out[:] = body_pos_left + alpha_body * (body_pos_right - body_pos_left)

        body_quat_left = np.stack([self._body_quat_w_frames[idx] for idx in left], axis=0)
        body_quat_right = np.stack([self._body_quat_w_frames[idx] for idx in right], axis=0)
        body_quat_w_out[:] = _quat_slerp_batch(
            body_quat_left,
            body_quat_right,
            alpha,
            normalize_inputs=False,
            eps=1e-8,
        ).astype(np.float32, copy=False)

    def clear(self) -> None:
        """Drop all buffered frames, so the reference falls back to the default posture.

        With no frames buffered, :meth:`get_obs` fills the window with
        ``_default_joint_pos`` / ``_default_body_*`` (the FK'd ``default_qpos`` stand,
        i.e. ``ZMQ_DEFAULT_QPOS``) — so right after a reset the policy tracks the
        **default joint positions** instead of whatever the reference stream held
        before (or is publishing now). Mirrors the "start from the default pose"
        behaviour of a fresh episode.

        The next payload from the stream re-fills the buffer as usual, and the
        record-control toggle is deliberately left untouched (it is not motion data).
        """
        with self._lock:
            self._timestamps_ns.clear()
            self._joint_pos_frames.clear()
            self._body_pos_w_frames.clear()
            self._body_quat_w_frames.clear()
        # Re-arm the yaw alignment: the next batch of frames is a fresh reference, so
        # the empty -> non-empty edge fires again (the C++ re-aligns the same way).
        # Dropping the value (rather than keeping it) matters for the in-between
        # window: while empty we serve the FK default posture, and it must be aligned
        # by its own yaw (-> yaw(robot)), not by the previous reference's delta.
        self._ready_prev = False
        self._align_quat = None
        self._align_yaw_deg = None

    def _warn_if_lookahead_collapsed(self) -> None:
        """Warn (rate-limited) when the look-ahead window has collapsed to one frame.

        That means every future step resolves to the same pose, so the policy sees
        ``root_pos_diff_b == 0``: it still tracks, but purely reactively, with no
        feed-forward — the robot lags and easily goes unstable instead of failing
        outright. Worth surfacing because nothing else in the pipeline signals it.
        """
        s = self.diagnostics()
        # An empty buffer is a *different* state ("no data yet" — before the publisher
        # starts, or right after clear()), not a collapse: the window then holds the FK
        # default posture by design. Only warn once there ARE frames to interpolate.
        if s["payloads"] == 0 or s["buffered"] == 0 or s["window_frames"] > 1:
            self._collapsed_warned = False
            return
        now = time.monotonic()
        if self._collapsed_warned and now - self._collapsed_warned_at < 5.0:
            return
        self._collapsed_warned, self._collapsed_warned_at = True, now
        logger.warning(
            "Reference look-ahead collapsed: all {} future step(s) resolve to ONE "
            "buffered frame (buffered={}, clamped={}, ts_offset={} ms, "
            "smplx_offset={} ms). The policy now sees root_pos_diff_b == 0 and "
            "cannot anticipate motion. Usual cause: the stream is keyed on a "
            "timestamp from another clock domain, or the publisher stalled.",
            len(self.future_steps), s["buffered"], s["clamped"],
            None if s["ts_offset_ms"] is None else round(s["ts_offset_ms"], 1),
            None if s["smplx_offset_ms"] is None else round(s["smplx_offset_ms"], 1),
        )

    def cleanup(self, cutoff_ns: int) -> None:
        with self._lock:
            while len(self._timestamps_ns) > 1 and self._timestamps_ns[1] < cutoff_ns:
                self._timestamps_ns.pop(0)
                self._joint_pos_frames.pop(0)
                self._body_pos_w_frames.pop(0)
                self._body_quat_w_frames.pop(0)

    def get_obs(self) -> MotionData:
        current_time_ns = time.time_ns()
        num_steps = self.future_steps.shape[0]
        retain_cutoff_ns = current_time_ns - self._history_ns
        self.cleanup(retain_cutoff_ns)

        target_base_ns = current_time_ns - self._delay_ns
        target_times_ns = target_base_ns + self._future_steps_ns

        joint_pos = np.zeros((1, num_steps, self._num_joints), dtype=np.float32)
        joint_vel = np.zeros_like(joint_pos)
        body_pos_w = np.zeros((1, num_steps, self._num_bodies, 3), dtype=np.float32)
        body_lin_vel_w = np.zeros_like(body_pos_w)
        body_quat_w = np.empty((1, num_steps, self._num_bodies, 4), dtype=np.float32)
        body_ang_vel_w = np.zeros((1, num_steps, self._num_bodies, 3), dtype=np.float32)

        with self._lock:
            self._fill_sample_frames_locked(
                target_times_ns,
                joint_pos[0],
                body_pos_w[0],
                body_quat_w[0],
            )

        self._warn_if_lookahead_collapsed()

        # Yaw alignment: (re)computed right here, on the empty -> non-empty edge,
        # exactly like the C++ FSM does around its get_obs() call. The reference
        # sample is this window's first step — the frame the C++ uses
        # (`frames[0].root_quat_w`).
        anchor_idx = self._anchor_index()
        self._update_align_quat(
            None if anchor_idx is None else body_quat_w[0, 0, anchor_idx]
        )
        align_quat = (
            self._identity_quat.reshape(1, 4)
            if self._align_quat is None
            else self._align_quat.reshape(1, 4)
        )

        return MotionData(
            motion_id=self._motion_id_template,
            step=self._step_template,
            timestamps_ns=target_times_ns.reshape(1, -1),
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            body_pos_w=body_pos_w,
            body_lin_vel_w=body_lin_vel_w,
            body_quat_w=body_quat_w,
            body_ang_vel_w=body_ang_vel_w,
            # (1, 4) so MotionData.__getitem__'s ndarray-only filter keeps it and
            # slices the batch dim rather than a component.
            align_quat=align_quat,
        )

    def get_latest_frame(self) -> Optional[LatestMotionFrame]:
        """Return the most recent reference frame (no look-back), or ``None``.

        Record-time counterpart of :meth:`get_obs`: it reads the newest buffered
        frame directly and applies **no** ``_delay_ns`` look-back, so the
        reference is sampled at the present wall-clock instant and therefore
        aligns with the live robot state/camera it is recorded against. Falls
        back to ``None`` (rather than a metadata-only frame) so callers can
        distinguish "no reference data yet" from a real frame.
        """
        with self._lock:
            n = len(self._timestamps_ns)
            if n == 0:
                return None
            return LatestMotionFrame(
                joint_pos=self._joint_pos_frames[-1],
                body_pos_w=self._body_pos_w_frames[-1],
                body_quat_w=self._body_quat_w_frames[-1],
                timestamp_ns=self._timestamps_ns[-1],
                prev_body_pos_w=(self._body_pos_w_frames[-2] if n >= 2 else None),
                prev_timestamp_ns=(self._timestamps_ns[-2] if n >= 2 else None),
            )

    def close(self) -> None:
        self._motion_stream_stop.set()
        if self._motion_stream_socket is not None:
            self._motion_stream_socket.close(0)


# ============================================================================
# RealtimeMotionBufferVla — VLA 部署专用实时参考轨迹缓冲
# ============================================================================
# Python 移植自 C++ ``RealtimeMotionBufferVla`` (KpiDeployReal), 适配
# openpi-eval 客户端 30Hz 二进制下发。与旧 RealtimeMotionBuffer 的差异:
#
#   [1] 双协议入口: 首字节 '{' → JSON(pico 遥操作兼容); 否则 → 二进制 v1
#       (openpi-eval: [topic "pose"][1280B JSON 头][f32/i64 负载],
#        joint_pos/joint_vel/body_pos_w/body_quat_w/frame_index)。
#       二进制流关节已是 SIM/JAKA 序(客户端已重排), 入库不重排。
#   [2] frame_index 去重: 2 帧滑动窗口 [i-1, i] 中重叠帧只插入一次。
#   [3] 时间戳重锚定: VLA 推理停顿期间数据时间轴"暂停", 恢复后新帧重锚为
#       "上帧 + 名义帧周期", 数据轴上不留下停顿空洞(无大间隙插值)。
#   [4] 数据驱动播放时钟: P = min(P + dt, 最新帧时间 - delay); 停顿期 P
#       冻结 → 参考轨迹位级冻结(等价 sonic 的游标钳制), 延迟恒为 120ms。
#   [5] cleanup 基于播放时钟 P(而非墙钟), 停顿期不删除未播放的保留尾巴。
#
# 二进制流只携带 anchor body 的位姿: 入库时只填 anchor 槽位
# (body_names 中的 ANCHOR_BODY), 其余 body 用默认姿态 FK 值填充。
# ============================================================================

_BINARY_HEADER_SIZE = 1280
_BINARY_TOPIC = b"pose"
_BINARY_DTYPE_ELEMS = {
    "f64": 8,
    "i64": 8,
    "f32": 4,
    "i32": 4,
    "i16": 2,
    "f16": 2,
    "i8": 1,
    "u8": 1,
    "bool": 1,
}


def _resolve_default_posture(joint_names, body_names, mj_model, default_qpos):
    """FK 默认站立姿态, 返回 (joint_pos, body_pos_w, body_quat_w)。

    与旧类 ``_init_default_posture`` 同语义(在 SCRATCH MjData 上 FK, 不碰
    实况 mj_data); FK 失败时回退到全零姿态。
    """
    if mj_model is not None:
        try:
            d = mujoco.MjData(mj_model)
            if default_qpos is not None:
                d.qpos[: len(default_qpos)] = np.asarray(default_qpos, dtype=float)
            mujoco.mj_forward(mj_model, d)

            joint_ids = [
                mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in joint_names
            ]
            body_ids = [
                mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, name)
                for name in body_names
            ]
            if any(j < 0 for j in joint_ids) or any(b < 0 for b in body_ids):
                raise ValueError("Failed to resolve joint/body name in MJCF")

            joint_pos = np.asarray(
                [d.qpos[int(mj_model.jnt_qposadr[j])] for j in joint_ids],
                dtype=np.float32,
            )
            body_pos_w = np.asarray([d.xpos[b] for b in body_ids], dtype=np.float32)
            body_quat_w = np.asarray([d.xquat[b] for b in body_ids], dtype=np.float32)
            logger.info("RealtimeMotionBufferVla: default standing posture via MuJoCo FK.")
            return joint_pos, body_pos_w, body_quat_w
        except Exception as exc:  # noqa: BLE001
            logger.error(
                f"RealtimeMotionBufferVla: default posture FK failed: {exc}. "
                "Falling back to ALL ZEROS."
            )

    zero_joint = np.zeros(len(joint_names), dtype=np.float32)
    zero_pos = np.zeros((len(body_names), 3), dtype=np.float32)
    zero_quat = np.tile(
        np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (len(body_names), 1)
    )
    return zero_joint, zero_pos, zero_quat


def _decode_binary_v1(data: bytes):
    """解码一条二进制协议 v1 ZMQ 消息。

    线格式: [可选 topic "pose"][1280B JSON 头(null 填充)][按头中字段顺序拼接的二进制]

    返回 ``(joint_pos (N,J) f32, body_pos_w (N,3) f32, body_quat_w (N,4) f32,
    frame_index (N,) int64)``。消息不合法时抛 ValueError。
    """
    off = 4 if data[:4] == _BINARY_TOPIC else 0
    if len(data) - off < _BINARY_HEADER_SIZE:
        raise ValueError(f"binary message too short: {len(data)} bytes")

    header_bytes = data[off : off + _BINARY_HEADER_SIZE]
    nul = header_bytes.find(b"\x00")
    header_json = header_bytes[: nul if nul >= 0 else _BINARY_HEADER_SIZE].decode("utf-8")
    hdr = json.loads(header_json)
    version = int(hdr.get("v", 0))
    if version != 1:
        raise ValueError(f"unsupported protocol version {version}")
    big_endian = hdr.get("endian", "le") == "be"

    # 按头中字段顺序计算各字段字节偏移
    field_meta: dict[str, tuple[int, list[int]]] = {}
    byte_offset = 0
    for f in hdr["fields"]:
        name = f["name"]
        dtype = f["dtype"]
        elem = _BINARY_DTYPE_ELEMS.get(dtype)
        if elem is None:
            raise ValueError(f"unknown dtype {dtype}")
        shape = [int(d) for d in f.get("shape", [])]
        nelems = int(np.prod(shape)) if shape else 1
        field_meta[name] = (byte_offset, shape)
        byte_offset += nelems * elem
    if byte_offset > len(data) - off - _BINARY_HEADER_SIZE:
        raise ValueError("payload shorter than header declares")

    payload_off = off + _BINARY_HEADER_SIZE

    def read_field(name, np_kind, expect_ndim):
        if name not in field_meta:
            raise ValueError(f"missing field {name}")
        foff, shape = field_meta[name]
        if len(shape) != expect_ndim:
            raise ValueError(f"field {name}: expected {expect_ndim}D, got shape {shape}")
        dt = np.dtype(np_kind).newbyteorder(">" if big_endian else "<")
        arr = np.frombuffer(
            data, dtype=dt, count=int(np.prod(shape)), offset=payload_off + foff
        )
        return arr.reshape(shape)

    # joint_vel / action_hand_* 解析后丢弃(与 C++ RealtimeMotionBufferVla 一致)
    joint_pos = read_field("joint_pos", "f4", 2).astype(np.float32, copy=True)
    body_pos_w = read_field("body_pos_w", "f4", 2).astype(np.float32, copy=True)
    body_quat_w = read_field("body_quat_w", "f4", 2).astype(np.float32, copy=True)
    frame_index = read_field("frame_index", "i8", 1).astype(np.int64, copy=True)
    return joint_pos, body_pos_w, body_quat_w, frame_index


class RealtimeMotionBufferVla:
    """VLA 部署专用实时参考轨迹缓冲(移植自 C++ RealtimeMotionBufferVla)。

    与旧 ``RealtimeMotionBuffer`` 保持相同的消费接口: ``get_obs()`` 返回
    ``MotionData``(5 个未来插值帧, SIM/JAKA 关节序, 多 body 结构),
    ``jaka_frame_stack_mf`` 无需任何改动即可消费。
    """

    def __init__(
        self,
        joint_names: Iterable[str],
        body_names: Iterable[str],
        future_steps: Iterable[int],
        *,
        mj_model: mujoco.MjModel | None = None,
        mj_data: mujoco.MjData | None = None,
        default_qpos: np.ndarray | None = None,
        motion_zmq_connect: str | None = "tcp://127.0.0.1:28701",
        motion_zmq_hwm: int = 1,
        dt_s: float = 0.02,
        tolerance_s: float = 0.04,
        # VLA 适配新增参数(与 C++ 版一致, 可从 motion 配置传入):
        nominal_frame_s: float = 1.0 / 30.0,  # 重锚定名义帧周期(30Hz)
        gap_threshold_s: float = 0.05,        # 到达间隙 > 此值判定为推理停顿
        align_first_frame_yaw: bool = True,
    ):
        if dt_s <= 0.0:
            raise ValueError("dt_s must be positive")
        if nominal_frame_s <= 0.0:
            raise ValueError("nominal_frame_s must be positive")
        if gap_threshold_s <= 0.0:
            raise ValueError("gap_threshold_s must be positive")

        self.joint_names: list[str] = list(joint_names)
        self.body_names: list[str] = list(body_names)
        self._num_joints = len(self.joint_names)
        self._num_bodies = len(self.body_names)
        if self._num_joints <= 0:
            raise ValueError("joint_names must not be empty")

        self.future_steps = np.asarray(list(future_steps), dtype=int)
        if self.future_steps.ndim != 1:
            raise ValueError(f"future_steps must be 1D, got {self.future_steps.shape}")

        self.dt_s = float(dt_s)
        self.tolerance_s = float(tolerance_s)
        self.min_future_step = (
            int(np.min(self.future_steps)) if self.future_steps.size else 0
        )
        self.max_future_step = (
            int(np.max(self.future_steps)) if self.future_steps.size else 0
        )
        self._dt_ns = int(self.dt_s * 1e9)
        self._tolerance_ns = int(self.tolerance_s * 1e9)
        self._nominal_frame_ns = int(nominal_frame_s * 1e9)
        self._gap_threshold_ns = int(gap_threshold_s * 1e9)
        self._future_steps_ns = self.future_steps.astype(np.int64, copy=False) * self._dt_ns
        # 播放延迟: 保证 P + 未来跨度 始终 ≤ 最新帧 (tolerance 为抖动容差)
        self._delay_ns = self.max_future_step * self._dt_ns + self._tolerance_ns
        self._history_ns = self._delay_ns + abs(self.min_future_step) * self._dt_ns
        self.delay_s = float(self._delay_ns / 1e9)
        self._identity_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        self._lock = threading.Lock()
        self._timestamps_ns: list[int] = []          # 数据时间轴(重锚定后)
        self._joint_pos_frames: list[np.ndarray] = []   # (27,) SIM/JAKA 序
        self._body_pos_w_frames: list[np.ndarray] = []  # (num_bodies, 3)
        self._body_quat_w_frames: list[np.ndarray] = []  # (num_bodies, 4) wxyz

        # 播放时钟(仅 policy 线程经 get_obs() 访问; 假设每控制 tick 调用一次)
        self._playback_time_ns = 0
        self._playback_initialized = False

        # ZMQ 接收线程专用
        self._last_arrival_wall_ns = 0  # 上次到达墙钟(仅用于间隙检测)
        self._last_frame_index = -1     # frame_index 去重(仅二进制路径)

        # 首帧 yaw 对齐(空->非空边沿), 与旧类一致; jaka_mf 消费 align_quat
        self._mj_model = mj_model
        self._mj_data = mj_data
        self.align_first_frame_yaw = bool(align_first_frame_yaw)
        self._align_quat: np.ndarray | None = None
        self._align_yaw_deg: float | None = None
        self._ready_prev = False
        self._align_warned = False

        self._default_joint_pos, self._default_body_pos_w, self._default_body_quat_w = (
            _resolve_default_posture(self.joint_names, self.body_names, mj_model, default_qpos)
        )

        self._motion_id_template = np.zeros((1, self.future_steps.shape[0]), dtype=np.int64)
        self._step_template = self.future_steps.reshape(1, -1)

        self._zmq_context = zmq.Context.instance()
        self._motion_zmq_connect = motion_zmq_connect
        self._motion_zmq_hwm = int(motion_zmq_hwm)
        self._motion_stream_socket = None
        self._motion_stream_thread: threading.Thread | None = None
        self._motion_stream_stop = threading.Event()
        if self._motion_zmq_connect:
            self._start_motion_stream()

    # ------------------------------------------------------------------
    # ZMQ 订阅线程(双协议入口)
    # ------------------------------------------------------------------
    def _start_motion_stream(self) -> None:
        if self._motion_stream_thread is not None:
            return

        sock = self._zmq_context.socket(zmq.SUB)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.RCVHWM, self._motion_zmq_hwm)
        sock.setsockopt(zmq.SUBSCRIBE, b"")
        sock.connect(self._motion_zmq_connect)
        self._motion_stream_socket = sock

        def _stream_loop() -> None:
            while not self._motion_stream_stop.is_set():
                try:
                    raw = sock.recv(flags=zmq.NOBLOCK)  # bytes(二进制消息非 UTF-8)
                except zmq.Again:
                    time.sleep(0.001)
                    continue
                except Exception as exc:  # noqa: BLE001
                    logger.warning(f"RealtimeMotionBufferVla subscriber error: {exc}")
                    time.sleep(0.01)
                    continue

                try:
                    if raw and raw[:1] == b"{":
                        self._handle_json_message(raw.decode("utf-8"))
                    else:
                        self._handle_binary_message(raw)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(f"RealtimeMotionBufferVla ingest error: {exc}")

        self._motion_stream_thread = threading.Thread(target=_stream_loop, daemon=True)
        self._motion_stream_thread.start()

    def close(self) -> None:
        self._motion_stream_stop.set()
        if self._motion_stream_socket is not None:
            self._motion_stream_socket.close(0)

    # ------------------------------------------------------------------
    # JSON 协议入口(pico 遥操作兼容; 时间戳同样走重锚定)
    # ------------------------------------------------------------------
    def _handle_json_message(self, raw: str) -> None:
        payload = json.loads(raw.strip())
        if not isinstance(payload, dict):
            raise TypeError(f"Unsupported payload type: {type(payload)}")

        joint_pos = payload.get("joint_pos", payload.get("dof_pos", payload.get("qpos", None)))
        if joint_pos is None:
            raise ValueError("Payload missing joint_pos/dof_pos/qpos")
        joint_pos = _ensure_np(joint_pos, 1)
        if joint_pos.shape[0] >= 7 + self._num_joints and payload.get("joint_pos") is None:
            joint_pos = joint_pos[7 : 7 + self._num_joints]
        if joint_pos.shape[0] != self._num_joints:
            raise ValueError(
                f"Expected {self._num_joints} joint positions, got {joint_pos.shape[0]}"
            )

        body_pos_w = payload.get("body_pos_w", None)
        body_quat_w = payload.get("body_quat_w", None)
        if body_pos_w is None or body_quat_w is None:
            raise ValueError("Payload missing body_pos_w/body_quat_w")
        body_pos_w = _ensure_np(body_pos_w, 2)
        body_quat_w = _ensure_np(body_quat_w, 2)
        if body_pos_w.shape[-1] != 3:
            raise ValueError(f"Expected body_pos_w[..., 3], got {body_pos_w.shape}")
        if body_quat_w.shape[-1] != 4:
            raise ValueError(f"Expected body_quat_w[..., 4], got {body_quat_w.shape}")
        if body_pos_w.shape[-2] != self._num_bodies or body_quat_w.shape[-2] != self._num_bodies:
            raise ValueError(
                f"Expected {self._num_bodies} bodies, got {body_pos_w.shape[-2]}"
            )

        joint_pos_frame = joint_pos.astype(np.float32, copy=True)
        body_pos_w_frame = body_pos_w.astype(np.float32, copy=True)
        body_quat_w_frame = _normalize_quat_batch(
            body_quat_w.astype(np.float32, copy=False), eps=1e-8
        ).astype(np.float32, copy=True)

        with self._lock:
            ts = self._compute_anchor_ts_locked()
            self._insert_frame_locked(ts, joint_pos_frame, body_pos_w_frame, body_quat_w_frame)

    # ------------------------------------------------------------------
    # [修改1] 二进制协议 v1 入口 + [修改2] frame_index 去重
    # ------------------------------------------------------------------
    def _handle_binary_message(self, data: bytes) -> None:
        try:
            joint_pos, body_pos_w, body_quat_w, frame_index = _decode_binary_v1(data)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"RealtimeMotionBufferVla binary decode failed: {exc}")
            return

        n = frame_index.shape[0]
        if n == 0:
            return
        if joint_pos.shape != (n, self._num_joints):
            logger.warning(
                f"RealtimeMotionBufferVla joint dim mismatch: {joint_pos.shape} "
                f"!= {(n, self._num_joints)}"
            )
            return
        if body_pos_w.shape != (n, 3) or body_quat_w.shape != (n, 4):
            logger.warning(
                f"RealtimeMotionBufferVla body dim mismatch: "
                f"{body_pos_w.shape} / {body_quat_w.shape}"
            )
            return

        # 2 帧滑动窗口 [i-1, i]: 只取帧号 > 已见最大帧号的"新帧"
        new_rows = [int(i) for i in range(n) if frame_index[i] > self._last_frame_index]
        if not new_rows:
            return  # 全部为重叠帧 → 静默丢弃
        self._last_frame_index = int(frame_index.max())

        # 二进制流只携带 anchor body 位姿: 只填 anchor 槽位, 其余 body 用默认 FK 姿态
        anchor_idx = self._anchor_index()
        if anchor_idx is None:
            anchor_idx = 0 if self._num_bodies > 0 else -1

        rows: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for row in new_rows:
            bp = self._default_body_pos_w.copy()
            bq = self._default_body_quat_w.copy()
            if anchor_idx >= 0:
                bp[anchor_idx] = body_pos_w[row].astype(np.float32, copy=True)
                bq[anchor_idx] = _normalize_quat_batch(
                    body_quat_w[row][None, :], eps=1e-8
                )[0].astype(np.float32)
            rows.append((joint_pos[row].astype(np.float32, copy=True), bp, bq))

        with self._lock:
            anchor_ns = self._compute_anchor_ts_locked()
            # 最老的新帧放在 anchor - (cnt-1-j)*nominal, 最新一帧恰落在 anchor 上
            for j, (jp, bp, bq) in enumerate(rows):
                ts = anchor_ns - (len(rows) - 1 - j) * self._nominal_frame_ns
                self._insert_frame_locked(ts, jp, bp, bq)

    # ------------------------------------------------------------------
    # [修改3] 时间戳重锚定(调用方必须持有 _lock)
    # ------------------------------------------------------------------
    def _compute_anchor_ts_locked(self) -> int:
        wall = time.time_ns()
        if not self._timestamps_ns:
            anchor = 0  # 数据时间轴原点
        else:
            delta = wall - self._last_arrival_wall_ns
            if delta > self._gap_threshold_ns:
                # 检测到推理停顿: 时间轴暂停, 新帧重锚到上帧 + 名义周期
                anchor = self._timestamps_ns[-1] + self._nominal_frame_ns
            else:
                # 正常连续流: 按到达间隔推进数据时间
                anchor = self._timestamps_ns[-1] + delta
        self._last_arrival_wall_ns = wall
        return anchor

    def _insert_frame_locked(
        self,
        ts_ns: int,
        joint_pos_frame: np.ndarray,
        body_pos_w_frame: np.ndarray,
        body_quat_w_frame: np.ndarray,
    ) -> None:
        if not self._timestamps_ns or ts_ns >= self._timestamps_ns[-1]:
            self._timestamps_ns.append(ts_ns)
            self._joint_pos_frames.append(joint_pos_frame)
            self._body_pos_w_frames.append(body_pos_w_frame)
            self._body_quat_w_frames.append(body_quat_w_frame)
        else:
            idx = bisect_right(self._timestamps_ns, ts_ns)
            self._timestamps_ns.insert(idx, ts_ns)
            self._joint_pos_frames.insert(idx, joint_pos_frame)
            self._body_pos_w_frames.insert(idx, body_pos_w_frame)
            self._body_quat_w_frames.insert(idx, body_quat_w_frame)

    # ------------------------------------------------------------------
    # 首帧 yaw 对齐(与旧类一致: 空->非空边沿重算; jaka_mf 消费 align_quat)
    # ------------------------------------------------------------------
    def _anchor_index(self) -> int | None:
        try:
            return self.body_names.index(ANCHOR_BODY)
        except ValueError:
            return None

    def _robot_anchor_quat(self) -> np.ndarray | None:
        if self._mj_model is None or self._mj_data is None:
            return None
        try:
            bid = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_BODY, ANCHOR_BODY)
            if bid < 0:
                return None
            quat = np.asarray(self._mj_data.xquat[bid], dtype=np.float64).copy()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                f"RealtimeMotionBufferVla: could not read robot anchor quat for "
                f"yaw alignment: {exc}"
            )
            return None
        norm = float(np.linalg.norm(quat))
        if not np.all(np.isfinite(quat)) or norm < 1e-6:
            return None
        return (quat / norm).astype(np.float32)

    def _compute_align_quat(self, ref_quat: np.ndarray | None) -> np.ndarray | None:
        robot_quat = self._robot_anchor_quat()
        if robot_quat is None or ref_quat is None:
            if not self._align_warned:
                self._align_warned = True
                logger.warning(
                    "RealtimeMotionBufferVla: first-frame yaw alignment unavailable — "
                    "align_quat stays identity."
                )
            return None
        rob_yaw = yaw_quat(np.asarray(robot_quat, dtype=np.float64))
        ref_yaw = yaw_quat(np.asarray(ref_quat, dtype=np.float64))
        align = quat_mul(
            np.asarray(rob_yaw, dtype=np.float32).reshape(1, 4),
            np.asarray(quat_conjugate(ref_yaw), dtype=np.float32).reshape(1, 4),
        ).reshape(4)
        align = quat_normalize(align.reshape(1, 4)).reshape(4)
        self._align_yaw_deg = float(np.rad2deg(np.asarray(yaw_from_quat(align)).reshape(-1)[0]))
        return align.astype(np.float32)

    def _update_align_quat(self, first_frame_quat: np.ndarray | None) -> None:
        if not self.align_first_frame_yaw:
            return
        ready = bool(self._timestamps_ns)
        first_call = self._align_quat is None
        if (ready and not self._ready_prev) or first_call:
            align = self._compute_align_quat(first_frame_quat)
            if align is not None:
                self._align_quat = align
                logger.info(
                    "RealtimeMotionBufferVla: align_quat {:+.1f} deg.", self._align_yaw_deg
                )
        self._ready_prev = ready

    # ------------------------------------------------------------------
    # 旧帧清理(始终保留至少 1 帧; cutoff 由播放时钟给出)
    # ------------------------------------------------------------------
    def _cleanup_locked(self, cutoff_ns: int) -> None:
        while len(self._timestamps_ns) > 1 and self._timestamps_ns[1] < cutoff_ns:
            self._timestamps_ns.pop(0)
            self._joint_pos_frames.pop(0)
            self._body_pos_w_frames.pop(0)
            self._body_quat_w_frames.pop(0)

    # ------------------------------------------------------------------
    # 插值采样(时间 clamp + 二分定位 + lerp/slerp, 与旧类一致)
    # ------------------------------------------------------------------
    def _fill_sample_frames_locked(
        self,
        target_times_ns: np.ndarray,
        joint_pos_out: np.ndarray,
        body_pos_w_out: np.ndarray,
        body_quat_w_out: np.ndarray,
    ) -> None:
        if not self._timestamps_ns:
            joint_pos_out[:] = self._default_joint_pos[None, :]
            body_pos_w_out[:] = self._default_body_pos_w[None, :, :]
            body_quat_w_out[:] = self._default_body_quat_w[None, :, :]
            return

        if len(self._timestamps_ns) == 1:
            joint_pos_out[:] = self._joint_pos_frames[0]
            body_pos_w_out[:] = self._body_pos_w_frames[0]
            body_quat_w_out[:] = self._body_quat_w_frames[0]
            return

        timestamps_ns = np.asarray(self._timestamps_ns, dtype=np.int64)
        clamped = np.clip(target_times_ns, timestamps_ns[0], timestamps_ns[-1])
        right = np.searchsorted(timestamps_ns, clamped, side="right")
        right = np.clip(right, 1, timestamps_ns.shape[0] - 1)
        left = right - 1
        t0 = timestamps_ns[left]
        t1 = timestamps_ns[right]
        alpha = np.divide(
            clamped - t0,
            t1 - t0,
            out=np.zeros_like(clamped, dtype=np.float32),
            where=t1 > t0,
        ).astype(np.float32, copy=False)

        jp_left = np.stack([self._joint_pos_frames[idx] for idx in left], axis=0)
        jp_right = np.stack([self._joint_pos_frames[idx] for idx in right], axis=0)
        joint_pos_out[:] = jp_left + alpha[:, None] * (jp_right - jp_left)

        bp_left = np.stack([self._body_pos_w_frames[idx] for idx in left], axis=0)
        bp_right = np.stack([self._body_pos_w_frames[idx] for idx in right], axis=0)
        body_pos_w_out[:] = bp_left + alpha[:, None, None] * (bp_right - bp_left)

        bq_left = np.stack([self._body_quat_w_frames[idx] for idx in left], axis=0)
        bq_right = np.stack([self._body_quat_w_frames[idx] for idx in right], axis=0)
        body_quat_w_out[:] = _quat_slerp_batch(
            bq_left, bq_right, alpha, normalize_inputs=False, eps=1e-8
        ).astype(np.float32, copy=False)

    # ------------------------------------------------------------------
    # 状态查询
    # ------------------------------------------------------------------
    def ready(self) -> bool:
        with self._lock:
            return bool(self._timestamps_ns)

    @property
    def latest_timestamp_ns(self) -> int | None:
        """最新帧时间戳(数据时间轴, 重锚定后; 非墙钟)"""
        with self._lock:
            return self._timestamps_ns[-1] if self._timestamps_ns else None

    def playback_time_ns(self) -> int:
        """当前播放点 P(诊断用; 仅 policy 线程访问)"""
        return self._playback_time_ns

    def stale_ms(self) -> int:
        """距上次到达的墙钟毫秒数(诊断用, 0 = 尚未收到数据)"""
        with self._lock:
            if self._last_arrival_wall_ns == 0:
                return 0
            return (time.time_ns() - self._last_arrival_wall_ns) // 1_000_000

    @property
    def latest_toggle_data_collection(self) -> None:
        """录制电平(遥操作专属)。VLA 二进制流不携带该字段, 恒为 None。

        teleop_jaka_mf 主循环在 ``reset_on_record_end`` 开启时访问此属性
        (``level is not None`` 才触发 reset), 返回 None 即该触发条件天然失效。
        """
        return None

    def clear(self) -> None:
        """清空缓冲: 参考回退到默认姿态, 重装 yaw 对齐与播放时钟。

        ``_last_frame_index`` / ``_last_arrival_wall_ns`` 刻意保留(与 C++
        一致): 客户端帧号单调, clear 后旧帧仍需去重。
        """
        with self._lock:
            self._timestamps_ns.clear()
            self._joint_pos_frames.clear()
            self._body_pos_w_frames.clear()
            self._body_quat_w_frames.clear()
        self._ready_prev = False
        self._align_quat = None
        self._align_yaw_deg = None
        self._playback_time_ns = 0
        self._playback_initialized = False

    # ------------------------------------------------------------------
    # [修改4/5] 数据驱动播放时钟 + 基于播放时钟的 cleanup
    # ------------------------------------------------------------------
    def get_obs(self) -> MotionData:
        num_steps = self.future_steps.shape[0]
        joint_pos = np.zeros((1, num_steps, self._num_joints), dtype=np.float32)
        joint_vel = np.zeros_like(joint_pos)
        body_pos_w = np.zeros((1, num_steps, self._num_bodies, 3), dtype=np.float32)
        body_lin_vel_w = np.zeros_like(body_pos_w)
        body_quat_w = np.empty((1, num_steps, self._num_bodies, 4), dtype=np.float32)
        body_ang_vel_w = np.zeros((1, num_steps, self._num_bodies, 3), dtype=np.float32)

        empty = False
        with self._lock:
            if not self._timestamps_ns:
                empty = True
            else:
                newest = self._timestamps_ns[-1]

        if empty:
            # 尚无数据 → 默认姿态窗口(与旧类一致)
            joint_pos[:] = self._default_joint_pos[None, None, :]
            body_pos_w[:] = self._default_body_pos_w[None, None, :, :]
            body_quat_w[:] = self._default_body_quat_w[None, None, :, :]
            target_times_ns = np.zeros((1, num_steps), dtype=np.int64)
        else:
            # 播放时钟推进: 每 tick +dt, 被"最新帧 - delay"钳制
            if not self._playback_initialized:
                self._playback_time_ns = newest - self._delay_ns
                self._playback_initialized = True
            else:
                self._playback_time_ns += self._dt_ns
                cap = newest - self._delay_ns  # ← VLA 停顿期 newest 冻结 → P 冻结
                if self._playback_time_ns > cap:
                    self._playback_time_ns = cap

            target_times_ns = (self._playback_time_ns + self._future_steps_ns).reshape(1, -1)

            # cleanup 基于播放时钟(而非墙钟): 停顿期 cutoff 冻结, 保留尾巴不被删
            cutoff_ns = self._playback_time_ns - self._history_ns
            with self._lock:
                self._cleanup_locked(cutoff_ns)
                self._fill_sample_frames_locked(
                    target_times_ns[0], joint_pos[0], body_pos_w[0], body_quat_w[0]
                )

        # 首帧 yaw 对齐(空->非空边沿), 与旧类同位置计算
        anchor_idx = self._anchor_index()
        self._update_align_quat(None if anchor_idx is None else body_quat_w[0, 0, anchor_idx])
        align_quat = (
            self._identity_quat.reshape(1, 4)
            if self._align_quat is None
            else self._align_quat.reshape(1, 4)
        )

        return MotionData(
            motion_id=self._motion_id_template,
            step=self._step_template,
            timestamps_ns=target_times_ns.astype(np.int64),
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            body_pos_w=body_pos_w,
            body_lin_vel_w=body_lin_vel_w,
            body_quat_w=body_quat_w,
            body_ang_vel_w=body_ang_vel_w,
            align_quat=align_quat,
        )


__all__ = ["RealtimeMotionBuffer", "RealtimeMotionBufferVla", "LatestMotionFrame"]
