"""Jaka tabletop-pick deployment environment for openpi-eval.

Mirrors :class:`sonic_g1_env.SonicG1Env`'s interface (``reset`` / ``step`` /
``get_observation`` / ``close``) for the ``jaka_tabletop_pick`` pi0.5 policy,
whose action space is 33-dimensional.

Action layout produced by ``JakaOutputs`` (policy order, arms first)::

    0-5    left arm   (shoulder pitch/roll/yaw, elbow, wrist roll/yaw)
    6-11   right arm
    12-17  left leg   (hip pitch/roll/yaw, knee, ankle pitch/roll)
    18-23  right leg
    24     waist_yaw
    25-26  neck yaw/pitch
    27-29  anchor: roll (absolute), pitch (absolute), yaw_vel (angular velocity)
    30-32  anchor_lin_vel x/y/z -- **in the ANCHOR BODY FRAME**, not world

Khan real-robot VLA deployments use the reference anchor ``base_link``.
The 33-D action supplies rates. The outgoing wire carries the world pose
integrated from those rates and the configured starting pose, exactly as
``implement_action_analysis/action_trunk_reconstruct.py::reconstruct_anchor_motion``
does offline::

    yaw[i]      = yaw[0] + sum_{k<i} yaw_vel[k] * dt
    quat[i]     = from_rpy(roll[i], pitch[i], yaw[i])
    world_vel[i]= R(quat[i]) . lin_vel_body[i]          # body frame -> world
    pos[i+1]    = pos[i] + world_vel[i] * dt

Note the ordering: each published frame is a snapshot of the pose at the *start* of
its interval, so the quat is built from the accumulated yaw **before** this frame's
``yaw_vel`` is applied. See ``JAKA_RECORDED_DATA_REFERENCE.md`` section 4.

The SuperZmq C++ receiver expects binary protocol v1 with 27 joints in JAKA
order (legs first), the world pose of base_link and a stream_id. The same
source frame keeps its frame_index when it appears in overlapping packets.
"""

import json
import struct
import time
import uuid
from collections import deque
from pathlib import Path

import numpy as np
import zmq
from scipy.spatial.transform import Rotation as R

from sonic_g1_env import _euler_xyz_to_quat_wxyz, pack_pose_message


# ---------------------------------------------------------------------------
# Joint order: policy (saved / OpenPI, arms first) <-> SIM/MuJoCo (legs first)
# ---------------------------------------------------------------------------

# Policy order, dims 0..26 of the 33-dim action. Names taken from the merged
# dataset's meta/info.json ``observation.state`` feature.
POLICY_JOINT_NAMES = [
    # left arm (6)
    "Left_shoulder_pitch_joint", "Left_shoulder_roll_joint", "Left_shoulder_yaw_joint",
    "Left_elbow_joint", "Left_wrist_roll_joint", "Left_wrist_yaw_joint",
    # right arm (6)
    "Right_shoulder_pitch_joint", "Right_shoulder_roll_joint", "Right_shoulder_yaw_joint",
    "Right_elbow_joint", "Right_wrist_roll_joint", "Right_wrist_yaw_joint",
    # left leg (6)
    "Left_hip_pitch_joint", "Left_hip_roll_joint", "Left_hip_yaw_joint",
    "Left_knee_joint", "Left_ankle_pitch_joint", "Left_ankle_roll_joint",
    # right leg (6)
    "Right_hip_pitch_joint", "Right_hip_roll_joint", "Right_hip_yaw_joint",
    "Right_knee_joint", "Right_ankle_pitch_joint", "Right_ankle_roll_joint",
    # waist (1), neck (2)
    "waist_yaw_joint", "Neck_yaw_joint", "Neck_pitch_joint",
]

# MuJoCo/SIM order required on the wire (legs first) -- the same names, as
# recorded in the raw teleop sessions, reordered by
# ``merge_teleop_jaka_mf.py::reorder``. SuperZmq maps this wire order to its buffer order.
SIM_JOINT_NAMES = [
    # left leg (6)
    "Left_hip_pitch_joint", "Left_hip_roll_joint", "Left_hip_yaw_joint",
    "Left_knee_joint", "Left_ankle_pitch_joint", "Left_ankle_roll_joint",
    # right leg (6)
    "Right_hip_pitch_joint", "Right_hip_roll_joint", "Right_hip_yaw_joint",
    "Right_knee_joint", "Right_ankle_pitch_joint", "Right_ankle_roll_joint",
    # waist (1)
    "waist_yaw_joint",
    # left arm (6)
    "Left_shoulder_pitch_joint", "Left_shoulder_roll_joint", "Left_shoulder_yaw_joint",
    "Left_elbow_joint", "Left_wrist_roll_joint", "Left_wrist_yaw_joint",
    # right arm (6)
    "Right_shoulder_pitch_joint", "Right_shoulder_roll_joint", "Right_shoulder_yaw_joint",
    "Right_elbow_joint", "Right_wrist_roll_joint", "Right_wrist_yaw_joint",
    # neck (2)
    "Neck_yaw_joint", "Neck_pitch_joint",
]


def _build_permutation(policy_names: list, sim_names: list) -> np.ndarray:
    """Permutation ``p`` such that ``sim = policy[p]``, derived from name lists."""
    index = {name: i for i, name in enumerate(policy_names)}
    return np.array([index[name] for name in sim_names], dtype=np.int64)


def _quat_rotate_wxyz(quat_wxyz: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate ``vec`` by ``quat_wxyz`` (active rotation, ``R(q) . v``).

    Counterpart of ``simple.jaka_rl.math.quat_rotate_numpy``, which
    ``reconstruct_anchor_motion`` uses to take the anchor body-frame linear
    velocity into the world frame.
    """
    w, x, y, z = (float(c) for c in quat_wxyz)
    qv = np.array([x, y, z], dtype=np.float64)
    vec = np.asarray(vec, dtype=np.float64)
    t = 2.0 * np.cross(qv, vec)
    return (vec + w * t + np.cross(qv, t)).astype(np.float32)


# ``sim = policy[PERM_POLICY_TO_SIM]`` / ``policy = sim[PERM_SIM_TO_POLICY]``.
# Kept as explicit lists so a drift in either name list is caught by the
# assertion below instead of silently producing a wrong joint order.
PERM_POLICY_TO_SIM = np.array([
    12, 13, 14, 15, 16, 17,   # left leg   <- policy dims 12..17
    18, 19, 20, 21, 22, 23,   # right leg  <- policy dims 18..23
    24,                       # waist_yaw  <- policy dim 24
    0, 1, 2, 3, 4, 5,         # left arm   <- policy dims 0..5
    6, 7, 8, 9, 10, 11,       # right arm  <- policy dims 6..11
    25, 26,                   # neck       <- policy dims 25..26
], dtype=np.int64)
PERM_SIM_TO_POLICY = np.argsort(PERM_POLICY_TO_SIM)

assert np.array_equal(PERM_POLICY_TO_SIM, _build_permutation(POLICY_JOINT_NAMES, SIM_JOINT_NAMES)), (
    "PERM_POLICY_TO_SIM is out of sync with the joint name lists"
)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

HEAD_IMAGE_HEIGHT = 224
HEAD_IMAGE_WIDTH = 224
HEAD_IMAGE_CHANNELS = 3

# 30-dim dataset mean in policy order (weighted over all 47 episodes of the legacy
# teleop_jaka_mf/simple/JakaTabletopPickTeleop-v0/level-0 set). Used only by the
# offline ``JakaDefaultPosePolicyClient`` fake policy.
DEFAULT_STATE_JAKA = np.array([
    -0.4613, -1.3431,  0.1592,  1.0753, -0.6990,  0.3478,   # left arm
    -0.5167, -1.2769,  0.2206,  1.2026, -0.4773,  0.2827,   # right arm
    -0.1980,  0.0250,  0.0745,  0.2314,  0.0699, -0.0092,   # left leg
    -0.1302,  0.0290,  0.1551,  0.2358,  0.1070, -0.0148,   # right leg
     0.0337,  0.3293, -0.0373,                               # waist_yaw, neck yaw/pitch
     0.0044,  0.0277, -0.0065,                               # root roll/pitch, yaw_vel
], dtype=np.float32)

# Dataset means, for reference when overriding --jaka-initial-anchor-* (new 40-episode
# `jaka_pick` set, anchor = base_link):
#   anchor_pos_w  (0.27934, -0.95991, 0.60442)
#   anchor_quat_w (0.62867, 0.02642, -0.00725, 0.76849)
# x/y are arbitrary: the tracker's ``root_pos_diff_b`` is built from anchor position
# *differences*, so the whole trajectory is translation-invariant in x/y. z is NOT --
# ``root_z_mf`` reads the anchor's ABSOLUTE world z -- so the default replicates the
# recorded BASE height (0.604 m; recorded z spans 0.587-0.614, first frames 0.608)
# rather than 0, which would command a 0.6 m drop. The downstream anchor is base_link:
# do NOT add the old waist offset (waist_yaw_Link sits 0.22385 m above base_link), or
# the reference would sit at ~0.83 m and be ~0.22 m too high.
# The yaw seed only sets the heading offset: the receiver re-aligns the reference yaw
# to the live robot on its first frame (`align_quat`).
DEFAULT_INITIAL_ANCHOR_POS = (0.0, 0.0, 0.604)
JAKA_ANCHOR_BODY = "base_link"
DEFAULT_INITIAL_ANCHOR_RPY = (0.0, 0.0, 0.0)


def validate_base_replay_dataset(root: str, *, assume_base_link: bool = False) -> None:
    """Check the source pose semantics before publishing labelled base references."""
    p = Path(root) / "conversion_manifest.json"
    anchor = json.loads(p.read_text()).get("anchor_body_name") if p.is_file() else None
    if anchor == JAKA_ANCHOR_BODY or (anchor is None and assume_base_link):
        return
    if anchor is None:
        raise ValueError("Unknown source anchor: use a Khan base_link conversion_manifest.json or "
                         "explicitly pass --assume-base-link after verifying the original poses")
    raise ValueError(f"Cannot publish {anchor!r} poses as base_link; convert the source poses first")


class JakaTabletopEnv:
    """Deployment environment for ``jaka_tabletop_pick``.

    ``step`` converts a 33-dim policy action into one protocol-v1 frame and
    publishes it (2-frame sliding window) on a ZMQ PUB socket that the
    downstream reference buffer subscribes to.
    """

    def __init__(
        self,
        control_hz: int = 30,
        mock: bool = False,
        motion_zmq_address: str = "*",
        motion_zmq_port: int = 28701,
        state_address: str = "127.0.0.1",
        state_port: int = 28711,
        head_image_address: str = "127.0.0.1",
        head_image_port: int = 28712,
        num_frames_to_send: int = 2,
        initial_anchor_pos: tuple = DEFAULT_INITIAL_ANCHOR_POS,
        initial_anchor_rpy: tuple = DEFAULT_INITIAL_ANCHOR_RPY,
        publish: bool = True,
        observation_max_age_s: float = 0.30,
        observation_start_timeout_s: float = 3.0,
    ):
        """
        Args:
            control_hz: Control frequency in Hz (the publish cadence).
            mock: If True, observations are random and no observation socket is
                connected (the action PUB socket is still bound, so the whole
                downstream stream can be exercised offline).
            motion_zmq_address / motion_zmq_port: Bind address of the action PUB
                socket. SuperZmq connects to the publishing PC on this port.
            state_address / state_port: Source of the 30-dim state. Defaults to
                the robot state publisher (28711).
            head_image_address / head_image_port: Source of the head camera
                image. Defaults to the robot camera publisher port (28712).
            num_frames_to_send: Frames buffered per ZMQ message (2 = sliding
                window ``[i-1, i]``, matching the receiver's dedup contract).
            initial_anchor_pos: Initial base_link world position (metres).
                Only z is semantically load-bearing (it is the tracker's absolute
                ``root_z_mf``); x/y merely fix an arbitrary translation origin.
            initial_anchor_rpy: Anchor initial roll/pitch/yaw in radians (only the
                yaw seed matters -- it is accumulated from ``yaw_vel``).
            publish: If False, no ZMQ socket is bound at all.
        """
        if control_hz != 30 or not 2 <= num_frames_to_send <= 16:
            raise ValueError("Khan base_link expects control_hz=30 and 2..16 frames per sliding window")
        if not all(np.isfinite(x) and x > 0 for x in (observation_max_age_s, observation_start_timeout_s)):
            raise ValueError("Observation freshness and startup timeouts must be positive and finite")
        self._pose_metadata = {"anchor_body": JAKA_ANCHOR_BODY, "joint_order": "jaka", "stream_id": str(uuid.uuid4())}
        for value in (initial_anchor_pos, initial_anchor_rpy):
            if np.shape(value) != (3,) or not np.isfinite(np.asarray(value, dtype=np.float32)).all():
                raise ValueError("Initial anchor position and RPY must each have three finite components")
        self._observation_max_age = observation_max_age_s
        self._observation_start_timeout = observation_start_timeout_s
        self._latest_observations = {}
        self._last_observation_poll = None
        self.mock = mock
        self.control_hz = control_hz
        self._dt = 1.0 / float(control_hz)

        self.action_dim = 33  # 27 joints + root(3) + anchor_lin_vel(3)
        self.state_dim = 30   # 27 joints (policy order) + root(3)

        self._num_frames_to_send = num_frames_to_send
        self._action_frame_buffer = deque(maxlen=num_frames_to_send - 1)

        # One timeline per publisher instance. Each overlapping window carries its stream_id.
        self._frame_index = 0
        self._prev_joint_pos = None

        # Anchor pose. roll/pitch are absolute (taken from the action as-is);
        # only yaw is accumulated from yaw_vel, and the position is integrated
        # from anchor_lin_vel. Both survive reset() so the reference trajectory
        # does not jump between episodes.
        self._accumulated_yaw = float(initial_anchor_rpy[2])
        self._body_pos_w = np.asarray(initial_anchor_pos, dtype=np.float32).copy()

        self._zmq_context = zmq.Context()
        if publish:
            self._zmq_socket = self._zmq_context.socket(zmq.PUB)
            self._zmq_socket.bind(f"tcp://{motion_zmq_address}:{motion_zmq_port}")
            print(f"Jaka motion ZMQ PUB bound on tcp://{motion_zmq_address}:{motion_zmq_port}")
        else:
            self._zmq_socket = None

        # Protocols shared by the real robot's StatePublisher and cam_pub.py
        # and SIMPLE's simulated JakaTeleopZmqPublisher:
        #
        #   state  (:28711, ``state_zmq_bind``) -- raw JSON, NO topic prefix:
        #     {"publish_t_ns": i64, "smplx_t_ns": i64, "paused": bool, "seq": i64,
        #      "joint_pos": (27,) f32  in JAKA/MuJoCo order,
        #      "body_pos_w": (nb, 3) f32, "body_quat_w": (nb, 4) wxyz,
        #      "qpos": (nq,) f32}
        #     ``body_quat_w[0]`` is ``base_link`` in BOTH body orderings, which is
        #     what the recorded ``state[27:30]`` (roll/pitch/yaw_vel) is built from.
        #
        #   camera (:28712, ``camera_zmq_bind``) -- raw bytes, NO topic prefix:
        #     struct.pack("iii", w, h, c) + HWC uint8 RGB payload
        #
        # Both sockets SUBSCRIBE to b"" (everything) and use CONFLATE=1: the
        # publisher is latest-only, so a slow consumer must drop stale frames
        # rather than queue them.
        self._state_sub = self._zmq_context.socket(zmq.SUB)
        self._state_sub.setsockopt(zmq.SUBSCRIBE, b"")
        self._state_sub.setsockopt(zmq.CONFLATE, 1)
        self._head_sub = self._zmq_context.socket(zmq.SUB)
        self._head_sub.setsockopt(zmq.SUBSCRIBE, b"")
        self._head_sub.setsockopt(zmq.CONFLATE, 1)
        if not self.mock:
            self._state_sub.connect(f"tcp://{state_address}:{state_port}")
            self._head_sub.connect(f"tcp://{head_image_address}:{head_image_port}")

        self._last_state_publish_ns = None
        self._previous_observed_yaw = None  # (local monotonic time, yaw) of selected state

    # ------------------------------------------------------------------
    # Action path
    # ------------------------------------------------------------------

    def step(self, action):
        """Execute one 33-dim policy action and publish it to the motion stream."""
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (self.action_dim,):
            raise ValueError(f"Action shape must be ({self.action_dim},), got {action.shape}")
        if not np.isfinite(action).all():
            raise ValueError("JAKA action contains nonfinite values")

        # 1) Reorder the 27 joints from policy order (arms first) to SIM order (legs first).
        joint_pos = action[0:27][PERM_POLICY_TO_SIM]

        # 2) Joint velocities by finite difference (parsed then discarded downstream;
        #    kept for wire parity with SonicG1Env).
        if self._prev_joint_pos is None:
            joint_vel = np.zeros(27, dtype=np.float32)
        else:
            joint_vel = ((joint_pos - self._prev_joint_pos) / self._dt).astype(np.float32)

        # 3) Anchor orientation: roll/pitch are absolute, yaw accumulates yaw_vel.
        #    The quat is built from the yaw BEFORE this frame's yaw_vel is applied --
        #    the published frame is the pose at the START of its interval (see the
        #    module docstring / reconstruct_anchor_motion).
        roll = float(action[27])
        pitch = float(action[28])
        yaw_vel = float(action[29])
        body_quat_w = _euler_xyz_to_quat_wxyz(roll, pitch, self._accumulated_yaw)

        # 4) Anchor position: anchor_lin_vel is a velocity in the ANCHOR BODY frame, so
        #    it must go through the anchor orientation before being integrated.
        world_vel = _quat_rotate_wxyz(body_quat_w, action[30:33])
        next_pos = (self._body_pos_w + world_vel * self._dt).astype(np.float32)
        next_yaw = self._accumulated_yaw + yaw_vel * self._dt
        if not np.isfinite(joint_vel).all() or not np.isfinite(next_pos).all() or not np.isfinite(next_yaw):
            raise ValueError("JAKA action produced a nonfinite velocity or integrated pose")

        # 5) Buffer into the sliding window and publish once it is full.
        self._send_motion_zmq(joint_pos, joint_vel, self._body_pos_w.copy(), body_quat_w)

        # 6) Advance the anchor state for the NEXT frame.
        self._prev_joint_pos = joint_pos.copy()
        self._body_pos_w = next_pos
        self._accumulated_yaw = next_yaw

    def _send_motion_zmq(self, joint_pos, joint_vel, body_pos_w, body_quat_w) -> None:
        """Append one frame to the window and publish it once the window is full."""
        frame = {
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "body_pos_w": body_pos_w,
            "body_quat_w": body_quat_w,
            "frame_index": np.int64(self._frame_index),
        }
        if self._zmq_socket is not None and len(self._action_frame_buffer) + 1 == self._num_frames_to_send:
            window = [*self._action_frame_buffer, frame]
            arrays = {key: np.stack([item[key] for item in window]) for key in frame}
            self._zmq_socket.send(pack_pose_message(arrays, metadata=self._pose_metadata))

        # Commit the frame only after send() succeeds, so a retry reproduces the same overlap.
        self._action_frame_buffer.append(frame)
        self._frame_index += 1

    # ------------------------------------------------------------------
    # Observation path
    # ------------------------------------------------------------------

    def reset(self, default_pose=None):
        """Clear the send window and return a fresh observation.

        The same publisher retains its stream ID, frame indices and integrated
        pose. A new publisher instance starts a separate trajectory.

        Args:
            default_pose: Accepted for interface parity with SonicG1Env; unused.
        """
        self._action_frame_buffer.clear()
        self._prev_joint_pos = None
        self._previous_observed_yaw = None
        return self.get_observation()

    def _poll_observation(self):
        """Non-blocking reads of the state (:28711) and camera (:28712) channels."""
        head = None
        try:
            buf = self._head_sub.recv(zmq.NOBLOCK)
            if len(buf) >= 12:
                w, h, c = struct.unpack("iii", buf[:12])
                payload = buf[12:]
                if w > 0 and h > 0 and len(payload) == w * h * c and c in (3, 4):
                    img = np.frombuffer(payload, dtype=np.uint8).reshape((h, w, c))
                    head = img[..., :3] if c == 4 else img  # RGB(A) -> RGB
                else:
                    print(
                        f"Error reading jaka head image: header says {w}x{h}x{c} "
                        f"but payload is {len(payload)} bytes"
                    )
            else:
                print(f"Error reading jaka head image: short frame ({len(buf)} bytes)")
        except zmq.Again:
            pass
        except Exception as e:  # noqa: BLE001
            print(f"Error reading jaka head image: {e}")

        state = None
        try:
            raw = self._state_sub.recv(zmq.NOBLOCK)
            m = json.loads(raw.decode("utf-8"))

            joints_sim = np.asarray(m["joint_pos"], dtype=np.float32)
            if joints_sim.shape != (27,) or not np.isfinite(joints_sim).all():
                raise ValueError("jaka state requires 27 finite joint positions")

            # base_link is body 0 in both orderings; roll/pitch are absolute, and
            # yaw_vel is differentiated here (see __init__).
            quat_wxyz = np.asarray(m["body_quat_w"][0], dtype=np.float64)
            n = float(np.linalg.norm(quat_wxyz))
            if quat_wxyz.shape != (4,) or not np.isfinite(n) or n < 1e-6:
                raise ValueError("jaka state body_quat_w[0] is degenerate")
            quat_wxyz /= n

            t_ns = int(m.get("publish_t_ns") or m.get("smplx_t_ns") or 0)
            if t_ns <= 0 or (self._last_state_publish_ns is not None and t_ns <= self._last_state_publish_ns):
                raise ValueError("base_link state requires an advancing publisher timestamp")
            roll, pitch, raw_yaw = R.from_quat(
                [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
            ).as_euler("xyz")

            self._last_state_publish_ns = t_ns
            state = (joints_sim[PERM_SIM_TO_POLICY], roll, pitch, raw_yaw)
        except zmq.Again:
            pass
        except Exception as e:  # noqa: BLE001
            print(f"Error reading jaka state: {e}")

        return head, state

    def _fresh_base_observation(self):
        """Use local receipt ages for independently arriving camera and state samples."""
        now = time.monotonic()
        if self._last_observation_poll is None or now - self._last_observation_poll > self._observation_max_age:
            # CONFLATE may retain a frame indefinitely during a pause in polling.
            # Discard pending packets once, then wait for both channels to deliver.
            self._poll_observation()
            self._latest_observations.clear()
            # Keep the publisher timestamp watermark across gaps in local polling.
            self._previous_observed_yaw = None
        wait = self._observation_start_timeout if self._last_observation_poll is None else self._observation_max_age
        deadline = now + wait
        keys = ("head_image_left", "state")
        while True:
            pair = self._poll_observation()
            now = self._last_observation_poll = time.monotonic()
            for key, value in zip(keys, pair, strict=True):
                if value is not None:
                    self._latest_observations[key] = (value, now)
            stale = [key for key in keys if key not in self._latest_observations
                     or now - self._latest_observations[key][1] > self._observation_max_age]
            if not stale:
                joints, roll, pitch, yaw = self._latest_observations["state"][0]
                yaw_rate = 0.0
                if self._previous_observed_yaw is not None:
                    prev_t, prev_yaw = self._previous_observed_yaw
                    dt = now - prev_t
                    if dt > 0:
                        yaw_rate = ((yaw - prev_yaw + np.pi) % (2 * np.pi) - np.pi) / dt
                self._previous_observed_yaw = (now, yaw)
                state = np.concatenate((joints, np.asarray([roll, pitch, yaw_rate], dtype=np.float32)))
                return {"head_image_left": self._latest_observations["head_image_left"][0], "state": state}
            if now >= deadline:
                raise TimeoutError(f"No fresh {' and '.join(stale)} for JAKA base_link inference")
            time.sleep(min(0.002, deadline - now))

    def get_observation(self):
        """Return ``{"head_image_left": (H, W, 3) uint8 RGB, "state": (30,) float32}``.

        Only one camera: ``JakaInputs`` masks the wrist views out and synthesizes
        black placeholders server-side, so they must not be sent as request keys.
        """
        if self.mock:
            head_image_left = np.random.randint(
                0, 256,
                (HEAD_IMAGE_HEIGHT, HEAD_IMAGE_WIDTH, HEAD_IMAGE_CHANNELS),
                dtype=np.uint8,
            )
            state = np.random.uniform(-1.0, 1.0, size=(self.state_dim,)).astype(np.float32)
        else:
            return self._fresh_base_observation()

        return {"head_image_left": head_image_left, "state": state}

    def close(self):
        """Close sockets without flushing stale reference frames on shutdown."""
        for attr in ("_zmq_socket", "_state_sub", "_head_sub"):
            sock = getattr(self, attr, None)
            if sock is not None:
                sock.close(linger=0)
        if getattr(self, "_zmq_context", None) is not None:
            self._zmq_context.term()
            self._zmq_context = None
