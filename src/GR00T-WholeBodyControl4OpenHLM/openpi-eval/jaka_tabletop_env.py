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

The anchor fields describe the reference anchor (``waist_yaw_Link`` -- the body the
tracker's ``root_pos_diff_b`` / ``anchor_ori`` are built around), which is also the
frame the reference joints are commanded in. The world pose is not carried on the
wire: it is re-integrated here from the per-frame velocity + orientation, exactly as
``implement_action_analysis/action_trunk_reconstruct.py::reconstruct_anchor_motion``
does offline::

    yaw[i]      = yaw[0] + sum_{k<i} yaw_vel[k] * dt
    quat[i]     = from_rpy(roll[i], pitch[i], yaw[i])
    world_vel[i]= R(quat[i]) . lin_vel_body[i]          # body frame -> world
    pos[i+1]    = pos[i] + world_vel[i] * dt

Note the ordering: each published frame is a snapshot of the pose at the *start* of
its interval, so the quat is built from the accumulated yaw **before** this frame's
``yaw_vel`` is applied. See ``JAKA_RECORDED_DATA_REFERENCE.md`` section 4.

The downstream reference buffer (``RealtimeMotionBufferVla``, from
``implement_action_analysis/src/simple/jaka_rl/motion_buffer.py``) subscribes to
binary protocol v1 over ZMQ and expects the 27 joints in MuJoCo/SIM order (legs
first), the anchor (``waist_yaw_Link``) world pose, and a strictly monotonic
``frame_index`` -- see ``implementation_analysis.md`` section 5.3.
"""

import json
import struct
import time
from collections import defaultdict, deque

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
# ``merge_teleop_jaka_mf.py::reorder``. The receiver does NOT reorder.
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

# 30-dim dataset mean in policy order (weighted over all 47 episodes of
# teleop_jaka_mf/simple/JakaTabletopPickTeleop-v0/level-0). Used only by the
# offline ``JakaDefaultPosePolicyClient`` fake policy.
DEFAULT_STATE_JAKA = np.array([
    -0.4613, -1.3431,  0.1592,  1.0753, -0.6990,  0.3478,   # left arm
    -0.5167, -1.2769,  0.2206,  1.2026, -0.4773,  0.2827,   # right arm
    -0.1980,  0.0250,  0.0745,  0.2314,  0.0699, -0.0092,   # left leg
    -0.1302,  0.0290,  0.1551,  0.2358,  0.1070, -0.0148,   # right leg
     0.0337,  0.3293, -0.0373,                               # waist_yaw, neck yaw/pitch
     0.0044,  0.0277, -0.0065,                               # root roll/pitch, yaw_vel
], dtype=np.float32)

# Dataset means, for reference when overriding --jaka-initial-anchor-*:
#   anchor_pos_w  (12.53212, 4.91023, 0.82769)  [pico/GMR world of the recording]
#   anchor_quat_w (0.80923, 0.01067, 0.01590, -0.05691)
# x/y are arbitrary: the tracker's ``root_pos_diff_b`` is built from anchor position
# *differences*, so the whole trajectory is translation-invariant in x/y. z is NOT --
# ``root_z_mf`` reads the anchor's ABSOLUTE world z -- so the default replicates the
# recorded anchor height (~0.83 m, the standing waist height) rather than 0, which
# would command a 0.83 m drop. The receiver's own empty-buffer fallback posture puts
# the anchor at ~0.874 (motion_buffer.py `_init_default_posture`).
# The yaw seed only sets the heading offset: the receiver re-aligns the reference yaw
# to the live robot on its first frame (`align_quat`, motion_buffer.py:327).
DEFAULT_INITIAL_ANCHOR_POS = (0.0, 0.0, 0.83)
DEFAULT_INITIAL_ANCHOR_RPY = (0.0, 0.0, 0.0)


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
    ):
        """
        Args:
            control_hz: Control frequency in Hz (the publish cadence).
            mock: If True, observations are random and no observation socket is
                connected (the action PUB socket is still bound, so the whole
                downstream stream can be exercised offline).
            motion_zmq_address / motion_zmq_port: Bind address of the action PUB
                socket. The receiver connects to ``tcp://127.0.0.1:28701``.
            state_address / state_port: Source of the 30-dim state. Defaults to
                SIMPLE's ``JakaTeleopZmqPublisher.state_zmq_bind`` (28711).
            head_image_address / head_image_port: Source of the head camera
                image. Defaults to SIMPLE's ``camera_zmq_bind`` (28712).
            num_frames_to_send: Frames buffered per ZMQ message (2 = sliding
                window ``[i-1, i]``, matching the receiver's dedup contract).
            initial_anchor_pos: Anchor (``waist_yaw_Link``) initial world position.
                Only z is semantically load-bearing (it is the tracker's absolute
                ``root_z_mf``); x/y merely fix an arbitrary translation origin.
            initial_anchor_rpy: Anchor initial roll/pitch/yaw in radians (only the
                yaw seed matters -- it is accumulated from ``yaw_vel``).
            publish: If False, no ZMQ socket is bound at all.
        """
        self.mock = mock
        self.control_hz = control_hz
        self._dt = 1.0 / float(control_hz)

        self.action_dim = 33  # 27 joints + root(3) + anchor_lin_vel(3)
        self.state_dim = 30   # 27 joints (policy order) + root(3)

        self._num_frames_to_send = num_frames_to_send
        self._action_frame_buffer = defaultdict(lambda: deque(maxlen=num_frames_to_send))

        # frame_index is monotonic for the lifetime of the process: the receiver
        # keeps its dedup baseline across ``clear()``, so resetting it here would
        # make every subsequent frame silently dropped downstream.
        self._frame_index = 0
        self._prev_joint_pos = None

        # Anchor pose. roll/pitch are absolute (taken from the action as-is);
        # only yaw is accumulated from yaw_vel, and the position is integrated
        # from anchor_lin_vel. Both survive reset() so the reference trajectory
        # does not jump between episodes.
        self._initial_roll = float(initial_anchor_rpy[0])
        self._initial_pitch = float(initial_anchor_rpy[1])
        self._accumulated_yaw = float(initial_anchor_rpy[2])
        self._body_pos_w = np.asarray(initial_anchor_pos, dtype=np.float32).copy()

        self._zmq_context = zmq.Context()
        if publish:
            self._zmq_socket = self._zmq_context.socket(zmq.PUB)
            self._zmq_socket.bind(f"tcp://{motion_zmq_address}:{motion_zmq_port}")
            print(f"Jaka motion ZMQ PUB bound on tcp://{motion_zmq_address}:{motion_zmq_port}")
        else:
            self._zmq_socket = None

        # Observation channels -- these match SIMPLE's ``JakaTeleopZmqPublisher``
        # (``jaka_zmq_pub.py``) verbatim, which is what the teleop/recorder process
        # already publishes while the Jaka robot runs under it:
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

        # ``yaw_vel`` is not carried on the wire: it is differentiated here from
        # ``base_link``'s world yaw, exactly like the recording side's
        # ``OpenHLMRootVel`` (wrapped difference / wall-clock delta; 0.0 on the
        # first frame). State lives here because it is a per-stream derivative.
        self._prev_raw_yaw: float | None = None
        self._prev_state_t_ns: int | None = None

    # ------------------------------------------------------------------
    # Action path
    # ------------------------------------------------------------------

    def step(self, action):
        """Execute one 33-dim policy action and publish it to the motion stream."""
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (self.action_dim,):
            raise ValueError(f"Action shape must be ({self.action_dim},), got {action.shape}")

        # 1) Reorder the 27 joints from policy order (arms first) to SIM order (legs first).
        joint_pos = action[0:27][PERM_POLICY_TO_SIM]

        # 2) Joint velocities by finite difference (parsed then discarded downstream;
        #    kept for wire parity with SonicG1Env).
        if self._prev_joint_pos is None:
            joint_vel = np.zeros(27, dtype=np.float32)
        else:
            joint_vel = ((joint_pos - self._prev_joint_pos) / self._dt).astype(np.float32)
        self._prev_joint_pos = joint_pos.copy()

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

        # 5) Buffer into the sliding window and publish once it is full.
        self._send_motion_zmq(joint_pos, joint_vel, self._body_pos_w.copy(), body_quat_w)

        # 6) Advance the anchor state for the NEXT frame.
        self._body_pos_w = (self._body_pos_w + world_vel * self._dt).astype(np.float32)
        self._accumulated_yaw += yaw_vel * self._dt

    def _send_motion_zmq(self, joint_pos, joint_vel, body_pos_w, body_quat_w) -> None:
        """Append one frame to the window and publish it once the window is full."""
        buf = self._action_frame_buffer
        buf["joint_pos"].append(np.asarray(joint_pos, dtype=np.float32))
        buf["joint_vel"].append(np.asarray(joint_vel, dtype=np.float32))
        buf["body_pos_w"].append(np.asarray(body_pos_w, dtype=np.float32))
        buf["body_quat_w"].append(np.asarray(body_quat_w, dtype=np.float32))
        buf["frame_index"].append(self._frame_index)
        self._frame_index += 1

        if len(buf["frame_index"]) < self._num_frames_to_send:
            return  # first step only fills the window

        numpy_data = {
            "joint_pos": np.stack(buf["joint_pos"], axis=0),              # (N, 27) f32
            "joint_vel": np.stack(buf["joint_vel"], axis=0),              # (N, 27) f32
            "body_pos_w": np.stack(buf["body_pos_w"], axis=0),            # (N, 3)  f32
            "body_quat_w": np.stack(buf["body_quat_w"], axis=0),          # (N, 4)  f32
            "frame_index": np.array(buf["frame_index"], dtype=np.int64),  # (N,)    i64
        }

        if self._zmq_socket is not None:
            try:
                self._zmq_socket.send(pack_pose_message(numpy_data, topic="pose", version=1))
            except Exception as e:  # noqa: BLE001 - keep the control loop alive
                print(f"Error sending motion via ZMQ: {e}")

    # ------------------------------------------------------------------
    # Observation path
    # ------------------------------------------------------------------

    def reset(self, default_pose=None):
        """Clear the 2-frame window and return a fresh observation.

        ``_frame_index``, ``_accumulated_yaw`` and ``_body_pos_w`` are deliberately
        kept: the receiver's dedup baseline survives its own ``clear()`` (a reset
        frame_index would be dropped forever), and the anchor trajectory must stay
        continuous across episodes.

        Args:
            default_pose: Accepted for interface parity with SonicG1Env; unused.
        """
        self._action_frame_buffer.clear()
        self._prev_joint_pos = None
        return self.get_observation()

    def _poll_observation(self):
        """Non-blocking drain of SIMPLE's state (:28711) and camera (:28712) channels."""
        head = None
        try:
            buf = self._head_sub.recv(zmq.NOBLOCK)
            if len(buf) >= 12:
                w, h, c = struct.unpack("iii", buf[:12])
                payload = buf[12:]
                if len(payload) == w * h * c and c in (3, 4):
                    img = np.frombuffer(payload, dtype=np.uint8).reshape((h, w, c))
                    head = img[..., :3] if c == 4 else img  # BGRA -> RGB
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
            if joints_sim.shape != (27,):
                raise ValueError(f"jaka state joint_pos must be (27,), got {joints_sim.shape}")

            # base_link is body 0 in both orderings; roll/pitch are absolute, and
            # yaw_vel is differentiated here (see __init__).
            quat_wxyz = np.asarray(m["body_quat_w"][0], dtype=np.float64)
            n = float(np.linalg.norm(quat_wxyz))
            if n < 1e-6:
                raise ValueError("jaka state body_quat_w[0] is degenerate")
            quat_wxyz /= n

            t_ns = int(m.get("publish_t_ns") or m.get("smplx_t_ns") or 0)
            roll, pitch, raw_yaw = R.from_quat(
                [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
            ).as_euler("xyz")

            yaw_vel = 0.0
            if self._prev_raw_yaw is not None and self._prev_state_t_ns is not None:
                dt = (t_ns - self._prev_state_t_ns) * 1e-9
                if dt > 0.0:
                    d_yaw = (raw_yaw - self._prev_raw_yaw + np.pi) % (2.0 * np.pi) - np.pi
                    yaw_vel = float(d_yaw / dt)
            self._prev_raw_yaw = float(raw_yaw)
            self._prev_state_t_ns = t_ns

            state = np.concatenate([
                joints_sim[PERM_SIM_TO_POLICY],
                np.array([float(roll), float(pitch), yaw_vel], dtype=np.float32),
            ])
        except zmq.Again:
            pass
        except Exception as e:  # noqa: BLE001
            print(f"Error reading jaka state: {e}")

        return head, state

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
            head, state = self._poll_observation()
            if head is None:
                head = np.zeros(
                    (HEAD_IMAGE_HEIGHT, HEAD_IMAGE_WIDTH, HEAD_IMAGE_CHANNELS), dtype=np.uint8
                )
            if state is None:
                state = np.zeros(self.state_dim, dtype=np.float32)
            head_image_left = np.asarray(head, dtype=np.uint8)
            state = np.asarray(state, dtype=np.float32)

        return {"head_image_left": head_image_left, "state": state}

    def close(self):
        """Close sockets and terminate the ZMQ context."""
        for attr in ("_zmq_socket", "_state_sub", "_head_sub"):
            sock = getattr(self, attr, None)
            if sock is not None:
                sock.close()
        if getattr(self, "_zmq_context", None) is not None:
            self._zmq_context.term()
            self._zmq_context = None
