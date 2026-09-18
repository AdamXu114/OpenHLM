"""Wire-contract test for JakaTabletopEnv against the downstream decoder.

Validates that what ``JakaTabletopEnv`` publishes can be decoded by the receiver
side (``RealtimeMotionBufferVla``). The authoritative decoder lives in
``implement_action_analysis/src/simple/jaka_rl/motion_buffer.py``; that module
imports mujoco/loguru/the ``simple`` package, so if it cannot be imported here a
byte-equivalent copy of ``_decode_binary_v1`` is used instead (it depends only on
json + numpy). Re-sync the vendored copy if the authoritative one changes.

Run:
    python jaka_tabletop_env_test.py
"""

import json
import os
import sys
import time

import numpy as np
import zmq

# Path setup: the GR00T repo root makes ``gear_sonic`` (pulled in by
# sonic_g1_env) importable when it is not installed in the active venv; the
# level above mirrors main.py's own ``../..`` append for ``openpi_client``.
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.abspath(os.path.join(_HERE, "..")), os.path.abspath(os.path.join(_HERE, "..", ".."))):
    if _p not in sys.path:
        sys.path.append(_p)

from jaka_tabletop_env import (
    PERM_POLICY_TO_SIM,
    PERM_SIM_TO_POLICY,
    JakaTabletopEnv,
)
from sonic_g1_env import _euler_xyz_to_quat_wxyz, pack_pose_message

MOTION_BUFFER_PATH = os.path.abspath(os.path.join(
    _HERE, "..", "implement_action_analysis", "src", "simple", "jaka_rl"
))
RECONSTRUCT_PATH = os.path.abspath(os.path.join(
    _HERE, "..", "implement_action_analysis", "action_trunk_reconstruct.py"
))


# ---------------------------------------------------------------------------
# Decoder: prefer the authoritative implementation, else the vendored copy
# ---------------------------------------------------------------------------

def _load_decoder():
    """Return the authoritative ``_decode_binary_v1`` from motion_buffer.py.

    The module cannot be imported directly outside the SIMPLE-jaka runtime (it
    needs mujoco / loguru / the ``simple`` package), but the decoder itself only
    depends on json + numpy -- so it is extracted from the source with ``ast``
    and executed here. That way the test exercises the real implementation rather
    than a re-typed copy.
    """
    src_path = os.path.join(MOTION_BUFFER_PATH, "motion_buffer.py")
    sys.path.insert(0, MOTION_BUFFER_PATH)
    try:
        import motion_buffer  # type: ignore  # noqa: PLC0415
        return motion_buffer._decode_binary_v1, "imported motion_buffer"
    except Exception as e:  # noqa: BLE001
        print(f"[note] cannot import motion_buffer ({type(e).__name__}: {e}); "
              f"extracting the decoder from source instead")

    import ast

    with open(src_path) as fh:
        tree = ast.parse(fh.read(), filename=src_path)

    wanted_globals = {"_BINARY_DTYPE_ELEMS", "_BINARY_HEADER_SIZE", "_BINARY_TOPIC"}
    namespace = {"json": json, "np": np}
    for node in tree.body:
        is_wanted_assign = (
            isinstance(node, ast.Assign)
            and any(getattr(t, "id", None) in wanted_globals for t in node.targets)
        )
        is_decoder = isinstance(node, ast.FunctionDef) and node.name == "_decode_binary_v1"
        if is_wanted_assign or is_decoder:
            exec(compile(ast.Module(body=[node], type_ignores=[]), src_path, "exec"), namespace)

    if "_decode_binary_v1" not in namespace:
        raise RuntimeError(f"could not locate _decode_binary_v1 in {src_path}")
    return namespace["_decode_binary_v1"], f"extracted from {src_path}"


decode_binary_v1, DECODER_SOURCE = _load_decoder()


def _load_quat_from_rpy():
    """Return the authoritative ``quat_from_rpy`` from action_trunk_reconstruct.py.

    Same trick as :func:`_load_decoder`: the module imports ``simple.jaka_rl.math``,
    which is not part of this analysis checkout, but ``quat_from_rpy`` itself only
    needs numpy. This is the exact rpy -> quat convention the offline reconstruction
    uses, so the live stream must agree with it.
    """
    import ast

    with open(RECONSTRUCT_PATH) as fh:
        tree = ast.parse(fh.read(), filename=RECONSTRUCT_PATH)
    namespace = {"np": np, "__name__": "extracted"}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "quat_from_rpy":
            exec(compile(ast.Module(body=[node], type_ignores=[]), RECONSTRUCT_PATH, "exec"), namespace)
    if "quat_from_rpy" not in namespace:
        raise RuntimeError(f"could not locate quat_from_rpy in {RECONSTRUCT_PATH}")
    return namespace["quat_from_rpy"]


quat_from_rpy = _load_quat_from_rpy()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_permutation():
    """The hard-coded permutation matches the name-derived one and its inverse."""
    assert np.array_equal(PERM_SIM_TO_POLICY, np.argsort(PERM_POLICY_TO_SIM)), \
        "PERM_SIM_TO_POLICY must be argsort(PERM_POLICY_TO_SIM)"
    # spot checks against the documented layout
    assert PERM_POLICY_TO_SIM[0] == 12, "SIM dim 0 (left hip pitch) <- policy dim 12"
    assert PERM_POLICY_TO_SIM[12] == 24, "SIM dim 12 (waist) <- policy dim 24"
    assert PERM_POLICY_TO_SIM[13] == 0, "SIM dim 13 (left shoulder pitch) <- policy dim 0"
    assert PERM_POLICY_TO_SIM[25] == 25, "SIM dim 25 (neck yaw) <- policy dim 25"
    print("PASS test_permutation")


def test_golden_roundtrip():
    """pack_pose_message -> decode_binary_v1 round-trips every field exactly."""
    rng = np.random.default_rng(0)
    data = {
        "joint_pos": rng.standard_normal((2, 27)).astype(np.float32),
        "joint_vel": rng.standard_normal((2, 27)).astype(np.float32),
        "body_pos_w": rng.standard_normal((2, 3)).astype(np.float32),
        "body_quat_w": np.array([[1.0, 0.0, 0.0, 0.0], [0.9, 0.1, 0.2, 0.3]], dtype=np.float32),
        "frame_index": np.array([7, 8], dtype=np.int64),
    }
    joint_pos, body_pos_w, body_quat_w, frame_index = decode_binary_v1(
        pack_pose_message(data, topic="pose", version=1)
    )
    assert np.array_equal(joint_pos, data["joint_pos"]), "joint_pos mismatch"
    assert np.array_equal(body_pos_w, data["body_pos_w"]), "body_pos_w mismatch"
    assert np.array_equal(body_quat_w, data["body_quat_w"]), "body_quat_w mismatch"
    assert np.array_equal(frame_index, data["frame_index"]), "frame_index mismatch"
    assert frame_index.dtype == np.int64, f"frame_index dtype {frame_index.dtype} != int64"
    print("PASS test_golden_roundtrip")


def test_quat_convention():
    """The env's rpy -> quat matches the authoritative offline reconstruction."""
    rng = np.random.default_rng(1)
    for r, p, y in rng.uniform(-1.2, 1.2, size=(25, 3)):
        a = _euler_xyz_to_quat_wxyz(float(r), float(p), float(y))
        b = quat_from_rpy(float(r), float(p), float(y))
        assert np.allclose(a, b, atol=1e-6) or np.allclose(a, -b, atol=1e-6), \
            f"rpy ({r:.4f}, {p:.4f}, {y:.4f}): {a} != {b}"
    print("PASS test_quat_convention")


def test_anchor_roundtrip():
    """A ground-truth anchor trajectory survives trunk -> env.step -> wire.

    Mirrors the self-test in ``action_trunk_reconstruct.py``: derive an action trunk
    from a known ``(pos, quat)`` trajectory, stream it through the env frame by frame,
    and require the published frames to reproduce that trajectory. This is what pins
    the two conventions that are easy to get wrong -- that ``lin_vel`` is expressed in
    the ANCHOR BODY frame, and that a frame carries the pose at the START of its
    interval (quat built before this frame's ``yaw_vel`` is applied).
    """
    from scipy.spatial.transform import Rotation

    fps = 30.0
    dt = 1.0 / fps
    T = 40

    yaw = 0.02 * np.arange(T)
    roll = 0.1 * np.sin(0.1 * np.arange(T))
    pitch = 0.05 * np.cos(0.1 * np.arange(T))
    quat = quat_from_rpy(roll, pitch, yaw)                       # (T,4) wxyz, authoritative

    # Ground-truth world trajectory: choose world velocities, integrate them, and
    # derive the body-frame velocities that must reproduce them.
    world_vel = np.stack([
        0.4 * np.cos(0.1 * np.arange(T)),
        0.2 * np.sin(0.1 * np.arange(T)),
        np.full(T, 0.05),
    ], axis=1)
    pos = np.zeros((T, 3), dtype=np.float64)
    pos[0] = (0.0, 0.0, 0.83)
    for i in range(T - 1):
        pos[i + 1] = pos[i] + world_vel[i] * dt
    pos = pos.astype(np.float32)

    # R(quat)ᵀ . world_vel via scipy -- an independent inverse rotation, not the env's.
    rot = Rotation.from_quat(np.roll(quat, -1, axis=1))          # wxyz -> xyzw
    body_vel = np.stack([rot[i].inv().apply(world_vel[i]) for i in range(T)]).astype(np.float32)
    yaw_vel = np.empty(T, dtype=np.float32)
    yaw_vel[:-1] = np.diff(yaw) / dt
    yaw_vel[-1] = yaw_vel[-2]

    actions = np.concatenate([
        np.zeros((T, 27), dtype=np.float32),
        np.stack([roll, pitch, yaw_vel], axis=1).astype(np.float32),
        body_vel,
    ], axis=1)

    port = 28710
    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.SUBSCRIBE, b"")
    sub.connect(f"tcp://127.0.0.1:{port}")

    env = JakaTabletopEnv(
        mock=True, motion_zmq_address="127.0.0.1", motion_zmq_port=port,
        initial_anchor_pos=tuple(pos[0]), initial_anchor_rpy=(0.0, 0.0, float(yaw[0])),
    )
    time.sleep(0.3)  # PUB slow-joiner
    for a in actions:
        env.step(a)

    msgs = _drain(sub, T - 1)
    assert len(msgs) == T - 1, f"expected {T - 1} messages, got {len(msgs)}"

    pos_err = quat_err = 0.0
    for m in msgs:
        _, body_pos_w, body_quat_w, frame_index = decode_binary_v1(m)
        for row, k in enumerate(frame_index):
            pos_err = max(pos_err, float(np.abs(body_pos_w[row] - pos[k]).max()))
            quat_err = max(quat_err, float(np.abs(body_quat_w[row] - quat[k]).max()))
    assert pos_err < 1e-5, f"anchor position reconstruction error {pos_err:.3e} m"
    assert quat_err < 1e-5, f"anchor quaternion reconstruction error {quat_err:.3e}"

    env.close()
    sub.close()
    ctx.term()
    print(f"PASS test_anchor_roundtrip (pos err {pos_err:.1e} m, quat err {quat_err:.1e})")


def _drain(sub, n, timeout_s=2.0):
    """Collect up to n messages from a SUB socket."""
    msgs = []
    deadline = time.time() + timeout_s
    while len(msgs) < n and time.time() < deadline:
        try:
            msgs.append(sub.recv(zmq.NOBLOCK))
        except zmq.Again:
            time.sleep(0.005)
    return msgs


def test_live_stream():
    """A live PUB/SUB round-trip with the real env: shapes, order, integration."""
    port = 28709
    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.SUBSCRIBE, b"")
    sub.connect(f"tcp://127.0.0.1:{port}")

    env = JakaTabletopEnv(mock=True, motion_zmq_address="127.0.0.1", motion_zmq_port=port)
    time.sleep(0.3)  # PUB slow-joiner

    # A distinct value per dim makes the joint reorder observable.
    base = np.arange(33, dtype=np.float32) / 100.0
    action = base.copy()
    action[27], action[28], action[29] = 0.1, 0.2, 0.3          # roll, pitch, yaw_vel
    action[30:33] = np.array([0.1, 0.2, 0.3], dtype=np.float32)  # anchor_lin_vel

    n_steps = 6
    for _ in range(n_steps):
        env.step(action)

    msgs = _drain(sub, n_steps - 1)
    assert len(msgs) == n_steps - 1, f"expected {n_steps - 1} messages, got {len(msgs)}"

    decoded = [decode_binary_v1(m) for m in msgs]

    for i, (joint_pos, body_pos_w, body_quat_w, frame_index) in enumerate(decoded):
        assert joint_pos.shape == (2, 27), f"msg {i}: joint_pos {joint_pos.shape}"
        assert body_pos_w.shape == (2, 3), f"msg {i}: body_pos_w {body_pos_w.shape}"
        assert body_quat_w.shape == (2, 4), f"msg {i}: body_quat_w {body_quat_w.shape}"
        assert frame_index.shape == (2,), f"msg {i}: frame_index {frame_index.shape}"
        assert frame_index.dtype == np.int64

    # frame_index: sliding window [k, k+1], strictly increasing across messages.
    for i, (_, _, _, frame_index) in enumerate(decoded):
        assert np.array_equal(frame_index, np.array([i, i + 1])), \
            f"msg {i}: frame_index {frame_index} != [{i}, {i + 1}]"

    # Joint reorder: policy order -> SIM order, every frame.
    expected_joints = action[0:27][PERM_POLICY_TO_SIM]
    for i, (joint_pos, _, _, _) in enumerate(decoded):
        for row in range(2):
            assert np.allclose(joint_pos[row], expected_joints, atol=1e-6), \
                f"msg {i} row {row}: joint order mismatch"
    # spot check the two most important slots
    assert PERM_POLICY_TO_SIM[0] == 12 and PERM_POLICY_TO_SIM[13] == 0

    # Anchor orientation: roll/pitch absolute, yaw integrated from yaw_vel. A frame
    # is the pose at the START of its interval, so frame k carries the yaw accumulated
    # over frames 0..k-1 -- this frame's own yaw_vel is not applied yet.
    # scipy is used deliberately: an independent implementation of the same rpy
    # convention, not a re-statement of the env's own arithmetic.
    dt = 1.0 / 30.0
    from scipy.spatial.transform import Rotation

    accel = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    expected_pos = {0: np.array([0.0, 0.0, 0.83])}  # DEFAULT_INITIAL_ANCHOR_POS
    for k in range(1, n_steps + 1):
        expected_pos[k] = expected_pos[k - 1] + Rotation.from_euler(
            "xyz", [0.1, 0.2, 0.3 * dt * (k - 1)]).apply(accel) * dt

    for i, (_, body_pos_w, body_quat_w, frame_index) in enumerate(decoded):
        for row, k in enumerate(frame_index):
            k = int(k)
            q = Rotation.from_euler("xyz", [0.1, 0.2, 0.3 * dt * k])
            assert np.allclose(body_quat_w[row], np.roll(q.as_quat(), 1), atol=1e-5), (
                f"msg {i} row {row}: quat {body_quat_w[row]} != {q.as_quat()} "
                f"(expected yaw {0.3 * dt * k})"
            )
            assert np.isclose(np.linalg.norm(body_quat_w[row]), 1.0, atol=1e-5), "quat not unit"
            # Position is the integral of the WORLD-frame velocity, i.e. the body-frame
            # lin_vel rotated by the frame's own anchor orientation.
            assert np.allclose(body_pos_w[row], expected_pos[k], atol=1e-6), (
                f"msg {i} row {row}: pos {body_pos_w[row]} != {expected_pos[k]}"
            )

    # reset() must NOT restart frame_index (the receiver would drop those frames).
    # It clears the sliding window, so N steps after it publish N-1 messages again.
    env.reset()
    last_index = int(decoded[-1][3].max())
    post_reset_steps = 3
    for _ in range(post_reset_steps):
        env.step(action)
    more = _drain(sub, post_reset_steps - 1)
    assert len(more) == post_reset_steps - 1, \
        f"expected {post_reset_steps - 1} messages after reset, got {len(more)}"
    post = [decode_binary_v1(m)[3] for m in more]
    for i, idx in enumerate(post):
        assert idx[0] < idx[1], f"post-reset msg {i}: frame_index not increasing: {idx}"
    # The window re-sends the previous frame by design (the receiver dedups it), so
    # the monotonicity invariant is on the *newest* index of each message.
    newest = [int(idx.max()) for idx in post]
    assert min(newest) > last_index, \
        f"frame_index went backwards across reset(): {newest} after {last_index}"
    assert newest == sorted(set(newest)), f"duplicate newest frame_index: {newest}"
    # ...and the overlap is exactly one frame: the previous newest is re-sent.
    assert int(post[0].min()) == last_index + 1, \
        f"expected the window to resume at {last_index + 1}, got {int(post[0].min())}"

    env.close()
    sub.close()
    ctx.term()
    print("PASS test_live_stream")


if __name__ == "__main__":
    print(f"decoder under test: {DECODER_SOURCE}")
    test_permutation()
    test_quat_convention()
    test_golden_roundtrip()
    test_anchor_roundtrip()
    test_live_stream()
    print("\nAll jaka wire-contract tests passed.")
