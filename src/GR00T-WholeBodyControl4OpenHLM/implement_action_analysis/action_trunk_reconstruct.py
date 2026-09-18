"""Reconstruct the Jaka MF (bottom-tracker) reference observation from an OpenHLM action trunk.

The VLA emits a ``[T, 40]`` action trunk. Each row (one frame) is::

    [ 27 Jaka dof | roll | pitch | yaw_vel | lin_vel_xyz | anchor_pos_w_xyz | anchor_quat_w_wxyz ]

(DEBUG: ``anchor_pos_w`` at 33:36 and ``anchor_quat_w`` at 36:40 are the raw saved anchor pose,
so the verifier can check them against motion_mf directly and also measure how far integrating
``yaw_vel`` drifts over 1 s. The absolute yaw is not stored — it is recoverable from the quat.)

Only the first 33 columns (dof / rpy / yaw_vel / lin_vel) are needed to reconstruct the motion;
the trailing raw-pose columns are truncated here.

The ``roll/pitch/yaw_vel`` and ``lin_vel`` describe the reference ANCHOR (``waist_yaw_Link``
by default) body — the frame the frozen ``jaka_frame_stack_mf`` policy builds its command
(``root_pos_diff_b``) in — so the trunk is fully aligned with the tracker.

The MF policy builds its reference command from ``motion_data.body_pos_w`` / ``body_quat_w``
over a ``future_steps`` window. From the action trunk we only get the anchor pose velocity +
orientation per frame, so these helpers integrate the body-frame linear velocity back to a
world anchor pose trajectory (``anchor_pos``, ``anchor_quat`` wxyz), then produce the pieces
the MF command and ``anchor_ori`` need — all in the anchor frame.

Caveat: the anchor yaw is recovered by integrating the saved ``yaw_vel`` (EMA'd/clipped or raw
depending on ``OpenHLMRootVel``), so it drifts from the original heading; roll/pitch are exact
per frame. Position integration is exact given the per-frame anchor quat (self-tested).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from simple.jaka_rl.math import (
    quat_conjugate,
    quat_mul,
    quat_rotate_inverse_numpy,
    quat_rotate_numpy,
)

DOF = 27
_R_ROLL, _R_PITCH, _R_YAWVEL = 27, 28, 29
_V_START = 30


def quat_from_rpy(roll, pitch, yaw) -> np.ndarray:
    """Roll/pitch/yaw (rad, intrinsic xyz euler) -> wxyz quaternion.

    Accepts scalars (returns ``(4,)``) or broadcastable arrays (returns ``(..., 4)``).
    """
    roll, pitch, yaw = np.broadcast_arrays(
        np.asarray(roll, dtype=np.float32),
        np.asarray(pitch, dtype=np.float32),
        np.asarray(yaw, dtype=np.float32),
    )
    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
    q = np.stack(
        [cr * cp * cy + sr * sp * sy,
         sr * cp * cy - cr * sp * sy,
         cr * sp * cy + sr * cp * sy,
         cr * cp * sy - sr * sp * cy],
        axis=-1,
    )
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    identity = np.array([1.0, 0.0, 0.0, 0.0], dtype=q.dtype)
    return np.where(n > 1e-8, q / n, identity).astype(np.float32)


def reconstruct_anchor_motion(
    actions: np.ndarray,
    *,
    fps: float = 50.0,
    start_pos: Optional[np.ndarray] = None,
    start_yaw: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate a ``[T,>=33]`` action trunk back to the reference anchor world pose trajectory.

    Args:
        actions: ``[T, >=33]`` = [27 dof | roll, pitch, yaw_vel | lin_vel xyz | ...]
            (``yaw_vel`` is the reference anchor yaw rate -- integrated here for the heading;
            cols >=33 are the raw anchor world pose, used only for verification and truncated).
        fps: recording / data fps (default 50 -> dt = 0.02 s).
        start_pos: initial anchor world position ``[3]`` (zeros if omitted). Only shifts the
            absolute position; ``root_pos_diff_b`` is invariant to it.
        start_yaw: initial anchor yaw (0 if omitted), seeds the yaw integration.

    Returns:
        ``(joint_pos[T,27], anchor_pos[T,3], anchor_quat[T,4] wxyz)`` in world frame.
    """
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2:
        actions = actions.reshape(-1, actions.shape[-1])
    if actions.shape[1] < 33:
        raise ValueError(f"expected >=33 action cols, got {actions.shape[1]}")
    # 前 33 列在所有布局里都是 [27 dof | roll,pitch,yaw_vel | lin_vel(30:33)];
    # 40 维新布局后面还有 anchor_pos_w/anchor_quat_w, 这里截掉只用前 33。
    actions = actions[:, :33]
    _v_start = _V_START
    T = actions.shape[0]
    dt = 1.0 / float(fps)

    dof = actions[:, :DOF].astype(np.float32)
    roll = actions[:, _R_ROLL]
    pitch = actions[:, _R_PITCH]
    yaw_vel = actions[:, _R_YAWVEL]
    vel = actions[:, _v_start:_v_start + 3].astype(np.float32)

    if T == 0:
        return dof, np.zeros((0, 3), dtype=np.float32), np.zeros((0, 4), dtype=np.float32)

    # Recover the heading by integrating the saved (EMA'd/clipped) yaw_vel -> approximate yaw.
    yaw = np.asarray(start_yaw, dtype=np.float32) + np.concatenate(
        [np.zeros(1, dtype=np.float32), np.cumsum(yaw_vel[:-1] * dt)]
    )

    # Quats from rpy (vectorized) and world-frame velocities from the body-frame ones.
    anchor_quat = quat_from_rpy(roll, pitch, yaw)   # (T,4)
    world_vel = quat_rotate_numpy(anchor_quat, vel)  # (T,3) world-frame velocity per row

    # Cumulative integration of the per-frame world velocity -> anchor world position.
    anchor_pos = np.zeros((T, 3), dtype=np.float32)
    anchor_pos[0] = np.asarray(start_pos, dtype=np.float32) if start_pos is not None else 0.0
    if T > 1:
        anchor_pos[1:] = anchor_pos[0] + np.cumsum(world_vel[:-1] * dt, axis=0)
    return dof, anchor_pos, anchor_quat


def mf_command(
    dof: np.ndarray,
    anchor_pos: np.ndarray,
    anchor_quat: np.ndarray,
    *,
    num_future: int = 5,
) -> np.ndarray:
    """Build the MF v1 command (155-dim) from reconstructed anchor motion.

    Mirrors ``jaka_frame_stack_mf._compute_command`` for the anchor frame:
    ``[root_pos_diff_b(5*3), root_z_mf(5), ref_joint_pos(5*27)]``.
    """
    cur_quat = anchor_quat[0]
    idx = np.arange(num_future)
    diff_b = quat_rotate_inverse_numpy(
        np.tile(cur_quat[None, :], (num_future, 1)), anchor_pos[idx] - anchor_pos[0:1]
    )
    root_pos_diff_b = diff_b.reshape(-1)          # 15
    root_z_mf = anchor_pos[idx, 2:3].reshape(-1)  # 5
    ref_joint_pos = dof[idx].reshape(-1)          # 27 * num_future
    return np.concatenate([root_pos_diff_b, root_z_mf, ref_joint_pos]).astype(np.float32)


def anchor_ori(
    anchor_quat: np.ndarray,
    robot_quat: np.ndarray,
    ref_to_robot_quat: np.ndarray,
    *,
    num_future: int = 5,
) -> np.ndarray:
    """Build the MF ``anchor_ori`` (rot6d, ``num_future*6``) from reconstructed anchor quats.

    Mirrors ``jaka_frame_stack_mf._compute_anchor_ori``. ``robot_quat`` is the robot's current
    anchor quat and ``ref_to_robot_quat`` the ref->robot yaw alignment, both computed at runtime.
    """
    future_anchor = quat_mul(
        np.tile(ref_to_robot_quat[None, :], (num_future, 1)), anchor_quat[:num_future]
    )
    robot_batch = np.tile(robot_quat[None, :], (num_future, 1))
    ori_b = quat_mul(quat_conjugate(robot_batch), future_anchor)  # [n,4]
    r, i, j, k = ori_b[:, 0], ori_b[:, 1], ori_b[:, 2], ori_b[:, 3]
    norm2 = r * r + i * i + j * j + k * k
    safe = norm2 > 1e-8
    two_s = np.divide(2.0, np.maximum(norm2, 1e-8), out=np.zeros_like(norm2), where=safe)
    ii = i * i; jj = j * j; kk = k * k; ij = i * j; kr = k * r
    ik = i * k; jr = j * r; jk = j * k; ir = i * r
    rot6d = np.stack(
        [1.0 - two_s * (jj + kk), two_s * (ij - kr),
         two_s * (ij + kr), 1.0 - two_s * (ii + kk),
         two_s * (ik - jr), two_s * (jk + ir)], axis=-1
    ).reshape(-1)
    rot6d = np.where(np.isfinite(rot6d), rot6d, np.tile([1, 0, 0, 1, 0, 0], num_future))
    return rot6d.astype(np.float32)


if __name__ == "__main__":
    # Round-trip self-test: build a reference anchor trajectory, encode it as an action
    # trunk (rpy + lin_vel), reconstruct it, and check pose + MF command match.
    fps = 50.0
    dt = 1.0 / fps
    T = 50
    ys = np.arange(T) * 0.03
    ps = 0.02 * np.sin(np.arange(T) * 0.3)
    rs = 0.01 * np.cos(np.arange(T) * 0.2)
    quat = quat_from_rpy(rs, ps, ys)  # (T,4)
    pos = np.zeros((T, 3), dtype=np.float32)
    for i in range(1, T):
        pos[i] = pos[i - 1] + np.array([0.4, 0.0, 0.0], np.float32) * dt
    pos[:, 2] = 0.72
    pos[:, 1] = (0.1 * ys).astype(np.float32)

    yaw_vel = np.full(T, float((ys[1] - ys[0]) / dt), dtype=np.float32)  # rad/s (constant)
    vel = np.zeros((T, 3), dtype=np.float32)
    for i in range(T - 1):
        vel[i] = quat_rotate_inverse_numpy(quat[i][None, :], (pos[i + 1] - pos[i])[None, :])[0] * (1.0 / dt)
    vel[-1] = vel[-2]
    rpy = np.stack([rs, ps, yaw_vel], axis=1)  # [T,3] = [roll, pitch, yaw_vel]
    actions = np.concatenate([np.zeros((T, 27), np.float32), rpy, vel], axis=-1)  # [T,33]

    dof, pos_rec, quat_rec = reconstruct_anchor_motion(actions, fps=fps, start_pos=pos[0], start_yaw=ys[0])
    print("anchor pos reconstruction max err: %.2e m" % np.max(np.abs(pos_rec - pos)))
    print("anchor quat reconstruction max err: %.2e" % np.max(np.abs(quat_rec - quat)))

    cmd_target = mf_command(dof, pos, quat)
    cmd_recon = mf_command(dof, pos_rec, quat_rec)
    print("MF command (155-dim) max err: %.2e" % np.max(np.abs(cmd_target - cmd_recon)))
