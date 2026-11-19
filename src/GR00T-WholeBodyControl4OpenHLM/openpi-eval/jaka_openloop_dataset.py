"""Open-loop deployment test: observations from a recorded dataset, actions on the wire.

Same control loop as ``main.py`` -- 30 Hz pacing, a policy query every ``open_loop_horizon``
steps, and ``JakaTabletopEnv.step`` publishing protocol-v1 frames -- but
``get_observation()`` reads a recorded LeRobot episode instead of the robot's ZMQ channels.
The motion buffer / low-level policy / simulator therefore receives exactly the frames a
real run would produce, while the real robot neither senses nor acts.

======================= 常用命令（直接复制） =======================

# 0) 前置：策略服务，单独开一个终端。已经在跑就跳过。
cd ~/workspace/OpenHLM/src/openpi4OpenHLM
./.venv/bin/python scripts/serve_policy.py --env JAKA --num-steps 10 --port 8000 \
    policy:checkpoint --policy.config=jaka_tabletop_pick \
    --policy.dir=checkpoints/jaka_tabletop_pick/jaka_pick_real_v1/30000

# 1) 后续命令都先执行这两行
cd ~/workspace/OpenHLM/src/GR00T-WholeBodyControl4OpenHLM/openpi-eval
export PYTHONPATH=~/workspace/OpenHLM/src/GR00T-WholeBodyControl4OpenHLM

# 2) 冒烟：ep0 跑 40 步（2 个 chunk），发到真实端口 28701
~/workspace/OpenHLM/src/openpi4OpenHLM/.venv/bin/python jaka_openloop_dataset.py \
    --dataset-episode 0 --max-steps 40 --jaka-motion-zmq-port 28701

# 3) 整条 episode：播完自动停（ep0 约 273 帧 / 9.1 s）
~/workspace/OpenHLM/src/openpi4OpenHLM/.venv/bin/python jaka_openloop_dataset.py \
    --dataset-episode 0 --max-steps 1000 --jaka-motion-zmq-port 28701

# 4) 循环播放：播完回到第 0 帧继续，直到 max-steps
~/workspace/OpenHLM/src/openpi4OpenHLM/.venv/bin/python jaka_openloop_dataset.py \
    --dataset-episode 0 --dataset-loop --max-steps 3000 --jaka-motion-zmq-port 28701

# 5) 旁路验证：只发到 28799，不碰仿真/机器人在收的 28701
~/workspace/OpenHLM/src/openpi4OpenHLM/.venv/bin/python jaka_openloop_dataset.py \
    --dataset-episode 0 --max-steps 40 --jaka-motion-zmq-port 28799

# 6) 换一条 episode（0..39；与源 episode 的对应关系见 conversion_manifest.json）
~/workspace/OpenHLM/src/openpi4OpenHLM/.venv/bin/python jaka_openloop_dataset.py \
    --dataset-episode 12 --max-steps 1000 --jaka-motion-zmq-port 28701

# 7) 指定延时日志（默认 visualization/openloop_dataset/epNN_<stamp>.json）
~/workspace/OpenHLM/src/openpi4OpenHLM/.venv/bin/python jaka_openloop_dataset.py \
    --dataset-episode 0 --max-steps 1000 --jaka-motion-zmq-port 28701 \
    --ctrl-log /tmp/openloop_ep0.json

# 8) 固定 anchor=(0,0,0.604)，不复现录制首帧位姿（默认复现录制首帧）
~/workspace/OpenHLM/src/openpi4OpenHLM/.venv/bin/python jaka_openloop_dataset.py \
    --dataset-episode 0 --no-anchor-from-dataset --max-steps 40 --jaka-motion-zmq-port 28701

# 9) 策略服务在另一台机器
~/workspace/OpenHLM/src/openpi4OpenHLM/.venv/bin/python jaka_openloop_dataset.py \
    --dataset-episode 0 --remote-host 192.168.1.10 --remote-port 8000 \
    --jaka-motion-zmq-port 28701

# 10) 数据集不在自动探测路径时（默认 record_teleop/record_data/lerobot_train/jaka_pick）
~/workspace/OpenHLM/src/openpi4OpenHLM/.venv/bin/python jaka_openloop_dataset.py \
    --dataset-root ~/lerobot_data/jaka_pick --dataset-episode 0 \
    --jaka-motion-zmq-port 28701

===================================================================

运行后按 ``s`` 开始；``Ctrl+C`` 提前停止但**仍会写延时日志**；``--max-steps`` 是安全上限。
结束时会打印 loop / step / infer 的 p50/p95/max 和超时步数（推理阻塞造成的 >1.5×dt 步）。

接收侧重点：
  * ``frame_index`` 连续、2 帧窗口不丢；
  * 每个 chunk 边界一次 ~130 ms 停顿（>50 ms）-> buffer 判停顿并重锚定；
  * 底层策略 620 维 obs：``ref_joint_pos`` 是 Isaac 序、``root_z_mf`` 在 base 高度（~0.6）。

日志字段（每步一条）：``frame / infer_ms / step_ms / loop_ms / hz``。
"""

from __future__ import annotations

import dataclasses
import io
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import tqdm
import tyro
from PIL import Image
from scipy.spatial.transform import Rotation

from openpi_client import image_tools
from openpi_client import websocket_client_policy

from jaka_tabletop_env import JakaTabletopEnv

# Reuse main.py's request builder + Ctrl+C deferral so the client side is identical.
from main import build_policy_request, prevent_keyboard_interrupt

# ---------------------------------------------------------------------------
# Dataset discovery (same candidates as the other replay tools)
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[2]
_DATASET_LEAF = ("jaka_pick",)
DATASET_ROOT_CANDIDATES = (
    _REPO_ROOT / "record_teleop" / "record_data" / "lerobot_train" / Path(*_DATASET_LEAF),
    Path(os.path.expanduser("~")) / "lerobot_data" / Path(*_DATASET_LEAF),
    _REPO_ROOT / "data" / Path(*_DATASET_LEAF),
)


def detect_dataset_root() -> str:
    for cand in DATASET_ROOT_CANDIDATES:
        if (cand / "meta" / "info.json").is_file():
            return str(cand)
    return str(DATASET_ROOT_CANDIDATES[0])


def load_episode(root: str, episode: int) -> dict:
    """Read one LeRobot episode: images (224x224 RGB), 30-dim states, 40-dim actions."""
    import pyarrow.parquet as pq

    path = Path(root) / "data" / "chunk-000" / f"episode_{episode:06d}.parquet"
    if not path.is_file():
        raise SystemExit(f"episode parquet not found: {path}")
    table = pq.read_table(path)
    blobs = table["head_image_left"].combine_chunks().field("bytes")
    images = np.stack([
        image_tools.resize_with_pad(
            np.asarray(Image.open(io.BytesIO(b.as_py())).convert("RGB"), dtype=np.uint8), 224, 224
        )
        for b in blobs
    ])
    return {
        "images": images,
        "states": np.array(table["state"].to_pylist(), dtype=np.float32),
        "actions": np.array(table["actions"].to_pylist(), dtype=np.float32),
    }


def dataset_task(root: str) -> str:
    with open(Path(root) / "meta" / "tasks.jsonl") as fh:
        return json.loads(fh.readline())["task"]


def yaw_of_quat(q) -> float:
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return float(Rotation.from_quat([x, y, z, w]).as_euler("xyz")[2])


class DatasetObsEnv(JakaTabletopEnv):
    """``JakaTabletopEnv`` whose observations come from a recorded episode.

    Only observation access is replaced; ``step()`` keeps the production joint reorder,
    anchor integration and protocol-v1 publish path unchanged.
    """

    def __init__(self, *args, images: np.ndarray, states: np.ndarray, **kwargs):
        super().__init__(*args, **kwargs)
        if len(images) != len(states) or len(states) == 0:
            raise ValueError("images/states must be non-empty and of equal length")
        self._images = images
        self._states = states
        self._t = 0

    @property
    def frame_count(self) -> int:
        return len(self._states)

    @property
    def frame_index(self) -> int:
        return self._t

    @property
    def exhausted(self) -> bool:
        return self._t >= len(self._states)

    def advance(self) -> None:
        self._t += 1

    def reset(self, default_pose=None):
        super().reset(default_pose)
        self._t = 0
        return self.get_observation()

    def get_observation(self):
        i = min(self._t, len(self._states) - 1)
        return {"head_image_left": self._images[i], "state": self._states[i].copy()}


# ---------------------------------------------------------------------------
# Args / main
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Args:
    dataset_root: str = ""                    # empty = auto-detect
    dataset_episode: int = 0
    dataset_loop: bool = False                # wrap to frame 0 when the episode ends

    max_steps: int = 1000                     # safety stop; episodes are 200-300 steps
    open_loop_horizon: int = 25               # same as main.py / deployment
    control_hz: int = 30

    remote_host: str = "127.0.0.1"
    remote_port: int = 8000
    prompt: str = ""                          # empty = the dataset's own task string

    # Motion stream (what the sim / motion buffer subscribes to)
    jaka_motion_zmq_address: str = "127.0.0.1"
    jaka_motion_zmq_port: int = 28701
    num_frames_to_send: int = 2

    # Anchor seed: by default reproduce the recording's own first-frame pose (x/y are
    # arbitrary in the recorded world; z is the recorded base height, load-bearing for
    # the tracker's root_z_mf). --no-anchor-from-dataset uses (0,0,0.604).
    anchor_from_dataset: bool = True
    initial_anchor_pos: tuple[float, float, float] = (0.0, 0.0, 0.604)
    initial_anchor_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)

    ctrl_log: str = ""                        # empty = visualization/openloop_dataset/<stamp>.json


def main(args: Args) -> None:
    root = args.dataset_root or detect_dataset_root()
    if not (Path(root) / "meta" / "info.json").is_file():
        raise SystemExit(f"dataset not found: {root}")

    ep = load_episode(root, args.dataset_episode)
    n = len(ep["states"])
    prompt = args.prompt or dataset_task(root)
    print(f"dataset : {root}")
    print(f"episode : {args.dataset_episode} ({n} frames, {n / args.control_hz:.2f}s)")
    print(f"prompt  : '{prompt}'")
    print(f"motion  : tcp://{args.jaka_motion_zmq_address}:{args.jaka_motion_zmq_port}")

    pos = tuple(float(v) for v in ep["actions"][0, 33:36]) if args.anchor_from_dataset else args.initial_anchor_pos
    rpy = (
        (float(ep["actions"][0, 27]), float(ep["actions"][0, 28]), yaw_of_quat(ep["actions"][0, 36:40]))
        if args.anchor_from_dataset
        else args.initial_anchor_rpy
    )
    print(f"anchor  : pos={tuple(round(v, 4) for v in pos)} rpy={tuple(round(v, 4) for v in rpy)}")

    env = DatasetObsEnv(
        control_hz=args.control_hz,
        mock=True,
        motion_zmq_address=args.jaka_motion_zmq_address,
        motion_zmq_port=args.jaka_motion_zmq_port,
        num_frames_to_send=args.num_frames_to_send,
        initial_anchor_pos=pos,
        initial_anchor_rpy=rpy,
        images=ep["images"],
        states=ep["states"],
    )
    client = websocket_client_policy.WebsocketClientPolicy(args.remote_host, args.remote_port)
    print(f"\nConnected to policy server at {args.remote_host}:{args.remote_port}")

    # Same warmup as main.py: 10 calls with the episode's first observation.
    print("\nWarming up policy server...")
    warm_start = time.time()
    obs = env.get_observation()
    for _ in range(10):
        client.infer(build_policy_request("jaka_tabletop_pick", obs, prompt, None))
    print(f"Policy server warmed up in {time.time() - warm_start:.3f} seconds.")

    print("\nPress 's' to start the open-loop run...")
    while input().strip().lower() != "s":
        print("Invalid key. Please press 's' to start.")

    records: list[dict] = []
    pred = None
    in_chunk = 0
    last_action = None
    target_dt = 1.0 / args.control_hz

    print("Running! Ctrl+C to stop early.")
    try:
        pbar = tqdm.tqdm(range(args.max_steps))
        for step_idx in pbar:
            start = time.time()
            frame = env.frame_index
            obs = env.get_observation()
            infer_ms = 0.0

            if pred is None or in_chunk >= args.open_loop_horizon:
                t_inf = time.time()
                with prevent_keyboard_interrupt():
                    pred = np.asarray(
                        client.infer(build_policy_request("jaka_tabletop_pick", obs, prompt, last_action))["actions"]
                    )
                infer_ms = (time.time() - t_inf) * 1000.0
                if pred.shape[0] < args.open_loop_horizon:
                    raise RuntimeError(f"policy returned {pred.shape}, need {args.open_loop_horizon} steps")
                in_chunk = 0

            action = pred[in_chunk]
            in_chunk += 1
            last_action = action.copy()

            t_step = time.time()
            env.step(action)
            step_ms = (time.time() - t_step) * 1000.0
            env.advance()

            if env.exhausted:
                if args.dataset_loop:
                    env.reset()
                    pred, in_chunk = None, 0
                else:
                    print(f"\nDataset episode finished ({env.frame_count} frames).")
                    break

            elapsed = time.time() - start
            if elapsed < target_dt:
                time.sleep(target_dt - elapsed)
            loop_ms = (time.time() - start) * 1000.0

            records.append({
                "step": step_idx,
                "frame": frame,
                "infer_ms": infer_ms,
                "step_ms": step_ms,
                "loop_ms": loop_ms,
                "hz": 1000.0 / loop_ms if loop_ms > 0 else 0.0,
            })
            pbar.set_postfix({"infer": f"{infer_ms:.0f}ms", "hz": f"{1000.0 / loop_ms:.1f}"})
    except KeyboardInterrupt:
        print("\nCtrl+C detected -- stopping open-loop run.")
    finally:
        env.close()

    if not records:
        print("No steps executed.")
        return
    out = args.ctrl_log or str(
        _HERE / "visualization" / "openloop_dataset" / f"ep{args.dataset_episode:02d}_{datetime.now():%Y%m%d_%H%M%S}.json"
    )
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(records, fh)

    loop = np.array([r["loop_ms"] for r in records])
    step = np.array([r["step_ms"] for r in records])
    inf = np.array([r["infer_ms"] for r in records if r["infer_ms"] > 0])
    overrun = int((loop > 1.5 * target_dt * 1000).sum())
    print(f"\n[ctrl-log] {len(records)} steps -> {out}")
    print(f"  loop  p50/p95/max = {np.percentile(loop, 50):.1f} / {np.percentile(loop, 95):.1f} / {loop.max():.1f} ms"
          f"   overruns: {overrun} ({100.0 * overrun / len(records):.1f}%)")
    print(f"  step  p50/p95/max = {np.percentile(step, 50):.2f} / {np.percentile(step, 95):.2f} / {step.max():.2f} ms")
    if inf.size:
        print(f"  infer p50/p95/max = {np.percentile(inf, 50):.1f} / {np.percentile(inf, 95):.1f} / {inf.max():.1f} ms ({inf.size} calls)")


if __name__ == "__main__":
    main(tyro.cli(Args))
