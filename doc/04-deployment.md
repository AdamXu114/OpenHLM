# 04 部署与推理

## 4.1 架构

```
┌─────────────────────────────┐        websocket :8000         ┌──────────────────────────┐
│  策略服务端（GPU 机器）      │ ◄───────────────────────────── │  部署客户端               │
│  scripts/serve_policy.py    │        观察 → 动作 chunk        │  openpi-eval/main.py      │
│  JAKA / SONICG1             │                                │  JakaTabletopEnv         │
└─────────────────────────────┘                                └────────────┬─────────────┘
                                                                            │ ZMQ tcp://*:28701
                                                                            │ 协议 v1 二进制帧（2 帧滑窗）
                                                                            ▼
                                                               ┌──────────────────────────┐
                                                               │  下游接收方 / 机器人        │
                                                               │  RealtimeMotionBufferVla │
                                                               └──────────────────────────┘
```

Jaka 侧的动作约定：33 维策略动作 → 按 `PERM_POLICY_TO_SIM` 从**策略序（手臂优先）**转成 **SIM 序（腿优先）**，
anchor 世界位姿由每帧速度 + 朝向重新积分。

## 4.2 策略服务端

```bash
docker exec -it xujinfan_dev bash
cd /workspace/OpenHLM/src/openpi4OpenHLM

export CUDA_VISIBLE_DEVICES=<预约的卡号>

# 用 JAKA 默认 checkpoint（写在 serve_policy.py 的 DEFAULT_CHECKPOINT 里）
./.venv/bin/python scripts/serve_policy.py --env JAKA --num-steps 10

# 或显式指定 config + checkpoint
./.venv/bin/python scripts/serve_policy.py \
  --env JAKA \
  --num-steps 10 \
  policy:checkpoint \
  --policy.config=jaka_tabletop_pick \
  --policy.dir=checkpoints/jaka_tabletop_pick/jaka_tabletop_pick_v1/30000
```

> `EnvMode.JAKA` 的值就是 `"jaka_tabletop_pick"`，两种写法等价。
> **换 checkpoint 后记得确认 `--policy.dir` 指到你要的那一步。**
> 在 `scripts/serve_policy.py` 的 `DEFAULT_CHECKPOINT[EnvMode.JAKA]` 里（约 96 行）。

常用参数：`--port <n>`（默认 8000）、`--default-prompt <text>`、`--record`。

## 4.3 部署客户端（Jaka）

**必须在容器里这样跑**（三个点都是必须的，否则直接挂）：

```bash
docker exec -i -w /workspace/OpenHLM/src/GR00T-WholeBodyControl4OpenHLM/openpi-eval \
  -e PYTHONPATH=/workspace/OpenHLM/src/GR00T-WholeBodyControl4OpenHLM \
  xujinfan_dev /workspace/OpenHLM/src/openpi4OpenHLM/.venv/bin/python main.py \
  --env jaka_tabletop_pick \
  --mock \
  --use-fake-policy \
  --max-steps 200 \
  --no-opencv-visualize
```

为什么要这么写：

| 要素 | 原因 |
| --- | --- |
| `PYTHONPATH=<GR00T repo root>` | `sonic_g1_env` 会 import `gear_sonic`，openpi 的 venv 里没有；不加这句 main.py 在 import 阶段就死（生产环境用的是 `.venv_teleop`，本容器里没有） |
| `--no-opencv-visualize` | 容器**无 X display**，cv2 在 `imshow` 上直接 abort。`opencv_visualize` **默认是 True**，必须显式关掉 |
| 喂 stdin（如 `<<< $'s\nn\n'`） | 主循环会阻塞在 `input()` 等 's' |

**常用客户端参数**（`openpi-eval/main.py`）：

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--env` | `sonic_g1` | `sonic_g1` \| `jaka_tabletop_pick` |
| `--remote-host` / `--remote-port` | `0.0.0.0` / `8000` | 策略服务端地址 |
| `--instruction` | 紫色软指任务 | 语言指令 |
| `--control-hz` | 30 | 控制频率 |
| `--open-loop-horizon` | 25 | 策略 horizon 是 50，最多开环 25 步 |
| `--max-steps` | 1000 | 跑多少步 |
| `--mock` | False | 不连真实机器人 |
| `--use-fake-policy` | False | 不连策略服务端，用假动作 |
| `--save-video` / `--video-save-dir` | False / `/data/eval_videos` | 存 MP4 |
| `--save-action-chunk` | False | 存预测的动作 chunk，调试用 |
| `--jaka-motion-zmq-port` | 28701 | 下游接收方订阅这个端口 |
| `--jaka-initial-anchor-pos` | `(0.0, 0.0, 0.83)` | anchor 初始位置，**z 是有意义的**（见下） |

### Jaka anchor 约定（最容易搞错的地方）

- `action[30:33]`（`anchor_lin_vel`）是 **anchor body 坐标系**（`waist_yaw_Link`）下的速度，**不是世界速度**
  —— 必须先按 anchor 四元数旋转再积分。
- 发布的帧是**它所在区间的起始时刻**的位姿，所以四元数由**累加后的 yaw** 构造，且要在该帧的
  `yaw_vel` 生效**之前**。
- `jaka_initial_anchor_pos` 的 **z 是有用的**（tracker 的 `root_z_mf` 是绝对世界 z，站立时约 0.83 m）；
  x/y 无所谓（`root_pos_diff_b` 是平移不变的）。

### 两个不能破坏的不变量

下游 `RealtimeMotionBufferVla` 依赖这两点，破坏了会**静默失败**：

1. **`frame_index` 永不重置。** 接收方的去重基线在它自己 `clear()` 之后依然存在，
   所以客户端重启会让参考轨迹**静默冻结**。`JakaTabletopEnv.reset()` 里特意不复位 `_frame_index`。
2. **~120 ms 的推理停顿不能被帧填满。**

## 4.4 协议改动的验证

改了发布协议之后**必须**重跑这个测试（它用 `ast` 从权威解码器
`implement_action_analysis/.../motion_buffer.py` 里抽出 `_decode_binary_v1` 来对拍）：

```bash
docker exec -i -w /workspace/OpenHLM/src/GR00T-WholeBodyControl4OpenHLM/openpi-eval \
  -e PYTHONPATH=/workspace/OpenHLM/src/GR00T-WholeBodyControl4OpenHLM \
  xujinfan_dev /workspace/OpenHLM/src/openpi4OpenHLM/.venv/bin/python jaka_tabletop_env_test.py
```

## 4.5 上游通用流程（G1 / SONIC）

完整流程见 [../src/openpi4OpenHLM/inference.md](../src/openpi4OpenHLM/inference.md)。要点：

**仿真测试**（三个终端）：

```bash
# T1: MuJoCo 虚拟机器人
cd src/GR00T-WholeBodyControl4OpenHLM
source .venv_teleop/bin/activate
python gear_sonic/scripts/run_sim_loop.py

# T2: 底层控制服务
cd src/GR00T-WholeBodyControl4OpenHLM/gear_sonic_deploy
source scripts/setup_env.sh
./deploy.sh --input-type zmq sim

# T3: OpenPI 部署客户端
cd src/GR00T-WholeBodyControl4OpenHLM
source .venv_teleop/bin/activate
python scripts/openpi-eval/main.py \
  --control_hz 30 --max_steps 10000 --save_video \
  --instruction "example" --exp_name openhlm_example
```

操作：T2 按 `]` 启动策略 → 按 `Enter` 进 streaming → T1 按 `9` 让机器人落地 →
T3 机器人抬起后按 `s` 开始推理循环。停止：T3 `Ctrl+C` 停高层策略并回初始位姿，T2 按 `O` 停底层服务。

**真机**：T1 换成 `cd src/GR00T-WholeBodyControl4OpenHLM/scripts && bash deploy_stream.sh`，其余同上。
硬件侧（G1 板载电脑）先跑 `uv run hardware_setup` 初始化相机/夹爪/VR（**记得在 gripper pane 填 sudo 密码**）。
