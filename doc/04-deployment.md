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

./.venv/bin/python scripts/serve_policy.py --env JAKA --num-steps 10 \
  policy:checkpoint --policy.config jaka_tabletop_pick \
  --policy.dir checkpoints/jaka_tabletop_pick/jaka_pick_real_v1/30000
```

服务端通过显式路径加载已确认由 base_link 数据训练的权重。JAKA 的 33 维输出本身无法标识训练时的 anchor。

常用参数：`--port <n>`（默认 8000）、`--default-prompt <text>`、`--record`。

## 4.3 部署客户端（Jaka）

容器内可使用以下回环命令。本机原生部署见 [11 本机部署](11-local-deploy.md)。

```bash
docker exec -i -w /workspace/OpenHLM/src/GR00T-WholeBodyControl4OpenHLM/openpi-eval \
  -e PYTHONPATH=/workspace/OpenHLM/src/GR00T-WholeBodyControl4OpenHLM \
  xujinfan_dev /workspace/OpenHLM/src/openpi4OpenHLM/.venv/bin/python main.py \
  --env jaka_tabletop_pick \
  --mock --jaka-motion-zmq-address 127.0.0.1 --jaka-motion-zmq-port 38893 \
  --use-fake-policy \
  --max-steps 200 \
  --no-opencv-visualize
```

为什么要这么写：

| 要素 | 原因 |
| --- | --- |
| `PYTHONPATH=<GR00T repo root>` | `sonic_g1_env` 会 import `gear_sonic`，直接运行的 main.py 也会从仓库定位该模块 |
| `--no-opencv-visualize` | 容器**无 X display**，cv2 在 `imshow` 上直接 abort。`opencv_visualize` **默认是 True**，必须显式关掉 |
| 喂 stdin（如 `<<< $'s\nn\n'`） | 主循环会阻塞在 `input()` 等 's' |

**常用客户端参数**（`openpi-eval/main.py`）：

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--env` | `sonic_g1` | `sonic_g1` \| `jaka_tabletop_pick` |
| `--remote-host` / `--remote-port` | `0.0.0.0` / `8000` | 策略服务端地址 |
| `--instruction` | JAKA 为 `pick up the box` | 可覆盖语言指令 |
| `--control-hz` | 30 | 控制频率 |
| `--open-loop-horizon` | 25 | 策略 horizon 是 50，最多开环 25 步 |
| `--max-steps` | 1000 | 跑多少步 |
| `--mock` | False | JAKA 使用合成观测，motion PUB 仍会绑定指定地址 |
| `--use-fake-policy` | False | 不连策略服务端，用假动作 |
| `--save-video` / `--video-save-dir` | False / `/data/eval_videos` | 存 MP4 |
| `--save-action-chunk` | False | 存预测的动作 chunk，调试用 |
| `--jaka-motion-zmq-port` | 28701 | 下游接收方订阅这个端口 |
| `--jaka-initial-anchor-pos X Y Z` | `0 0 0.604` | 起始姿态下的 base_link 世界位置（米） |

### Jaka anchor 约定（最容易搞错的地方）

- `action[30:33]` 为 `base_link` 坐标系内的线速度。发布器先把速度旋转到世界系，再积分下一帧位置。
- 本帧姿态使用当前 roll/pitch 和本帧动作生效前的累积 yaw。
- 实机 `--jaka-initial-anchor-pos` 的 z 应取起始姿态下 base_link 离地高度。模型输出 33 维，绝对高度由此初值确定。使用以 base_link 为 anchor 的 checkpoint。
- 实机 state 在 :28711 发 JSON，头部 RGB 在 :28712 发 12 字节宽高通道数加像素。通过 `--jaka-state-address`、`--jaka-head-image-address` 指向发布端。JAKA 等待两路有效观测，按本地接收年龄复用最近样本；超时会停止本次调用并报告缺失源。
- 每个 :28701 `pose` 包都携带 `anchor_body=base_link`、`joint_order=jaka` 和 `stream_id`。进程内 reset 清滑窗，帧号、积分位置和 yaw 延续。新发布器生成新 stream_id 并从新起点计数。正常换块只有发送停顿，帧号仍连续。

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
