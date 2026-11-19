# 12　Khan 实机部署：pi0.5 抓取策略与 180k EMA tracker

本流程使用 OpenHLM `dev` 的 `0827d35`、带 VLA motion buffer 的机器人控制器 `fde4c70`、实机抓取 checkpoint `jaka_pick_real_v1/30000` 和 tracker `ckpt_0180000_bones_ema.onnx`。假设机器人部署包、ROS 2 Humble 相机工作区和 OpenHLM 推理环境已经安装。权重下载和环境准备见 [11 本机部署](11-local-deploy.md)。

首轮顺序为 **策略服务和预热 → 机器人控制器留在 IDLE → 相机驱动和 RGB 桥接器 → 机器人到起始姿态并进入 SuperZmq → 启动客户端 → 在客户端输入 `s` 回车**。

## 主机和端口

| 监听端 | 连接端 | 示例 |
| --- | --- | --- |
| GPU 服务器的 `serve_policy.py`，WebSocket 8000 | 客户端 `main.py` | 二者同机时用 `127.0.0.1:8000`，分机时用 GPU 服务器的地址 |
| 客户端 PC 的 `main.py`，ZMQ 28701 | 机器人 SuperZmq 订阅参考动作 | 机器人 YAML 指向 `tcp://10.5.5.120:28701` |
| 机器人 SuperZmq 的 StatePublisher，ZMQ 28711 | `main.py` 订阅实测 state | 机器人有线地址示例为 `10.5.5.100` |
| 相机主机的 `cam_pub.py`，ZMQ 28712 | `main.py` 订阅头部 RGB | 桥接器在机器人上运行时也用 `10.5.5.100` |

以下命令各自在标注的终端执行。将示例 IP 和工作区路径替换成现场配置。使用 VLA 抓取时，28701 由 `main.py` 发布，先退出使用同一个端口的 PICO motion 发布器。

## 1. GPU 服务器：加载权重并预热

在 GPU 终端 A 启动策略服务。客户端在另一台 PC 时，该 PC 的 OpenHLM 也要使用上述 `dev` 版本。

```bash
cd ~/workspace/OpenHLM
git switch dev
git pull --ff-only origin dev
cd src/openpi4OpenHLM
export CUDA_VISIBLE_DEVICES=0
CKPT=checkpoints/jaka_tabletop_pick/jaka_pick_real_v1/30000

test -s "$CKPT/model.safetensors" &&
test -s "$CKPT/assets/jaka_pick/norm_stats.json" &&
./.venv/bin/python scripts/serve_policy.py \
  --env JAKA --num-steps 10 --port 8000 \
  policy:checkpoint --policy.config=jaka_tabletop_pick --policy.dir="$CKPT"
```

出现 `server listening on 0.0.0.0:8000` 后，在 GPU 终端 B 运行探针。它会用真实权重推理，不连接机器人。确认输出的动作形状是 `(50, 33)` 且数值有限。可以在机器人使能之前完成首次 GPU 预热。

```bash
cd ~/workspace/OpenHLM/src/GR00T-WholeBodyControl4OpenHLM/openpi-eval
PYTHONPATH=.. ~/workspace/OpenHLM/src/openpi4OpenHLM/.venv/bin/python \
  jaka_policy_server_probe.py --host 127.0.0.1 --port 8000
```

## 2. 机器人：配置和启动控制器，先留在 IDLE

进入包含 `KpiDeployReal`、`startup.sh`、`runtime_lib/` 和 `config/` 的实际部署目录。这里沿用版本化目录名。下面的校验值对应已验证的 180k EMA 模型。

```bash
cd ~/sdk_demo_vla_fde4c70
test -x ./KpiDeployReal
echo 'a9f28e9be75743d847f2c58ccde921d191fc8b8487de34fcaec1d83a930d24d7  config/policy_mimic/ckpt_0180000_bones_ema.onnx' | sha256sum -c -
nano config/KpiConfig.yaml
```

在现有 YAML 的相应位置核对以下字段，保留本机标定的增益、初始关节角和限位。`state_zmq_bind` 位于顶层，`zmq_connect` 位于已有的 SuperZmq 节内。

```yaml
state_zmq_bind: "tcp://*:28711"

jakaminimimic_superzmq:
  motion_backend: vla_pose_v1
  policy_path: "policy_mimic/ckpt_0180000_bones_ema.onnx"
  zmq_connect: "tcp://10.5.5.120:28701"
  vla_source_hz: 30
  vla_stale_timeout_s: 1.0
  dt_s: 0.02
  tolerance_s: 0.04
  future_steps: [0, 1, 2, 3, 4]
```

首轮手动接管时，在机器人尚未使能、可以切换服务的条件下执行以下命令。`jkkhan-ros2.service` 应为 `active`。从部署目录启动后，确认日志中有 `[SuperMimic real] 620 -> 27 actor loaded`，等 EtherCAT 和 IMU 就绪并进入 IDLE。

```bash
cd ~/sdk_demo_vla_fde4c70
sudo systemctl stop jkpi-kpideploy.service &&
sudo systemctl restart jkkhan-ros2.service &&
systemctl is-active --quiet jkkhan-ros2.service &&
./startup.sh
```

## 3. 相机主机：启动 Gemini 2L 和 RGB 桥接器

相机终端 A 运行驱动。两个相机终端使用同一 ROS 域和同一 Orbbec 工作区。将 `ORBBEC_WS` 设为实际工作区；示例默认 `~/ros2_ws`。

```bash
source /opt/ros/humble/setup.bash
source "${ORBBEC_WS:-$HOME/ros2_ws}/install/setup.bash"
ros2 launch orbbec_camera gemini2L.launch.py \
  enable_color:=true color_width:=640 color_height:=400 color_fps:=30 color_format:=MJPG \
  enable_depth:=false enable_ir:=false enable_point_cloud:=false enable_frame_sync:=false
```

相机终端 B 运行桥接器。如果 `cam_pub.py` 是单独复制到机器人上的，将 `CAM_BRIDGE_DIR` 设为以前运行 `uv run cam_pub.py` 的目录。

```bash
source /opt/ros/humble/setup.bash
source "${ORBBEC_WS:-$HOME/ros2_ws}/install/setup.bash"
cd "${CAM_BRIDGE_DIR:-$HOME/workspace/jaka_real_teleop/real/scripts}"
uv run cam_pub.py --topic /camera/color/image_raw \
  --bind 'tcp://*:28712' --color-format rgb
```

确认桥接日志持续显示 `resolution=640x400x3`，帧率接近 30 Hz。MJPG 是相机到驱动的格式，28712 上是原始 RGB。桥接器如无输出，可在同一 ROS 环境执行 `ros2 topic hz /camera/color/image_raw` 检查驱动端。

## 4. 机器人手柄：进入 SuperZmq

1. 在 IDLE 确认手柄在线且拥有 JOY 控制权，单按 **BACK**。等全部 27 个关节使能、无故障，终端出现 `init joint state bind`。
2. 用现场已确认方向的十字键选择 `init_target`，等待实际到位。代码中 `axes_DY < -0.5` 选初始姿态，`axes_DY > 0.5` 选 `zero_target`。同时确认机器人相对桌面的位置和物体摆放。
3. 从松开状态按 **LB+B** 进入 SuperZmq，然后松开。控制器从此时开始发布 28711 上的实测 state，并主动用 tracker 跟踪参考。首个 VLA 窗口就绪前使用控制器的默认参考。若报告 `cannot bind the measured-state publisher`，先解决 28711 的端口占用，再从 IDLE 重试。

## 5. 客户端 PC：开始本轮抓取

```bash
cd ~/workspace/OpenHLM/src/GR00T-WholeBodyControl4OpenHLM/openpi-eval
export PYTHONPATH="$HOME/workspace/OpenHLM/src/GR00T-WholeBodyControl4OpenHLM"
ROBOT_IP=10.5.5.100
CAMERA_IP="$ROBOT_IP"
POLICY_HOST=127.0.0.1

~/workspace/OpenHLM/src/openpi4OpenHLM/.venv/bin/python main.py \
  --env jaka_tabletop_pick \
  --remote-host "$POLICY_HOST" --remote-port 8000 \
  --jaka-state-address "$ROBOT_IP" --jaka-state-port 28711 \
  --jaka-head-image-address "$CAMERA_IP" --jaka-head-image-port 28712 \
  --jaka-motion-zmq-address '*' --jaka-motion-zmq-port 28701 \
  --jaka-initial-anchor-pos 0 0 0.604 \
  --control-hz 30 --open-loop-horizon 25 \
  --instruction "pick up the box" --max-steps 100000 --no-opencv-visualize
```

部署前按实际位置设置 `CAMERA_IP` 和 `POLICY_HOST`。末个 anchor 坐标是起始姿态下的 `base_link` 离地高度，示例取实机数据集均值 0.604 m。它决定本轮轨迹的初始高度。控制器 YAML 的 `default_root_pos` 用于首个参考窗口就绪前的 tracker 姿态。

客户端先取得有效的 state 和头部 RGB，用这份观测做 10 次预热推理。看到 **`Press 's' to start the evaluation...` 后，在客户端终端输入 `s` 并回车**。此后客户端重新取观测、正式推理，第二个动作开始发出第一组双帧参考 `[0,1]`。每次预测 50 步，默认执行 25 步后同步请求下一块。按键前 tracker 已在闭环运行；客户端的预热不发 VLA 动作。

### 出现 `No fresh ...` 时

- `state`：确认机器人仍在 SuperZmq、该模式已绑定 28711、客户端连接的是机器人地址，并且 state 的发布时间持续推进。
- `head_image_left`：检查桥接日志里的分辨率和帧率，以及客户端连接的是否为运行 `cam_pub.py` 那台主机的 28712。
- 客户端独立缓存两路有效数据。同一控制步可以使用新 state 和刚刚收到的上一张 RGB。两路按客户端读取时刻计龄，默认各自最多可复用 0.30 s；需要等待时，首次调用的等待预算是 3 s，后续调用为 0.30 s。等待期间本轮调用还没有执行下一个动作。
- 机器人另有独立的参考流超时。收到首个有效参考包后，默认连续 1 s 无新参考帧就锁存参考。排除中断原因后，按下一节从新起点接入。

## 6. 完成一次抓取，复位并开始下一轮

1. 抓取完成且机器人处于适合切换的稳定姿态时，在 **客户端 PC 的 `main.py` 终端**按 **Ctrl+C**。看到 `Do one more eval?` 后输入 **`n` 回车**。正在进行的同步推理可能要等返回才处理 Ctrl+C。客户端停止追加参考后，tracker 仍在闭环运行。
2. 在**机器人手柄**上松开后再按 **LB+B** 返回 IDLE，并松开。退出会释放旧 buffer 和 state 发布器。在 SuperZmq 中，手柄 **A** 的含义是请求断使能并进入 FAULT；正常换轮使用 LB+B。
3. 在 IDLE **单按 BACK**，等重新绑定关节目标，再选 `init_target`，等姿态实际到位。恢复机器人相对桌面的位置，重新摆好物体。
4. 再按 **LB+B** 进入 SuperZmq、松开，重跑第 5 节的客户端命令。出现启动提示后输入 **`s` 回车**。GPU 服务、相机和机器人的 `KpiDeployReal` 可跨轮持续运行。

客户端选择 `y` 会沿用同一个发布实例及其帧号、积分位置和 yaw。机器人已经物理复位时，退出旧客户端、重新进入 SuperZmq 并启动新客户端，才能让新一轮的世界位姿和帧号从选定的起点开始。
