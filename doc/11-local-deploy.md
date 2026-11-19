# 11 本机部署（RTX 3090 单卡）

> 本文只讲**本机**（Ubuntu 24.04 + 1×RTX 3090，仓库 `~/workspace/OpenHLM`）怎么把训练好的 Jaka 策略跑起来。
> 本机环境安装过程见 [08-local-machine.md](08-local-machine.md)；服务器/容器那套流程见
> [01](01-environment.md)–[07](07-jaka-deploy-tutorial.md)。anchor 约定和协议原理见 [04](04-deployment.md)。

## 0. 和服务器文档的差别

| 项 | 本机 | 服务器 |
| --- | --- | --- |
| 运行方式 | 原生（无容器） | `docker exec xujinfan_dev bash` |
| GPU | 1 × RTX 3090 24 GB（策略服务约占 7.1 GiB） | 4 × 4090 48 GB，需要 `hpc` 预约 |
| 代理 | 不需要 | 需要 `~/.ssh/config`/`Base_bash.md` 里的隧道 |
| 训练 | **不要在本机训练**（24 GB 放不下权重+优化器状态） | 训练在这里跑 |

## 1. 下载 checkpoint

训练在服务器完成，产物在：

```text
/data0/xujinfan/OpenHLM/src/openpi4OpenHLM/checkpoints/jaka_tabletop_pick/jaka_pick_real_v1/30000/
├── assets/            # 含 norm_stats.json，推理靠它
├── metadata.pt
└── model.safetensors  # bf16，整目录约 7.0 GB
```

下载到本机（保持同样的目录名，`serve_policy.py` 用相对路径）：

```bash
cd ~/workspace/OpenHLM/src/openpi4OpenHLM
mkdir -p checkpoints/jaka_tabletop_pick/jaka_pick_real_v1

rsync -avP -e 'ssh -p 9980' \
  xujinfan@zz.irmv.top:/data0/xujinfan/OpenHLM/src/openpi4OpenHLM/checkpoints/jaka_tabletop_pick/jaka_pick_real_v1/30000 \
  checkpoints/jaka_tabletop_pick/jaka_pick_real_v1/
```

校验：

```bash
ls checkpoints/jaka_tabletop_pick/jaka_pick_real_v1/30000
# assets  metadata.pt  model.safetensors
du -sh checkpoints/jaka_tabletop_pick/jaka_pick_real_v1/30000   # ≈ 7.0G
```

> 如果这次训练已经传到 HuggingFace，也可以 `hf download XuJinfan/<repo> --local-dir ...`，
> 旧 v1 的做法见 [07](07-jaka-deploy-tutorial.md) §准备。

本机目前还留着旧仿真数据训练的 `checkpoints/jaka_tabletop_pick/jaka_tabletop_pick_v1/30000`，
**不要**把它当成新模型。

## 2. norm stats

- **推理只需要 checkpoint 自带的** `30000/assets/.../norm_stats.json`：
  `create_trained_policy()` 是从 checkpoint 目录读统计量的（见
  `src/openpi4OpenHLM/src/openpi/policies/policy_config.py`），不是从仓库 `assets/`。
- 仓库里的 `assets/jaka_tabletop_pick/jaka_pick/norm_stats.json` 用于**训练/数据集评测**，
  已经随仓库提交，推理时不用管。
- 因此「只做推理」的话，除了 checkpoint 目录本身，不需要额外拷 norm stats。

## 3. 起策略服务

```bash
cd ~/workspace/OpenHLM/src/openpi4OpenHLM
export CUDA_VISIBLE_DEVICES=0

./.venv/bin/python scripts/serve_policy.py \
  --env JAKA \
  --num-steps 10 \
  --port 8000 \
  policy:checkpoint \
  --policy.config=jaka_tabletop_pick \
  --policy.dir=checkpoints/jaka_tabletop_pick/jaka_pick_real_v1/30000
```

- 服务端 CLI 的枚举参数用 `--env JAKA`，训练配置用 `--policy.config=jaka_tabletop_pick`。JAKA 服务要求显式选择 checkpoint，以免自动加载旧的 waist 权重。
- 首次加载约 1–2 分钟；日志出现 `server listening on 0.0.0.0:8000` 即就绪。
- 想换采样步数就改 `--num-steps`（默认 10，越少越快）。

## 4. 服务自检（不接机器人）

另开一个终端，直接对这个服务发一次推理：

```bash
cd ~/workspace/OpenHLM/src/GR00T-WholeBodyControl4OpenHLM/openpi-eval
PYTHONPATH=~/workspace/OpenHLM/src/GR00T-WholeBodyControl4OpenHLM \
  ~/workspace/OpenHLM/src/openpi4OpenHLM/.venv/bin/python jaka_policy_server_probe.py \
  --host 127.0.0.1 --port 8000
```

`jaka_policy_server_probe.py` 发送的 prompt 是 `pick up the box`（与数据集 task 一致），
并会打印首次/稳态推理耗时。本机 3090 实测：首次约 129 s（Triton autotune），
之后每次 50 步 chunk 约 140 ms。

## 5. 起客户端

### A. 连真机

在机器人实际使用的 `KpiConfig.yaml` 中，将 `jakaminimimic_superzmq.motion_backend` 设为 `vla_pose_v1`，`zmq_connect` 设为 `tcp://<运行 main.py 的 PC 地址>:28701`。客户端绑定 motion PUB，机器人主动连接它。要使用 180k EMA tracker，控制器的 `policy_path` 应指向已验证的 `policy_mimic/ckpt_0180000_bones_ema.onnx`。

启动 RGB 相机发布器并进入 SuperZmq，让实测 state（28711）和相机（28712）均开始发布。以下命令读取环境变量 `ROBOT_IP`；如果相机发布器位于另一台机器，可用 `CAMERA_IP` 单独指定。

```bash
cd ~/workspace/OpenHLM/src/GR00T-WholeBodyControl4OpenHLM/openpi-eval
export PYTHONPATH=~/workspace/OpenHLM/src/GR00T-WholeBodyControl4OpenHLM

~/workspace/OpenHLM/src/openpi4OpenHLM/.venv/bin/python main.py \
  --env jaka_tabletop_pick \
  --remote-host 127.0.0.1 --remote-port 8000 \
  --jaka-motion-zmq-port 28701 \
  --jaka-state-address "${ROBOT_IP:?请设置机器人 IP}" --jaka-state-port 28711 \
  --jaka-head-image-address "${CAMERA_IP:-$ROBOT_IP}" --jaka-head-image-port 28712 \
  --control-hz 30 --open-loop-horizon 25 \
  --instruction "pick up the box" --max-steps 100000
```

- 客户端默认最多等待 3 s 以取得初始观测并开始预热，输入 `s` 回车后开始发参考。SSH 无显示时加 `--no-opencv-visualize`。两路观测独立更新，短时复用最近的有效样本；本地接收年龄上限为 0.30 s，持续缺失或超龄会报错并停止。相机使用 RGB；相机线协议只含宽、高、通道数和像素，接收年龄无法反映曝光时刻。
- 默认 prompt 是训练文本 `pick up the box`。输入是 30 维 state 和一张按比例缩放并补边的 224×224 RGB 图。策略每次返回 50 个 33 维动作；每帧含 27 个参考关节角、roll、pitch、yaw 角速度、3 维 base 系线速度。
- 初始 `base_link` 高度使用实机数据集均值 0.604 m。若起始姿态不同，可传 `--jaka-initial-anchor-pos X Y Z`。每帧先发布当前位姿，再按 1/30 s 积分下一帧；模型的 33 维输出不含绝对高度。
- 28701 每包默认带相邻两帧、`anchor_body=base_link`、`joint_order=jaka`、本实例的 `stream_id`。同一实例换动作块时，帧号、位置、yaw 连续。结束一条任务、机器人复位后要从新原点开始时，退出 SuperZmq，重启客户端并从新的初始 pose 重新进入 SuperZmq。控制器默认在收到首批有效帧之后连续 1 s 没有新帧时锁存参考，需要退出并重新进入该模式。

### B. 本地自检（无机器人）

```bash
cd ~/workspace/OpenHLM/src/GR00T-WholeBodyControl4OpenHLM/openpi-eval
export PYTHONPATH=~/workspace/OpenHLM/src/GR00T-WholeBodyControl4OpenHLM
PY=~/workspace/OpenHLM/src/openpi4OpenHLM/.venv/bin/python

# 真策略服务 + 合成观测；发布到独立回环端口
"$PY" main.py --env jaka_tabletop_pick --remote-host 127.0.0.1 --remote-port 8000 \
  --mock --jaka-motion-zmq-address 127.0.0.1 --jaka-motion-zmq-port 38893 \
  --max-steps 50 --no-opencv-visualize

# 同一客户端和打包器，使用假策略，可单独运行
"$PY" main.py --env jaka_tabletop_pick --mock --use-fake-policy \
  --jaka-motion-zmq-address 127.0.0.1 --jaka-motion-zmq-port 38893 \
  --max-steps 50 --no-opencv-visualize
```

两种自检都等待输入 `s`，完成后输入 `n` 退出。`--mock` 使用合成观测，motion PUB 仍会发包，默认仅绑定回环地址。

## 6. 本机性能参考（RTX 3090 实测）

| 项 | 数值 |
| --- | --- |
| 模型显存 | 约 7.1 GiB |
| 首次推理 | ~129 s（Triton autotune + 首次 kernel） |
| 稳态推理 | ~140 ms / 50 步 chunk |
| 控制频率/开环 | 动作时间轴为 30 Hz，每执行 25 步同步推理；25 步对应约 0.83 s 的轨迹，推理耗时会出现在跨块发包间隔中 |

## 7. 已知问题（部署前必读）

1. **anchor 帧**：数据集与下游已统一为 `base_link`，本机默认初始 z 取 **0.604**
   （记录 base 高度均值；实测范围 0.587–0.614，每集首帧 0.608 ± 0.003）。
   **不要**再用旧 waist 的 0.83 —— 它比 base 高 0.22385 m，会让参考整体抬高 0.22 m。
   旧 `jaka_tabletop_pick_v1` 是 waist 标签训练的，不要在新下游上部署。
2. **prompt**：训练/部署必须同为 `pick up the box`。
3. **`PYTHONPATH`**：`main.py` 直接运行时会定位同仓库的 `gear_sonic`；回放脚本仍可沿用示例的 `PYTHONPATH`。
4. **opencv**：无 X display 时 `cv2.imshow` 会 abort，加 `--no-opencv-visualize`。
5. **别在本机训练**：3090 24 GB 放不下 3.6B 模型 + 优化器状态；训练在服务器上做。

## 8. 数据集回放 / 评测（可选）

- `jaka_policy_dataset_eval.py`：策略对数据集 ground truth 的误差对比（需要起策略服务）。
- `jaka_dataset_action_publisher.py`：把数据集的 33 维动作直接下发给控制器，
  可先用 `--dry-run --episodes 0 --anchor-from-dataset` 验证录制初始高度、yaw 和动作转换。发布器检查 `conversion_manifest.json` 的 `anchor_body_name`；旧数据缺少标识且已确认使用 base_link 时可传 `--assume-base-link`。

两个脚本都会自动探测数据集根目录（`record_teleop/record_data/lerobot_train/jaka_pick`），
也可用 `--dataset-root` / `--episodes` 指定。

## 参考

- [08-local-machine.md](08-local-machine.md)：本机环境安装（uv / venv / 权重）
- [04-deployment.md](04-deployment.md)：链路原理、anchor 约定、端口表
- [07-jaka-deploy-tutorial.md](07-jaka-deploy-tutorial.md)：机器人侧部署教程
