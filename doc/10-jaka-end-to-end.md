# 10 Jaka 全链路教程：从数据到训练再到部署

> 目标：把 Jaka 桌面抓取任务从**实机采集的数据**一路做到**机器人上跑 VLA 策略**。
> 部署命令对应实机 `base_link` 模型，启动和验证细节见 [11 本机部署](11-local-deploy.md)。路径以 `~/workspace/OpenHLM` 为根。
>
> 相关细节：[02 数据](02-data.md) · [03 训练](03-training.md) · [04 部署](04-deployment.md) ·
> [05 排错](05-troubleshooting.md) · [09 Jaka 文件核对](09-jaka-missing-files.md)

---

## 0. 链路总览

```
┌─────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│ 实机遥操作   │──►│ 合并 + 重排   │──►│ norm stats   │──►│ π0.5 微调     │
│ LeRobot 录制 │   │ merge 脚本   │   │ 统计归一化    │   │ train_pytorch│
└─────────────┘   └──────────────┘   └──────────────┘   └──────┬───────┘
                                                                 │ checkpoint
                                                                 ▼
┌─────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│ 机器人执行   │◄──│ 协议 v1 帧    │◄──│ 部署客户端    │◄──│ 策略服务端    │
│ (接收方自实现)│   │ ZMQ :28701   │   │ main.py      │   │ serve_policy │
└─────────────┘   └──────────────┘   │ websocket    │   │ :8000        │
                                     └──────────────┘   └──────────────┘
```

每一阶段的产物：

| 阶段 | 产物 | 位置 |
| --- | --- | --- |
| 采集 | 每个 session 一个 LeRobot v2.1 数据集 | `data/teleop_jaka_mf/<timestamp>/level-0/` |
| 转换/合并 | 训练用 repo（重排成"手臂优先"） | `<HF_LEROBOT_HOME>/jaka_pick/` |
| 统计 | `norm_stats.json` | `src/openpi4OpenHLM/assets/jaka_tabletop_pick/.../level-0/` |
| 训练 | `<exp>/<step>/`（含 `model.safetensors` + `assets/`） | `src/openpi4OpenHLM/checkpoints/jaka_tabletop_pick/` |
| 部署 | 策略服务 + 动作流 | 端口 `8000` / `28701` |

---

## 1. 前置准备

```bash
cd ~/workspace/OpenHLM/src/openpi4OpenHLM

# Python 环境（Python 3.11.18，uv venv，无 pip）
./.venv/bin/python -V
~/.local/bin/uv --version

# GPU
nvidia-smi          # 1 × RTX 3090 24GB

# 基座权重（训练起点）
ls ~/.cache/openpi/openpi-assets/checkpoints/pi05_base_pytorch/model.safetensors

# 数据（新采集转换的 40 episodes；作者发布的旧 47 episodes 在 data/teleop_jaka_mf/simple/...）
ls ~/workspace/OpenHLM/record_teleop/record_data/lerobot_train/jaka_pick/meta/info.json

# 部署用 checkpoint（可选：只做训练可跳过）
ls checkpoints/jaka_tabletop_pick/jaka_tabletop_pick_v1/30000/model.safetensors
```

缺什么补什么：[08 本机环境](08-local-machine.md) 有依赖安装与权重下载的完整步骤。

---

## 2. 数据：采集 → 合并 → 训练集

### 2.1 采集（实机）

用实机遥操作采集，每个 session 落成一个 **LeRobot v2.1** 数据集，目录形如：

```
data/teleop_jaka_mf/<timestamp>/level-0/
├── meta/{info.json, episodes.jsonl, tasks.jsonl, episodes_stats.jsonl}
├── data/chunk-XXX/episode_XXXXXX.parquet   # 图像 PNG 字节内嵌
└── videos/chunk-XXX/head_image_left/...    # 浏览副本（可选）
```

每条样本的字段要求（合并脚本会校验，不符会直接报错）：

| 字段 | 形状 | 说明 |
| --- | --- | --- |
| `state` | 30 | 27 关节 + root roll/pitch/yaw_vel（MuJoCo 关节序） |
| `actions` | 40 | 30 + anchor_lin_vel(3) + anchor_pos_w(3) + anchor_quat_w(4) |
| `head_image_left` | 224×224×3 | 单目头部相机，PNG 字节 |
| `timestamp` | 1 | LeRobot 合成（`frame_index/fps`），不用自己填 |

> 采集侧不在本仓库（硬件/遥操程序在机器人那边）。只要产出的 LeRobot 数据集满足上表即可。

### 2.2 合并 + 重排

把多个 session 合成一个训练集，并把关节顺序从"腿优先"重排成策略用的"手臂优先"：

```bash
cd ~/workspace/OpenHLM/src/openpi4OpenHLM

# 先 dry-run 看 episode 数、维度、排列
./.venv/bin/python examples/jaka/merge_teleop_jaka_mf.py \
  --source-dirs ~/workspace/OpenHLM/data/teleop_jaka_mf/<ts1>/level-0 \
                ~/workspace/OpenHLM/data/teleop_jaka_mf/<ts2>/level-0 \
  --hf-home ~/workspace/OpenHLM/data \
  --dry-run

# 确认无误后写入（会删除同名输出重建）
./.venv/bin/python examples/jaka/merge_teleop_jaka_mf.py \
  --source-dirs ~/workspace/OpenHLM/data/teleop_jaka_mf/<ts1>/level-0 \
                ~/workspace/OpenHLM/data/teleop_jaka_mf/<ts2>/level-0 \
  --hf-home ~/workspace/OpenHLM/data \
  --overwrite
```

输出：`$HF_LEROBOT_HOME/jaka_pick`（repo_id 必须与
`config.py::LeRobotJakaDataConfig.repo_id` 一致）。

> 直接用作者发布的数据集也行，本机已经下载好（47 eps / 11657 frames / 226 MB），见 [02](02-data.md) §2.3。

### 2.3 重算 norm stats

```bash
cd ~/workspace/OpenHLM/src/openpi4OpenHLM

HF_LEROBOT_HOME=~/workspace/OpenHLM/data \
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
./.venv/bin/python scripts/compute_norm_stats.py --config-name jaka_tabletop_pick
```

CPU 约 1 分钟，写到 `assets/jaka_tabletop_pick/jaka_pick/norm_stats.json`。

> ⚠️ **先读 [09](09-jaka-missing-files.md) §3**：仓库现有 stats 是 **delta 动作**统计，
> 而配置是绝对动作（`use_delta_joint_actions=False`）。重训前二选一：
> 重算成绝对动作统计，或把配置改成 delta。**只做部署则不要重算**。

---

## 3. 训练：π0.5 微调

> **训练不限定在本机**（可以在训练机 / 多卡机器上跑）。命令与原来训练机上的保持一致，
> **唯一可按显存调整的是 `--batch-size`**（以及步数类参数）；优化器与精度不要改。
> 路径按所在环境替换：训练机是 `/workspace/OpenHLM`，本机是 `~/workspace/OpenHLM`。

### 3.1 关于优化器（不要改）

`TrainConfig.pytorch_training_precision` 默认 `bfloat16`，模型构造时主权重会被转成 bf16
（`to_bfloat16_for_selected_params`），`torch.optim.AdamW` 的动量状态跟随参数也是 **bf16**，
即 **16-bit Adam**。

因此命令里**不加** `--use-8bit-adam`，也不用装 bitsandbytes。

### 3.2 启动

```bash
tmux new -s jakatrain
cd ~/workspace/OpenHLM/src/openpi4OpenHLM
mkdir -p logs

HF_LEROBOT_HOME=~/workspace/OpenHLM/data \
CUDA_VISIBLE_DEVICES=0 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
WANDB_MODE=offline PYTHONUNBUFFERED=1 \
./.venv/bin/python scripts/train_pytorch.py jaka_tabletop_pick \
  --exp-name jaka_tabletop_pick_v1 \
  --overwrite \
  --batch-size 8 \
  --num-train-steps 30000 \
  --save-interval 5000 \
  --lr-schedule.decay-steps 30000 \
  --lr-schedule.warmup-steps 1000 \
  2>&1 | tee logs/jaka_v1_$(date +%m%d_%H%M).log
```

脱离 tmux：`Ctrl-b` 松手再按 `d`。

### 3.3 观察与调参

```bash
# 不进 tmux 偷看
tmux capture-pane -pt jakatrain | tail -50

# 只看 loss
tail -f logs/jaka_v1_*.log | grep --line-buffered -o 'step=[0-9]* loss=[0-9.]*.*'

# 显存
watch -n 2 nvidia-smi
```

| 症状 | 处理 |
| --- | --- |
| CUDA OOM | 只降 `--batch-size`（例如 4/2/1），不要动优化器和精度 |
| loss 不动 / NaN | 先确认 norm stats 与动作空间一致（见 2.3），再降 lr |
| 整机卡死 | 系统内存只有 31 GB，别同时跑别的重内存程序 |

产出：`checkpoints/jaka_tabletop_pick/jaka_tabletop_pick_v1/<step>/`
（`model.safetensors` + `metadata.pt` + `assets/.../norm_stats.json`）。

---

## 4. 部署前验证（可选但推荐）

### 4.1 服务端能否加载

```bash
cd ~/workspace/OpenHLM/src/openpi4OpenHLM
CUDA_VISIBLE_DEVICES=0 ./.venv/bin/python scripts/serve_policy.py \
  --env JAKA --num-steps 10 \
  policy:checkpoint --policy.config=jaka_tabletop_pick \
  --policy.dir=checkpoints/jaka_tabletop_pick/jaka_pick_real_v1/30000
```

看到 `server listening on 0.0.0.0:8000` 即可 `Ctrl+C`。

### 4.2 协议自测

```bash
cd ~/workspace/OpenHLM/src/GR00T-WholeBodyControl4OpenHLM/openpi-eval
PYTHONPATH=~/workspace/OpenHLM/src/GR00T-WholeBodyControl4OpenHLM \
~/workspace/OpenHLM/src/openpi4OpenHLM/.venv/bin/python jaka_tabletop_env_test.py
# 期望：All jaka wire-contract tests passed.
```

---

## 5. 部署：策略服务 + 客户端 + 机器人接收方

### 5.1 链路与端口

| 端口 | 方向 | 内容 |
| --- | --- | --- |
| `8000` | 客户端 → 服务端 | websocket + msgpack，请求观测 / 返回动作 chunk |
| `28701` | 客户端 → 机器人 | 协议 v1 二进制动作帧（2 帧滑窗，30 Hz） |
| `28711` | 观测 → 客户端 | 实测 state（裸 JSON） |
| `28712` | 观测 → 客户端 | head 图（`[w,h,c]` int32 头 + RGB） |

实机客户端等待这两路有效观测，持续缺失会报告超时。离线合成观测使用 `--mock`。

### 5.2 起服务端（终端 1 / tmux 窗口 1）

```bash
cd ~/workspace/OpenHLM/src/openpi4OpenHLM
CUDA_VISIBLE_DEVICES=0 ./.venv/bin/python scripts/serve_policy.py --env JAKA --num-steps 10 \
  policy:checkpoint --policy.config=jaka_tabletop_pick \
  --policy.dir=checkpoints/jaka_tabletop_pick/jaka_pick_real_v1/30000
```

### 5.3 起客户端（终端 2 / tmux 窗口 2）

```bash
cd ~/workspace/OpenHLM/src/GR00T-WholeBodyControl4OpenHLM/openpi-eval
export PYTHONPATH=~/workspace/OpenHLM/src/GR00T-WholeBodyControl4OpenHLM

~/workspace/OpenHLM/src/openpi4OpenHLM/.venv/bin/python main.py \
  --env jaka_tabletop_pick \
  --remote-host 127.0.0.1 --remote-port 8000 \
  --jaka-state-address "${ROBOT_IP:?请设置机器人 IP}" --jaka-head-image-address "${CAMERA_IP:-$ROBOT_IP}" \
  --jaka-motion-zmq-port 28701 \
  --max-steps 100000
```

- 机器人还没接：加 `--mock --use-fake-policy` 可空跑控制循环；
- SSH / 无显示：加 `--no-opencv-visualize`；本地桌面要看画面则需 GUI 版 OpenCV（见 [08](08-local-machine.md)）。

### 5.4 机器人侧接收方（自实现）

客户端发出的是 **协议 v1**：

```
[topic "pose"(4B)][1280B JSON 头，\0 填充][按头中字段顺序拼接的二进制负载]
必需字段：joint_pos(N,27) f32 | frame_index(N,) i64 | body_pos_w(N,3) f32 | body_quat_w(N,4) f32(wxyz)
```

要点：

- `joint_pos` 是 **SIM/MuJoCo 关节序（腿优先）**，客户端已重排好；
- `body_*` 描述 `base_link` 世界位姿，包头声明 `joint_order=jaka` 和发布实例的 `stream_id`；
- 同一实例跨推理块保留帧号和积分姿态。控制器依帧号构建 30 Hz 源时间轴，短暂停包时窗口可以暂时固定；
- 机器人进入 `jakaminimimic_superzmq`，用 `motion_backend: vla_pose_v1` 连接客户端 PC 的 28701。新会话或默认连续 1 s 没有新帧会锁存参考。复位后重新开始任务的步骤见 [11 本机部署](11-local-deploy.md)。

---

## 6. 一次完整部署的操作顺序

```bash
# 终端 1：服务端
cd ~/workspace/OpenHLM/src/openpi4OpenHLM
CUDA_VISIBLE_DEVICES=0 ./.venv/bin/python scripts/serve_policy.py --env JAKA --num-steps 10 \
  policy:checkpoint --policy.config=jaka_tabletop_pick \
  --policy.dir=checkpoints/jaka_tabletop_pick/jaka_pick_real_v1/30000

# 终端 2：客户端（等终端 1 打印 listening 后再起）
cd ~/workspace/OpenHLM/src/GR00T-WholeBodyControl4OpenHLM/openpi-eval
export PYTHONPATH=~/workspace/OpenHLM/src/GR00T-WholeBodyControl4OpenHLM
~/workspace/OpenHLM/src/openpi4OpenHLM/.venv/bin/python main.py \
  --env jaka_tabletop_pick --remote-host 127.0.0.1 --remote-port 8000 \
  --jaka-state-address "${ROBOT_IP:?请设置机器人 IP}" --jaka-head-image-address "${CAMERA_IP:-$ROBOT_IP}" \
  --jaka-motion-zmq-port 28701 --max-steps 100000

# 机器人端先进入 SuperZmq，订阅 tcp://<客户端 PC 的 IP>:28701，并发布 state；相机另行启动。
```

停止本轮参考发布后，按客户端提示输入 `n` 退出。机器人复位并开始新任务时，从新的初始位姿重启客户端并重新进入 SuperZmq。

---

## 7. 各阶段验收清单

| 阶段 | 通过标准 |
| --- | --- |
| 数据 | `meta/info.json` 里 `total_episodes/total_frames` 符合预期；state 30 / actions 40 |
| 合并 | dry-run 打印的 episode 数与源数据一致；`--dry-run` 无 name 校验报错 |
| norm stats | 文件生成，`state` 30 维 / `actions` 33 维 |
| 训练 | 日志有 loss，`checkpoints/<exp>/<step>/` 有 `model.safetensors` + `assets/` |
| 服务端 | `server listening on 0.0.0.0:8000` |
| 客户端 | 连上服务端、warmup 完成、30 Hz 发帧不报错 |
| 协议 | `jaka_tabletop_env_test.py` 全过 |
| 机器人 | 收到动作且不报 `binary decode failed` / `joint dim mismatch` |

---

## 8. 常见坑（按出现频率排序）

1. **norm stats 与动作空间不一致**（delta vs absolute）→ [09 §3](09-jaka-missing-files.md)
2. **`pytorch_weight_path` 与所在机器不符** → 训练机上不用管；只有本机临时训练才用 `--pytorch-weight-path` 覆盖
3. **误加 `--use-8bit-adam`** → 不需要，默认 bf16 参数 + bf16 AdamW 状态就是 16-bit
4. **`HF_LEROBOT_HOME` 没设** → 数据集找不到（lerobot 不认 `LEROBOT_HOME`）
5. **`cv2.imshow` 报 function not implemented** → headless OpenCV，见 [08](08-local-machine.md)
6. **客户端重启后参考锁存** → 新的 `stream_id` 表示新的积分原点，需退出并重新进入 SuperZmq
7. **8000 端口 `address already in use`** → 多半是 TIME_WAIT，等 1 分钟或换端口

---

## 9. 一页命令汇总

```bash
# 环境
cd ~/workspace/OpenHLM/src/openpi4OpenHLM
PY=./.venv/bin/python; UV=~/.local/bin/uv

# 数据合并
$PY examples/jaka/merge_teleop_jaka_mf.py --source-dirs <dirs...> --hf-home ~/workspace/OpenHLM/data --dry-run
$PY examples/jaka/merge_teleop_jaka_mf.py --source-dirs <dirs...> --hf-home ~/workspace/OpenHLM/data --overwrite

# norm stats（CPU）
HF_LEROBOT_HOME=~/workspace/OpenHLM/data JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
  $PY scripts/compute_norm_stats.py --config-name jaka_tabletop_pick

# 训练
HF_LEROBOT_HOME=~/workspace/OpenHLM/data CUDA_VISIBLE_DEVICES=0 \
XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_MODE=offline PYTHONUNBUFFERED=1 \
  $PY scripts/train_pytorch.py jaka_tabletop_pick --exp-name jaka_tabletop_pick_v1 \
  --overwrite --batch-size 8 --num-train-steps 30000 --save-interval 5000 \
  --lr-schedule.decay-steps 30000 --lr-schedule.warmup-steps 1000 \
  2>&1 | tee logs/jaka_v1_$(date +%m%d_%H%M).log

# 服务端
CUDA_VISIBLE_DEVICES=0 $PY scripts/serve_policy.py --env JAKA --num-steps 10 \
  policy:checkpoint --policy.config=jaka_tabletop_pick \
  --policy.dir=checkpoints/jaka_tabletop_pick/jaka_pick_real_v1/30000

# 客户端
cd ~/workspace/OpenHLM/src/GR00T-WholeBodyControl4OpenHLM/openpi-eval
PYTHONPATH=~/workspace/OpenHLM/src/GR00T-WholeBodyControl4OpenHLM \
~/workspace/OpenHLM/src/openpi4OpenHLM/.venv/bin/python main.py \
  --env jaka_tabletop_pick --remote-host 127.0.0.1 --remote-port 8000 \
  --jaka-state-address "${ROBOT_IP:?请设置机器人 IP}" --jaka-head-image-address "${CAMERA_IP:-$ROBOT_IP}" \
  --jaka-motion-zmq-port 28701 --max-steps 100000
```
