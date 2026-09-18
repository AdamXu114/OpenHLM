# OpenHLM 常用命令文档

本目录记录**我们在 OpenHLM 上实际使用的命令**，按流程阶段整理，供后续项目直接复用。

> 上游官方文档在 `src/openpi4OpenHLM/docs/`（docker / norm_stats / remote_inference）和根目录 `README.md`，
> 那些是 openpi 通用说明；本目录写的是**我们这台机器、这个容器里的实际做法**和踩过的坑。

## 目录

| 文档 | 内容 |
| --- | --- |
| [01-environment.md](01-environment.md) | 容器进入、GPU 预约、代理、路径映射、Python 环境 |
| [02-data.md](02-data.md) | 数据录制 → 合并重排 → norm stats 计算 |
| [03-training.md](03-training.md) | 训练命令、日志、tmux 保活、checkpoint 管理 |
| [04-deployment.md](04-deployment.md) | 策略服务端 + 仿真/真机推理部署 |
| [05-troubleshooting.md](05-troubleshooting.md) | 踩坑记录与排查方法（**建议先读**） |
| [06-cheatsheet.md](06-cheatsheet.md) | 一页速查表，复制粘贴用 |

另有 [../src/openpi4OpenHLM/inference.md](../src/openpi4OpenHLM/inference.md)（上游推理流程）、
`~/Base_bash.md`（主机层基础命令：进容器 / 预约显卡 / 翻墙）。

## 5 分钟跑通一次训练

```bash
# 0) 主机上：预约 GPU（这一步不做，容器会被杀，见 05）
/usr/local/bin/hpc

# 1) 进容器
docker exec -it xujinfan_dev bash

# 2) 起 tmux（训练必须放 tmux，见 03）
tmux new -s jakatrain

# 3) 训练
cd /workspace/OpenHLM/src/openpi4OpenHLM
export HF_LEROBOT_HOME=/workspace/OpenHLM/data
export CUDA_VISIBLE_DEVICES=<你预约的那张卡>
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export WANDB_MODE=offline
export PYTHONUNBUFFERED=1

./.venv/bin/python scripts/train_pytorch.py jaka_tabletop_pick \
  --exp-name jaka_tabletop_pick_v1 \
  --overwrite \
  --batch-size 8 \
  --num-train-steps 30000 \
  --save-interval 5000 \
  --use-8bit-adam \
  --lr-schedule.decay-steps 30000 \
  --lr-schedule.warmup-steps 1000 \
  2>&1 | tee logs/jaka_v1_$(date +%m%d_%H%M).log

# 4) 脱离：Ctrl-b，松手，再按 d（不要按 Ctrl-d / exit！）
```

查看进度：

```bash
# 不进 tmux 偷看（推荐）
tmux capture-pane -pt jakatrain | tail -50

# 或者看文本日志
tail -f /workspace/OpenHLM/src/openpi4OpenHLM/logs/jaka_v1_*.log
```

## 当前项目状态（2026-09-18）

- 任务：`jaka_tabletop_pick`（Jaka 机械臂桌面抓取）
- 训练 run：`jaka_tabletop_pick_v1`，已跑到 step 30000
- Checkpoint：`src/openpi4OpenHLM/checkpoints/jaka_tabletop_pick/jaka_tabletop_pick_v1/<step>/`（5000/10000/15000/20000/25000/30000）
- 部署侧：`src/GR00T-WholeBodyControl4OpenHLM/openpi-eval/jaka_tabletop_env.py`（ZMQ 端口 28701），已通过 mock 测试，真机端到端待验证

## 约定

- 文档中**不带前缀的路径**都在容器内，即 `/workspace/...`；主机上同一批文件在 `/data0/xujinfan/...`
- 命令统一从 `/workspace/OpenHLM/src/openpi4OpenHLM` 出发（除特别说明）
- `<尖括号>` 是需要你替换的参数
