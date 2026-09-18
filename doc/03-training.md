# 03 训练

## 3.1 前置检查（每次开训前）

```bash
# 1) 主机上确认 GPU 预约还在（超过 1 小时没用过必须重新确认，见 01）
/usr/local/bin/hpc

# 2) 确认目标卡是空的
nvidia-smi

# 3) 数据集和 norm stats 是最新的（改过数据就必须重算，见 02）
ls /workspace/OpenHLM/data/teleop_jaka_mf/simple/JakaTabletopPickTeleop-v0/level-0
ls /workspace/OpenHLM/src/openpi4OpenHLM/assets/jaka_tabletop_pick/
```

## 3.2 训练命令

**完整版（推荐直接抄）：**

```bash
docker exec -it xujinfan_dev bash
tmux new -s jakatrain

cd /workspace/OpenHLM/src/openpi4OpenHLM

WANDB_MODE=offline \
PYTHONUNBUFFERED=1 \
HF_LEROBOT_HOME=/workspace/OpenHLM/data \
CUDA_VISIBLE_DEVICES=0 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
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
```

**环境变量**（少一个都可能出问题）：

| 变量 | 为什么必须 |
| --- | --- |
| `HF_LEROBOT_HOME=/workspace/OpenHLM/data` | vendored lerobot **不认** `LEROBOT_HOME`，不设直接报错 |
| `CUDA_VISIBLE_DEVICES=<卡号>` | 容器是 `--gpus all`，不设会抢别人的卡（→ 被 hpcd 杀） |
| `XLA_PYTHON_CLIENT_PREALLOCATE=false` | 否则 jax 预分配整卡 |
| `WANDB_MODE=offline` | 容器没有 wandb 凭据，在线模式会卡/失败 |
| `PYTHONUNBUFFERED=1` | 否则 tee 出来的日志是块缓冲，看不到实时进度 |

**关键参数**：

| 参数 | 说明 |
| --- | --- |
| `--exp-name <name>` | 实验名，决定 `checkpoints/jaka_tabletop_pick/<name>/` |
| `--overwrite` | 覆盖同名实验目录；**注意会删掉旧的** |
| `--use-8bit-adam` | **20 GB 预约必须加**（全精度 AdamW 要 35–40 GB） |
| `--batch-size 8` | 单卡 batch |
| `--num-train-steps` / `--save-interval` | 总步数 / 存 ckpt 间隔 |
| `--lr-schedule.decay-steps` / `--lr-schedule.warmup-steps` | 学习率调度，通常等于总步数 / 1000 |
| `--resume` | 从最新 ckpt 继续 —— **目前有 bug，见 3.6** |

> 训练参数走的是 `TrainConfig` dataclass + tyro，所以 flag 就是字段名（下划线转连字符），
> 嵌套字段用点号：`--lr-schedule.warmup-steps`。

**多卡**（需要 2 张预约）：

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=2 \
  scripts/train_pytorch.py jaka_tabletop_pick --exp-name <name> ...
```

## 3.3 tmux

训练**必须**放 tmux 里，否则断 SSH / 关终端就没了。

```bash
tmux new -s jakatrain     # 建会话
```

| 操作 | 命令 |
| --- | --- |
| **脱离**（训练继续跑） | `Ctrl-b`，松手，再按 `d` |
| 列出会话 | `tmux ls` |
| 重连 | `docker exec -it xujinfan_dev bash` → `tmux attach -t jakatrain` |
| **不进 tmux 偷看**（推荐） | `tmux capture-pane -pt jakatrain \| tail -50` |
| 翻历史输出 | 进去后 `Ctrl-b` `[`，方向键/PgUp 翻，`q` 退出 |

**为什么退出终端不会中断**：tmux server 启动后 daemonize，**PPID 变成 1**，和 `docker exec` 的会话彻底解耦。

```
385  ppid=1     tmux new -s jakatrain     ← server，挂在容器 init 上
397  ppid=386   train_pytorch.py ...      ← 训练进程，父进程是 tmux pane
```

> ⚠️ **三个别做**
> 1. 不要在 tmux 里按 `Ctrl-d` 或敲 `exit` —— 那是关掉 pane 的 shell，前台训练进程一起死。要脱离就 `Ctrl-b d`。
> 2. 别敲 `tmux kill-session` / `tmux kill-server`。
> 3. 别 `docker restart xujinfan_dev` —— tmux 活在容器里，容器一停全没。

## 3.4 日志

**默认不写日志文件** —— `init_logging()` 只挂了 `StreamHandler`（stdout），唯一的落盘记录是 wandb。
所以**每次训练都要自己 `2>&1 | tee logs/<name>_$(date +%m%d_%H%M).log`**。

日志格式（`<时间> [级别] ... (pid:文件:行号)`）：

```
# 每 100 步
07:01:06.817 [I] step=4100 loss=0.0527 lr=9.76e-05 grad_norm=0.17 time=326.5s

# 每 20 步
07:01:07.816 [I] Evaluation at step 4100: sample_mse_10steps=0.029570 sample_mse_1step=0.020718 eval_time=1.00s

# 显存
03:18:51.034 [I] Step 1 (after_backward): GPU memory - allocated: 20.92GB, reserved: 24.95GB, peak_reserved: 24.95GB
```

**日志里混了大量 tqdm 进度条**（每步重复刷，不带换行），直接 `tail` 会被淹没。过滤：

```bash
LOG=/workspace/OpenHLM/src/openpi4OpenHLM/logs/jaka_v1_0917_0314.log

# 只看进度行
grep -o 'step=[0-9]* loss=[0-9.]* lr=[0-9.e-]* grad_norm=[0-9.]*' "$LOG" | tail -20

# 只看 eval 曲线
grep -o 'Evaluation at step [0-9]*: sample_mse_10steps=[0-9.]* sample_mse_1step=[0-9.]*' "$LOG"

# 实时跟
tail -f "$LOG" | grep --line-buffered -o 'step=[0-9]* loss=[0-9.]*.*'
```

**wandb**：容器有网络到 api.wandb.ai 但**没有凭据**（`/root/.netrc` 和 `~/.config/wandb/` 都不存在），
所以自动落在 `wandb/offline-run-*`。offline run **本地没有查看器**，想看曲线必须：

```bash
cd /workspace/OpenHLM/src/openpi4OpenHLM
./.venv/bin/wandb login      # 先配代理，见 01
./.venv/bin/wandb sync wandb/offline-run-*   # 建议等训练跑完再 sync
```

> 容器时钟是 **UTC**，比主机早 8 小时。`$(date +%m%d_%H%M)` 出来的文件名会看着"错"，是正常的。

## 3.5 Checkpoint

```
checkpoints/jaka_tabletop_pick/<exp-name>/<step>/
```

每个 `step` 目录里 `metadata.pt` 只有 `global_step` / 完整 config / 时间戳，**没有 loss 历史**。
`save_optimizer=False`，所以 **ckpt 里没有优化器状态**，resume 会重置 Adam 的动量。

**用哪个 ckpt 推理**：改 [04-deployment.md](04-deployment.md) 里的路径，或用 `policy:checkpoint --policy.dir=` 显式指定。

## 3.6 已知问题：`--resume` 会崩

`--resume` + wandb 开启时，会在 `checkpoints/<cfg>/<exp>/wandb_id.txt` 上抛 `FileNotFoundError`
（`scripts/train_pytorch.py` 约 85 行，resume 时无条件读）。这个文件只有在原 run 开了 wandb 时才存在。

**目前的做法：不用 `--resume`，直接 `--overwrite` 重跑。** 如果要修，加 4 行 fallback 即可（尚未应用）。

## 3.7 action_dim=33 的权重处理

`action_dim=33 > 32` 会触发 `scripts/train_pytorch.py` 里的 `weight_surgery`：
拷贝预训练 `action_in_proj` / `action_out_proj` 的前 32 维，第 33 维用 Xavier 初始化。
这是**自动的**，不用手动干预，但换 action 维度时要记得它存在。

预训练权重：`/workspace/openpi-assets/checkpoints/pi05_base_pytorch`
