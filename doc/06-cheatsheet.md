# 06 速查表

复制粘贴用。详细说明见各章节链接。

## 环境

```bash
docker exec -it xujinfan_dev bash              # 进容器
/usr/local/bin/hpc                             # 主机上预约 GPU（用前必做）

export HF_LEROBOT_HOME=/workspace/OpenHLM/data
export CUDA_VISIBLE_DEVICES=0                  # 改成实际预约的卡
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export WANDB_MODE=offline
export PYTHONUNBUFFERED=1
```

路径映射：主机 `/data0/xujinfan` ↔ 容器 `/workspace`

代理（本地先建隧道 `ssh -R 17890:127.0.0.1:7897 -p 9980 xujinfan@zz.irmv.top`）：

```bash
export https_proxy=http://127.0.0.1:17890
export http_proxy=http://127.0.0.1:17890
```

装包（venv 没有 pip）：

```bash
/root/.local/bin/uv pip install --python /workspace/OpenHLM/src/openpi4OpenHLM/.venv/bin/python <pkg>
```

## 从零跑一次训练

```bash
tmux new -s jakatrain
cd /workspace/OpenHLM/src/openpi4OpenHLM

# --- 只在改过数据时做 ---
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
  ./.venv/bin/python scripts/compute_norm_stats.py --config-name jaka_tabletop_pick
# ------------------------

HF_LEROBOT_HOME=/workspace/OpenHLM/data \
CUDA_VISIBLE_DEVICES=0 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
WANDB_MODE=offline PYTHONUNBUFFERED=1 \
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

## 数据合并/reorder

```bash
cd /workspace/OpenHLM/src/openpi4OpenHLM
PY=/workspace/OpenHLM/src/openpi4OpenHLM/.venv/bin/python

$PY examples/jaka/merge_teleop_jaka_mf.py --dry-run     # 先看
$PY examples/jaka/merge_teleop_jaka_mf.py --overwrite   # 再写
```

## tmux

```bash
tmux new -s jakatrain                  # 建
tmux ls                                # 列
tmux attach -t jakatrain               # 进
tmux capture-pane -pt jakatrain | tail -50   # 不进会话偷看（推荐）
# 脱离：Ctrl-b 松手 d      ← 不是 Ctrl-d，不是 exit！
```

## 看日志

```bash
LOG=logs/jaka_v1_0917_0314.log

grep -o 'step=[0-9]* loss=[0-9.]* lr=[0-9.e-]* grad_norm=[0-9.]*' "$LOG" | tail -20
grep -o 'Evaluation at step [0-9]*: sample_mse_10steps=[0-9.]* sample_mse_1step=[0-9.]*' "$LOG"
tail -f "$LOG" | grep --line-buffered -o 'step=[0-9]* loss=[0-9.]*.*'
```

## 推理部署

```bash
# 服务端
cd /workspace/OpenHLM/src/openpi4OpenHLM
./.venv/bin/python scripts/serve_policy.py --env JAKA --num-steps 10
# 或指定 ckpt：
#   policy:checkpoint --policy.config=jaka_tabletop_pick \
#                     --policy.dir=checkpoints/jaka_tabletop_pick/jaka_tabletop_pick_v1/30000

# 客户端（容器内，三要素：PYTHONPATH + --no-opencv-visualize + 喂 stdin）
docker exec -i -w /workspace/OpenHLM/src/GR00T-WholeBodyControl4OpenHLM/openpi-eval \
  -e PYTHONPATH=/workspace/OpenHLM/src/GR00T-WholeBodyControl4OpenHLM \
  xujinfan_dev /workspace/OpenHLM/src/openpi4OpenHLM/.venv/bin/python main.py \
  --env jaka_tabletop_pick --mock --use-fake-policy --max-steps 200 --no-opencv-visualize

# 协议测试
docker exec -i -w /workspace/OpenHLM/src/GR00T-WholeBodyControl4OpenHLM/openpi-eval \
  -e PYTHONPATH=/workspace/OpenHLM/src/GR00T-WholeBodyControl4OpenHLM \
  xujinfan_dev /workspace/OpenHLM/src/openpi4OpenHLM/.venv/bin/python jaka_tabletop_env_test.py
```

## 关键路径

| 内容 | 路径 |
| --- | --- |
| 训练仓库根 | `/workspace/OpenHLM/src/openpi4OpenHLM` |
| 训练 venv | `src/openpi4OpenHLM/.venv/bin/python` |
| 数据集（训练用） | `data/teleop_jaka_mf/simple/JakaTabletopPickTeleop-v0/level-0` |
| 原始录制 | `data/teleop_jaka_mf/<timestamp>/level-0` |
| norm stats | `src/openpi4OpenHLM/assets/jaka_tabletop_pick/.../norm_stats.json` |
| 预训练权重 | `/workspace/openpi-assets/checkpoints/pi05_base_pytorch` |
| checkpoints | `src/openpi4OpenHLM/checkpoints/jaka_tabletop_pick/<exp>/<step>` |
| 部署客户端 | `src/GR00T-WholeBodyControl4OpenHLM/openpi-eval/main.py` |
| Jaka env | `src/GR00T-WholeBodyControl4OpenHLM/openpi-eval/jaka_tabletop_env.py` |
| 协议测试 | `src/GR00T-WholeBodyControl4OpenHLM/openpi-eval/jaka_tabletop_env_test.py` |

## 数字速记

| 项 | 值 |
| --- | --- |
| state 维度 | 30（dof27 + rpy3，**无速度**） |
| actions 维度（录制） | 40（30 + anchor_lin_vel3 + anchor_pos_w3 + anchor_quat_w4） |
| actions 维度（训练） | 33（dof27 + rpy3 + base_vel3） |
| 训练用排列 | 手臂优先：arm_l, arm_r, leg_l, leg_r, waist, neck, root |
| Jaka ZMQ 端口 | 28701（motion）/ 28702（state）/ 28703（head image） |
| 策略服务端口 | 8000 |
| 控制频率 / 开环 horizon | 30 Hz / 25 步（horizon 50） |
| 当前 ckpt | `jaka_tabletop_pick_v1` step 30000 |

## 应急

```bash
# 容器被杀了？先查是不是 GPU 预约问题
docker inspect xujinfan_dev --format '{{.State.OOMKilled}}'
/usr/local/bin/hpc

# 训练挂了想重跑（没有 --resume，直接覆盖）
# 就上面那条训练命令，保持 --exp-name 不变、--overwrite 即可
```
