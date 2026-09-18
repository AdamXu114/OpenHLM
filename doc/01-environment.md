# 01 环境与资源

## 1.1 机器与容器

| 项 | 值 |
| --- | --- |
| 主机 | `hpc`，4 × RTX 4090（每张 ~48 GB） |
| 容器名 | `xujinfan_dev` |
| 镜像 | `nvcr.io/nvidia/isaac-lab:2.3.2` |
| 容器 HOME | `/root` |
| 容器时区 | **UTC**（主机是 CST/UTC+8，`date` 会差 8 小时） |

**目录映射**（主机 ↔ 容器）：

```
/data0/xujinfan          ->  /workspace
```

所以主机上的 `/data0/xujinfan/OpenHLM`，在容器里就是 `/workspace/OpenHLM`。
本目录的文档（`doc/`）在容器里位于 `/workspace/OpenHLM/doc/`。

### 进入容器

```bash
docker exec -it xujinfan_dev bash
```

容器是 `sleep infinity` 常驻的，不会因为退出 exec 会话而停。

> ⚠️ **不要 `docker stop/restart xujinfan_dev`**。容器一停，里面所有 tmux 会话和训练进程全没。

## 1.2 GPU 预约（**最容易翻车的一步**）

hpc 主机上 4 张 4090 都**必须先预约才能用**。没预约就占用 GPU 的容器会被 `hpcd` 守护进程直接停掉：
先 SIGTERM，10 秒后 SIGKILL → **容器退出码 137，且 `OOMKilled=false`**。

> **看到 exit 137 先查预约，不要以为是内存不够。** 详见 [05-troubleshooting.md](05-troubleshooting.md#exit-137)。

### 预约

```bash
/usr/local/bin/hpc
```

TUI 操作：`tab` 切到 New Reservation → 空格勾选 GPU → 填 GPU、起止时间（`YYYY-MM-DD HH`）、用途 → 提交。
**无管理员授权码最多约 2 张卡。**

### 预约的坑

- 到期后只有 **15 分钟宽限**。
- **低利用率回收**：预约后连续 60 分钟 GPU 使用率 < 10% 就会被回收 —— **包括"完全没用"**。
  也就是说：预约 → 去准备数据 → 一小时后回来训练，预约可能已经没了，**一启动就被杀**。
  > 如果距上次真正用 GPU 超过一小时，**重新看一眼 TUI 里的预约还在不在**。
- 超出预约量不会立刻被杀（实测预约 20 GB，峰值跑到 34 GB 也活着），但别指望。

### 用之前永远先对齐

```bash
# 确认卡是空的
nvidia-smi

# 训练命令里永远显式指定（容器是 --gpus all，不设就会抢占别人的卡）
export CUDA_VISIBLE_DEVICES=<预约的卡号>
```

### 看预约数据库（不用 sudo）

```bash
# DB: /var/lib/hpcd/reservations.db
# HTTP API: unix socket /var/run/hpcd.sock  GET /reservations
docker run --rm -v /:/host:ro --entrypoint bash <img> -c "chroot /host /usr/bin/journalctl ..."
```

## 1.3 代理（下载模型 / wandb sync 用）<a id="proxy"></a>

**本地电脑执行**（建立反向隧道，把本地 7897 转发到服务器 17890）：

```bash
ssh -R 17890:127.0.0.1:7897 -p 9980 xujinfan@zz.irmv.top
```

**服务器/容器执行**（设置代理环境变量）：

```bash
export https_proxy=http://127.0.0.1:17890
export http_proxy=http://127.0.0.1:17890
```

## 1.4 Python 环境

**别用系统 python。** 项目自带的 venv：

```bash
/workspace/OpenHLM/src/openpi4OpenHLM/.venv/bin/python   # Python 3.11，uv venv
```

这是 **uv venv，里面没有 pip**。装包要用 uv：

```bash
/root/.local/bin/uv pip install --python /workspace/OpenHLM/src/openpi4OpenHLM/.venv/bin/python <package>
```

其他环境（按需）：

| 路径 | 用途 |
| --- | --- |
| `src/openpi4OpenHLM/.venv` | 训练 / 策略服务端（openpi 侧） |
| `src/GR00T-WholeBodyControl4OpenHLM/.venv_teleop` | 遥操作 / 数据采集 / 部署客户端（**当前容器里没有**，见 05） |

## 1.5 预训练权重

```
/workspace/openpi-assets/checkpoints/pi05_base_pytorch
```

> config 里原来写的 `~/.cache/...` 路径是错的 —— safetensors **不展开 `~`**，必须写绝对路径。

## 1.6 tmux

容器里已装 tmux 3.4（`/usr/bin/tmux`）。训练一律放 tmux 里跑，细节见 [03-training.md](03-training.md#33-tmux)。
