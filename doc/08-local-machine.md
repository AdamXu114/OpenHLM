# 本机环境配置记录（RTX 3090 / Ubuntu 24.04）

> 本文记录 **2026-11-07 在本机（非服务器、非容器）** 上按官方 README 配置 OpenHLM 的完整过程。
> 与 `01-environment.md` 里的容器 / 显卡预约流程无关，那套东西在本机不需要。

## 配置范围

| 组件 | 状态 | 说明 |
| --- | --- | --- |
| `src/openpi4OpenHLM`（VLA 训练 / 推理） | ✅ 已完成 | 本次配置目标：Python 环境 + pi05_base 权重 |
| `src/GR00T-WholeBodyControl4OpenHLM`（机器人控制 / 仿真 / 遥操） | ❌ 未配置 | 需要 TensorRT 10.13、sudo 装系统依赖、Git LFS 资产，且要有 G1 / VR 硬件 |
| `src/HuMI4OpenHLM`（人类数据采集） | ❌ 未配置 | 仅数据采集用，需要 `exiftool`（sudo）与 Vive tracker |

## 机器现状

- Ubuntu 24.04.5 LTS（2026-11-07 从 22.04 升级），内核 6.8.0-149，16 核，31 GB 内存（swap 仅 1 GB），`/` 剩余约 360 GB
- GPU：RTX 3090 24 GB，驱动 595.99.02（open kernel module，DKMS 已为 6.8.0-149 重建），系统装的是 CUDA 12.8 toolkit
- Python：系统/conda 是 3.14，**openpi 用 uv 自带的 CPython 3.11.18**，不依赖系统 Python

## 最终结果

| 项目 | 路径 |
| --- | --- |
| uv 可执行文件 | `~/.local/bin/uv`（0.13.7） |
| Python 虚拟环境 | `src/openpi4OpenHLM/.venv`（Python 3.11.18，torch 2.9.0+cu128，jax 0.5.3） |
| JAX 原始权重 | `~/.cache/openpi/openpi-assets/checkpoints/pi05_base`（12.4 GB） |
| PyTorch 权重 | `~/.cache/openpi/openpi-assets/checkpoints/pi05_base_pytorch`（7.0 GB，bf16） |
| LeRobot 数据集默认目录 | `~/.cache/huggingface/lerobot` |

## 复现步骤

### 1. 恢复 third_party 子模块

仓库虽然记录了 `third_party/{lerobot,dlimp,Muon}` 的 gitlink，但**没有提交 `.gitmodules`**，所以 `git submodule update --init` 会直接报错。本机已补上 `.gitmodules` 并按原 commit 拉取：

```bash
cd <repo>
git init third_party/lerobot && git -C third_party/lerobot remote add origin https://github.com/huggingface/lerobot.git
git -C third_party/lerobot fetch --depth 1 origin 0cf864870cf29f4738d3ade893e6fd13fbd7cdb5
git -C third_party/lerobot checkout FETCH_HEAD
# dlimp:   https://github.com/kvablack/dlimp.git        ad72ce3a9b414db2185bc0b38461d4101a65477a
# Muon:    https://github.com/KellerJordan/Muon.git     f98f1cacc0263b04290753e32be8d498c1efc806
```

### 2. 安装 uv 并同步依赖

```bash
uv sync
uv pip install -e .
cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/
```

`GIT_LFS_SKIP_SMUDGE=1` 在 `uv sync` 时是必需的（lerobot 是 LFS 仓库）。

### 3. 下载并转换 pi05_base 权重

```bash
uv run examples/convert_jax_model_to_pytorch.py \
  --checkpoint_dir ~/.cache/openpi/openpi-assets/checkpoints/pi05_base \
  --config_name pi05_aloha \
  --output_path ~/.cache/openpi/openpi-assets/checkpoints/pi05_base_pytorch
```

**注意：官方这条命令在本机会把机器打爆（见下方坑 3），实际使用的是 `tools/convert_pi05_lowmem.py`。**

## 踩过的坑（重要）

### 1. 仓库没有 `.gitmodules`，也没有 `uv.lock`

- `uv.lock` 被 `.gitignore` 忽略，所以 `uv sync` 会解析到**最新**依赖，而不是服务器上那套。
- 已补 `.gitmodules`；依赖版本靠 `pyproject.toml` 约束兜底。

### 2. `pyarrow` 与 `numpy<2` 冲突

openpi 固定了 `numpy<2.0.0`（`pyproject.toml`），但解析出的 `pyarrow 26` 要求 NumPy ≥ 2，
一执行 `import datasets` 就报：

```
ImportError: pyarrow requires NumPy 2.0 or newer, found 1.26.4
```

已在 `pyproject.toml` 的 `[tool.uv] override-dependencies` 里加上 `"pyarrow<21"`（实际安装 20.0.0）。
**不要删这条**，否则训练脚本 import 数据模块时会挂。

### 3. 官方转换脚本需要 >28 GB 内存，本机被 OOM 打死

`examples/convert_jax_model_to_pytorch.py` 会同时持有：

- float32 JAX 参数树：12.4 GB
- CPU 上的 float32 torch 模型：13.2 GB

两者叠加超过本机 31 GB 内存（swap 只有 1 GB），实测直接把整机拖死。

`tools/convert_pi05_lowmem.py` 是等价但低内存的版本：

1. 切片完 JAX 参数后立即 `del` 释放；
2. torch 模型直接以 bf16 建在 **GPU** 上（6.6 GB 显存），不再占 13 GB 内存；
3. `load_state_dict` 之后释放 host 端 state dict。

峰值 host 内存约 13 GB，可安全重跑。转换产物 `model.safetensors` 为 bf16（7.0 GB）。

### 4. 转换脚本复制 assets 的路径是错的

上游代码从 `checkpoint_dir/../assets` 复制 norm stats，但实际资产在
`checkpoints/pi05_base/assets/`。转换完成后需手动补：

```bash
cp -r ~/.cache/openpi/openpi-assets/checkpoints/pi05_base/assets \
      ~/.cache/openpi/openpi-assets/checkpoints/pi05_base_pytorch/assets
```

否则 `create_trained_policy` 找不到归一化统计。**本次已复制好。**

### 5. `openhlm_example` 里的权重路径还是服务器的

`src/openpi/training/config.py` 中 `openhlm_example` / `jaka_tabletop_pick` 的
`pytorch_weight_path` 是按**训练机**写的 `/workspace/openpi-assets/checkpoints/pi05_base_pytorch`。

- 在训练机上训练：不用管，路径本来就对；
- 只在本机临时试跑：用命令行覆盖（safetensors 不做 `~` 展开，要绝对路径）：

```
--pytorch-weight-path /home/xjf/.cache/openpi/openpi-assets/checkpoints/pi05_base_pytorch
```

`serve_policy` 读 checkpoint 内的资源，不受这条配置影响。

### 6. 内存只有 31 GB / swap 1 GB

任何多进程训练或大 batch 都可能触发 OOM。建议：

- 训练先用小 `--batch_size`（如 8～16）+ 单卡，`nvidia-smi` / `free -g` 观察；
- 需要长时间任务时用 `tmux`，避免终端断开；
- 不要在转换 / 训练同时跑浏览器等大内存程序。

## 验证命令

```bash
cd src/openpi4OpenHLM

# 依赖与 GPU
.venv/bin/python -c "import torch, jax; print(torch.cuda.get_device_name(0), jax.devices())"

# 权重可加载 + 前向推理（约 30 s，首次有 triton 编译）
.venv/bin/python -c "
from openpi.training import config
from openpi.policies import policy_config, aloha_policy
cfg = config.get_config('pi05_aloha')
p = policy_config.create_trained_policy(cfg, '$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_base_pytorch')
print(p.infer(aloha_policy.make_aloha_example())['actions'].shape)
"
# 期望输出 (50, 14)
```

本次实测：`torch 2.9.0+cu128`、`jax 0.5.3` 均识别到 RTX 3090，推理返回 `(50, 14)`。

### 系统升级到 Ubuntu 24.04 后的复核（2026-11-07 20:00）

| 检查项 | 结果 |
| --- | --- |
| 发行版 / 内核 | Ubuntu 24.04.5 LTS，6.8.0-149-generic |
| NVIDIA 驱动 | 595.99.02 仍正常；`dkms status` 显示已为 6.8.0-149 重建；GSP 固件在 `/lib/firmware/nvidia/595.99.02/` |
| `nvidia-smi -L` | `GPU 0: NVIDIA GeForce RTX 3090`，24576 MiB |
| PyTorch | `2.9.0+cu128`，GPU 矩阵乘正常 |
| JAX | `0.5.3`，`jax.devices()` → `[CudaDevice(id=0)]` |
| venv | Python 3.11.18 未受影响；torch/jax/flax/transformers/lerobot/cv2/zmq/scipy 均可导入 |
| Jaka 推理 | checkpoint 加载成功，dummy 推理返回 `(50, 33)` |
| 策略服务端 | `serve_policy.py --env JAKA` 正常监听 `0.0.0.0:8000`（首次重试用 8001 排除了 TIME_WAIT 占用） |
| 训练链路 | Jaka 数据集出一个 batch：`state (8,33)`、`actions (8,50,33)` |
| 协议自测 | `jaka_tabletop_env_test.py` 5/5 PASS |

> 说明：在本仓库的受限执行环境里 `nvidia-smi` 会报
> `Failed to initialize NVML: Unknown Error`、torch 报 CUDA error 304，
> 原因是沙箱屏蔽了 CUDA ioctl（`/dev/nvidia*` 虽可见但不可用）。
> 在普通终端（沙箱外）上述检查全部通过，**不是驱动或系统升级的问题**。

## 接下来训练 / 推理

**训练不限定在本机**：正式训练一般在训练机上跑，命令见
[03-training.md](03-training.md) 和 [10-jaka-end-to-end.md](10-jaka-end-to-end.md)。
与训练机原命令相比，**只有 `--batch-size` 可按显存调整**；默认 `bfloat16` 精度下
AdamW 状态也是 bf16（16-bit Adam），不要加 `--use-8bit-adam`。

本机的实机 checkpoint、服务启动和客户端自检命令见 [11 本机部署](11-local-deploy.md)。

> 显存提醒：如果确实要在本机 3090（24 GB）上微调，先只调 `--batch-size`（1～4 起）；
> 官方给出的全量微调参考显存是 >70 GB（A100/H100）。

## 常见问题：`cv2.imshow` 报 “The function is not implemented”

**现象**：`main.py` 跑到 `cv2.imshow(...)` 抛
`error: (-2:Unspecified error) The function is not implemented. Rebuild the library with ... GTK+ 2.x ...`。

**原因**：venv 里同时装了 `opencv-python` 和 `opencv-python-headless`（lerobot 依赖 headless），
`cv2` 最终加载的是 **headless 构建**（`cv2.getBuildInformation()` 里 `GUI: NONE`），
headless 包不含 `highgui` 窗口实现。

**两种处理**：

```bash
# 方案 A（推荐，本地桌面要看实时画面）：重装带 GUI 的 opencv-python
cd src/openpi4OpenHLM
UV_CACHE_DIR=$PWD/../../.hf/uv-cache ~/.local/bin/uv pip install \
  --force-reinstall --no-deps "opencv-python==4.11.0.86"
.venv/bin/python -c "import cv2; print([l for l in cv2.getBuildInformation().splitlines() if l.strip().startswith('GUI')])"
# 期望 GUI: QT5
```

```bash
# 方案 B（SSH / 无显示 / 不需要画面）：启动时关掉可视化
python main.py ... --no-opencv-visualize
```

> 注意：以后只要再跑 `uv sync`，lerobot 依赖可能把 headless 包装回来，`cv2` 又变回
> `GUI: NONE`，重跑方案 A 那条命令即可。

## 维护提示

- 每次重装 / 升级 `transformers` 后都要重新执行一次
  `cp -r src/openpi/models_pytorch/transformers_replace/* .venv/.../transformers/`，
  uv 默认 hardlink 模式会污染全局缓存，必要时 `uv cache clean transformers`。
- 新增数据后记得重算 norm stats，否则训练会报缺 `norm_stats.json`。
