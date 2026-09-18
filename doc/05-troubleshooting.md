# 05 排错

按"症状 → 原因 → 处理"组织。每条都是**实际踩过的**。

---

## 训练/容器

### 容器突然挂了，exit 137 <a id="exit-137"></a>

**先别怀疑 OOM。** 在 hpc 上这几乎总是 **GPU 预约问题**。

**怎么确认**：

```bash
docker inspect <container> --format '{{.State.OOMKilled}}'   # false 就更说明不是内存
nvidia-smi                                                    # 看卡上还有没有别人
/usr/local/bin/hpc                                            # 看你的预约还在不在
```

日志里会有：`enforcer: stopping unauthorized container <id> (user=... gpu=... mem=...)`。

**三个常见触发点**：

1. 压根没预约。
2. 预约**过期**了（只有 15 分钟宽限）。
3. **预约被低利用率回收**：连续 60 分钟 GPU 使用率 < 10% 会被回收 —— **包括完全没用**。
   最坑的场景：预约 → 去准备数据 → 一小时后回来训练 → 一启动就被杀。

**处理**：重新 `hpc` 预约，并**确认 `CUDA_VISIBLE_DEVICES` 指向预约的卡**。

> 实测数据：cgroup 内存限制全是 `max`，30 GB 纯内存测试能活；而 20 GB 预约跑出 34 GB 峰值也没被杀。
> 所以“超预约”不是问题，“没预约”才是。

### 训练日志里看不到 loss / 没有日志文件

**默认就不写日志文件** —— `init_logging()` 只挂 `StreamHandler`（stdout）。
`metadata.pt` 里也只有 `global_step` / config / 时间戳，没有 loss 历史。

**处理**：起训练时就 `2>&1 | tee logs/xxx.log`。已经跑起来了又忘了？
只能 `tmux capture-pane` 捞当前屏幕，或者靠 wandb。

### tmux 里按了 Ctrl-d / exit，训练没了

`exit` 关掉的是 pane 的 shell，前台训练进程跟着死。

**脱离要按 `Ctrl-b` 然后 `d`**。已经死了就只能重跑（`--overwrite`）。

### `--resume` 报 `FileNotFoundError: .../wandb_id.txt`

`scripts/train_pytorch.py`（约 85 行）在 resume 时无条件读 `wandb_id.txt`，
但这个文件只有原 run 开了 wandb 才存在。

**处理**：目前不用 `--resume`，直接 `--overwrite` 重跑。（真正的修法是加 4 行 fallback，尚未应用。）

### 20 GB 预约跑不起来 / 显存爆

全精度 AdamW 要 35–40 GB。

**处理**：加 `--use-8bit-adam`（bitsandbytes 0.50.2 已装）。

### 训练很慢 / 显存分配异常

`XLA_PYTHON_CLIENT_PREALLOCATE=false` 没设，jax 预分配了整张卡。

---

## 数据

### `compute_norm_stats.py` OOM

jax 默认预分配整张 GPU，在 20 GB 预约上直接 OOM。

**处理**：**强制走 CPU**：

```bash
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" ./.venv/bin/python scripts/compute_norm_stats.py --config-name jaka_tabletop_pick
```

### 报错找不到 `LEROBOT_HOME` / 数据集加载失败

vendored lerobot **只认 `HF_LEROBOT_HOME`**，不认 `LEROBOT_HOME`。

```bash
export HF_LEROBOT_HOME=/workspace/OpenHLM/data
```

### 换了数据但训练效果不对 / norm stats 不匹配

**每次改数据格式（重排、换数据集、增删 session）都必须重算 norm stats**，见 [02-data.md](02-data.md#recompute-norm-stats)。

### 写合并脚本时的两个坑

- `LeRobotDataset.create(root=X)`：`X` 是**数据集目录本身**，不会自动拼 `repo_id`；
  只有 `HF_LEROBOT_HOME / repo_id` 的默认形式才拼。
- `add_frame` 的 `timestamp` 不接受普通 float（它按 shape `(1,)` float32 ndarray 校验）。
  **直接不传**，自动推导的 `frame_index / fps` 和源数据逐位相同。

---

## 部署/推理

### `main.py` 一启动就 import 失败

`sonic_g1_env` 要 import `gear_sonic`，openpi 的 venv 里没有。

**处理**：加 `-e PYTHONPATH=/workspace/OpenHLM/src/GR00T-WholeBodyControl4OpenHLM`。

### cv2 在 `imshow` 上 abort / 段错误

容器**没有 X display**，而 `opencv_visualize` **默认是 True**。

**处理**：显式加 `--no-opencv-visualize`。

### 客户端卡住不动

主循环在 `input()` 等 's'。

**处理**：喂 stdin，如 `<<< $'s\nn\n'`。

### 下游轨迹"静默冻结"

`frame_index` 被重置了。接收方的去重基线在它自己 `clear()` 之后依然存在，
所以**客户端重启会让参考轨迹静默冻结**（不报错）。

**处理**：`JakaTabletopEnv.reset()` 里**故意不复位** `_frame_index` / `_accumulated_yaw` / `_body_pos_w`。
改这段代码时不要"顺手"加上复位。

### 改了发布协议后行为异常

**处理**：重跑 `jaka_tabletop_env_test.py` 对拍权威解码器（见 [04-deployment.md](04-deployment.md#44-协议改动的验证)）。

### anchor 位姿对不上

检查三点：`anchor_lin_vel` 是否**先做了 anchor quat 旋转**再积分（它是 body 系速度，不是世界速度）；
发布帧的四元数是否用的是**该帧 `yaw_vel` 生效前**的累加 yaw；
`--jaka-initial-anchor-pos` 的 **z 是否合理**（站立约 0.83）。

---

## 环境

### 装包时 `pip: command not found`

openpi 的 `.venv` 是 **uv venv，没有 pip**。

```bash
/root/.local/bin/uv pip install --python /workspace/OpenHLM/src/openpi4OpenHLM/.venv/bin/python <package>
```

### 权重加载失败 / 路径找不到

safetensors **不展开 `~`**。config 里原来写的 `~/.cache/...` 是错的，必须用绝对路径：

```
/workspace/openpi-assets/checkpoints/pi05_base_pytorch
```

### 日志文件名的时间看着不对

容器时区是 **UTC**，主机是 CST（UTC+8），差 8 小时。不是 bug。

### 想看 wandb 曲线但 offline run 没界面

容器有网络到 api.wandb.ai 但**没有凭据**，所以自动落 offline。offline run **本地没有查看器**。

**处理**：配好代理（见 [01-environment.md](01-environment.md#proxy)）后
`wandb login` + `wandb sync`。或者干脆解析文本日志，见 [03-training.md](03-training.md#34-日志)。
