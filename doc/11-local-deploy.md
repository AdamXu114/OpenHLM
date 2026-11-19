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

- `EnvMode.JAKA` 的值就是 `jaka_tabletop_pick`，两种写法等价。
- **不传 `policy:checkpoint` 会退回 `DEFAULT_CHECKPOINT[EnvMode.JAKA]`**，
  那是旧路径 `checkpoints/jaka_tabletop_pick/jaka_tabletop_pick_v1/30000`，注意别加载错。
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

前提：机器人侧 `state`(:28711)、`camera`(:28712) 在发布，下游 `RealtimeMotionBufferVla` 在监听 `:28701`。

```bash
cd ~/workspace/OpenHLM/src/GR00T-WholeBodyControl4OpenHLM/openpi-eval
export PYTHONPATH=~/workspace/OpenHLM/src/GR00T-WholeBodyControl4OpenHLM

~/workspace/OpenHLM/src/openpi4OpenHLM/.venv/bin/python main.py \
  --env jaka_tabletop_pick \
  --remote-host 127.0.0.1 --remote-port 8000 \
  --jaka-motion-zmq-port 28701 \
  --jaka-state-address <机器人IP> --jaka-state-port 28711 \
  --jaka-head-image-address <机器人IP> --jaka-head-image-port 28712 \
  --instruction "pick up the box" \
  --max-steps 100000
```

- 客户端会阻塞等一个 `s` 再开始，SSH/无显示时加 `--no-opencv-visualize`。
- 指令默认就是 `pick up the box`，`--instruction` 不传也一样；**不要**改成别的说法，
  否则和训练时的 language conditioning 不一致。
- 推理端动作是 33 维（27 关节 + root 3 + anchor_lin_vel 3），`JakaOutputs` 已截断，
  `JakaTabletopEnv.step()` 负责转成 SIM 顺序并发布协议 v1 帧。

### B. 本地自检（无机器人）

```bash
# 真策略服务 + 随机观测（验证 观测→推理→发布 全链路，不发到真机也不会报错）
... python main.py --env jaka_tabletop_pick \
  --remote-host 127.0.0.1 --remote-port 8000 \
  --mock --max-steps 50 --no-opencv-visualize

# 完全不连服务：假策略，只验证发布链路
... python main.py --env jaka_tabletop_pick --mock --use-fake-policy \
  --max-steps 50 --no-opencv-visualize
```

## 6. 本机性能参考（RTX 3090 实测）

| 项 | 数值 |
| --- | --- |
| 模型显存 | 加载后约 7.5 GB，峰值 **7.11 GiB** |
| 首次推理 | ~129 s（Triton autotune + 首次 kernel） |
| 稳态推理 | ~140 ms / 50 步 chunk |
| 控制频率/开环 | 30 Hz / 25 步，每次推理预算约 0.83 s，余量充足 |

## 7. 已知问题（部署前必读）

1. **anchor 帧残差**：新数据集的 anchor body 是 `base_link`（旧数据是 `waist_yaw_Link`），
   当前部署端把发布的位姿按 `waist_yaw_Link` 解释，靠初始 z=0.83 抵消了 0.22385 m 的
   高度差，但还残留 **位置 ≤2 cm、朝向 ≤7°**（正好等于 `waist_yaw_joint` 角）。
   `env.step` 的速度积分本身没有错（forward/nominal 标签与其严格互逆），
   需要改的是「发布前补一步 base→waist FK」，方案已讨论过，尚未实施。
2. **prompt**：训练/部署必须同为 `pick up the box`。
3. **`PYTHONPATH`**：`main.py` 依赖 `src/GR00T-WholeBodyControl4OpenHLM` 在 `PYTHONPATH` 里。
4. **opencv**：无 X display 时 `cv2.imshow` 会 abort，加 `--no-opencv-visualize`。
5. **别在本机训练**：3090 24 GB 放不下 3.6B 模型 + 优化器状态；训练在服务器上做。

## 8. 数据集回放 / 评测（可选）

- `jaka_policy_dataset_eval.py`：策略对数据集 ground truth 的误差对比（需要起策略服务）。
- `jaka_dataset_action_publisher.py`：把数据集的 33 维动作直接下发给控制器，
  先加 `--dry-run` 验证，再真发。

两个脚本都会自动探测数据集根目录（`record_teleop/record_data/lerobot_train/jaka_pick`），
也可用 `--dataset-root` / `--episodes` 指定。

## 参考

- [08-local-machine.md](08-local-machine.md)：本机环境安装（uv / venv / 权重）
- [04-deployment.md](04-deployment.md)：链路原理、anchor 约定、端口表
- [07-jaka-deploy-tutorial.md](07-jaka-deploy-tutorial.md)：机器人侧部署教程
