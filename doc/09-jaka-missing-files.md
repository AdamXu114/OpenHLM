# 09 Jaka 相关文件完整性核对（GitHub 上传缺失清单）

> 核对时间：2026-11-07。范围：`doc/01`–`doc/08` 中所有 Jaka 段落 +
> `src/GR00T-WholeBodyControl4OpenHLM/implement_action_analysis/*.md` +
> `openpi-eval/jaka_*.py` + `examples/jaka/` + `jaka_policy.py` / `config.py`。
>
> 方法：把文档里出现的每个路径做 exists 检查；实际 `import` 部署链路；
> 实际跑 `jaka_tabletop_env_test.py`；核对 HuggingFace 上的 checkpoint 仓库。
>
> 结论先行：**部署链路的三段代码都在，且能跑通自测；缺的是权重、训练数据，
> 以及下游接收端依赖的 SIMPLE 包**。这些文件不是"被删了"，而是**从来没提交**
> （`git log --all --diff-filter=D` 只删过一个 `scripts/main.sh`）。

> **2026-11-07 范围更新**：后续数据采集走**实机采集**，部署也**不再经过 SIMPLE**。
> 因此本文第 2 节关于 SIMPLE 包的缺失只作历史记录，不再阻塞任何流程；
> 当前只需保证 **OpenHLM 仓库内的训练 + VLA 推理链路**完整。

---

## 1. 部署链路（`doc/07` 用的文件）

| 作用 | 路径 | 状态 |
| --- | --- | --- |
| 策略服务端 | `src/openpi4OpenHLM/scripts/serve_policy.py` | ✅ 在（`DEFAULT_CHECKPOINT[EnvMode.JAKA]` 已配） |
| 部署客户端 | `src/GR00T-WholeBodyControl4OpenHLM/openpi-eval/main.py` | ✅ 在，`--help` 正常，含 `--env jaka_tabletop_pick` |
| Jaka 环境 | `openpi-eval/jaka_tabletop_env.py` | ✅ 在，`import` 通过（27 关节、28701/28711/28712） |
| 协议自测 | `openpi-eval/jaka_tabletop_env_test.py` | ✅ 在，实测 **5/5 PASS** |
| 训练/推理策略变换 | `src/openpi4OpenHLM/src/openpi/policies/jaka_policy.py` | ✅ 在 |
| 训练配置 | `src/openpi4OpenHLM/src/openpi/training/config.py`（`jaka_tabletop_pick`） | ✅ 在 |
| norm stats | `src/openpi4OpenHLM/assets/jaka_tabletop_pick/.../level-0/norm_stats.json` | ✅ 在（已提交，276 行） |
| **策略权重 30000** | `src/openpi4OpenHLM/checkpoints/jaka_tabletop_pick/jaka_tabletop_pick_v1/30000/` | ✅ **2026-11-07 已从 HF 下载并解包**（不在 git，`checkpoints/` 被忽略） |
| HF 权重仓库 | https://huggingface.co/XuJinfan/jaka-tabletop-pick-v1 | ✅ 公开，单文件 `upload` 7.47 GB |
| 基座权重（训练用） | `~/.cache/openpi/.../pi05_base_pytorch` / HF `XuJinfan/pi05-base-pytorch` | ✅ 本机已有（见 [08](08-local-machine.md)） |

权重内容已核对（tar 共 9 个条目）：

```
30000/
├── model.safetensors   7,473,099,660 B   （812 个张量，fp32；action_in_proj (1024,33)）
├── metadata.pt                 3,507 B
└── assets/teleop_jaka_mf/simple/JakaTabletopPickTeleop-v0/level-0/norm_stats.json
                          7,575 B   （与仓库 assets/ 里的那份逐字节相同）
```

实测加载 + 推理通过：`JAKA_INFER_OK (50, 33)`（GPU，dummy 观测）。

VLA 推理链路另外测过：

- `scripts/serve_policy.py --env JAKA --num-steps 10`：加载 norm stats、checkpoint 成功，监听 `0.0.0.0:8000`；
- `openpi-eval/main.py --env jaka_tabletop_pick --mock --use-fake-policy`：20 步控制循环正常跑完并复位。
下载命令：

```bash
cd src/openpi4OpenHLM
mkdir -p checkpoints/jaka_tabletop_pick/jaka_tabletop_pick_v1
huggingface-cli download XuJinfan/jaka-tabletop-pick-v1 upload --local-dir /tmp/jaka_ckpt
tar -xf /tmp/jaka_ckpt/upload -C checkpoints/jaka_tabletop_pick/jaka_tabletop_pick_v1/
# tar 顶层就是 30000/，含 model.safetensors + assets/
```

> ✅ `doc/07` 已同步修正：HF 上只有一个名为 `upload` 的文件（7,473,121,280 B，tar 格式，
> 顶层 `30000/`）。`model.safetensors` 本身正好是 7,473,099,660 B，与原文预期一致。

---

## 2. 下游接收端：`RealtimeMotionBufferVla` 依赖缺失（重要）

> **2026-11-07 更新**：完整的 SIMPLE 项目已在本机 `../SIMPLE-jaka`
> （`/home/xjf/workspace/SIMPLE-jaka`）。下面列的"缺失"指**本仓库没上传**，
> 不代表拿不到；端口也以该仓库为准（见 §5）。

`implement_action_analysis/` 里只上传了 **3 个 py 文件**，而 `motion_buffer.py`
自己 import 的模块大半不在仓库里，所以它**根本 import 不起来**：

```
ModuleNotFoundError: No module named 'loguru'      ← 第三方包
ModuleNotFoundError: No module named 'simple.jaka_rl.math'  ← 仓库内缺失
```

实测（`PYTHONPATH=src python -m simple.jaka_rl.action_trunk_reconstruct`）直接失败。
这也是为什么 `jaka_tabletop_env_test.py` 要用 `ast` 从源码里抽取 `_decode_binary_v1`
来对拍——它绕开了 import。

| 被 import/引用的模块 | 状态 | 谁需要 |
| --- | --- | --- |
| `simple/jaka_rl/config.py` | ❌ 本仓库 / ✅ SIMPLE | `motion_buffer.py`（`ANCHOR_BODY`、`TOGGLE_DATA_COLLECTION_KEY`） |
| `simple/jaka_rl/math.py` | ❌ 本仓库 / ✅ SIMPLE | `motion_buffer.py`、`action_trunk_reconstruct.py` |
| `simple/jaka_rl/motion.py` | ❌ 本仓库 / ✅ SIMPLE | `motion_buffer.py`、`state_processor.py`（`MotionData`） |
| `simple/jaka_rl/npz_motion.py` | ❌ 本仓库 / ✅ SIMPLE | `state_processor.py`（`NpzMotionDataset`） |
| `simple/jaka_rl/observations/jaka_mf.py` | ❌ 本仓库 / ✅ SIMPLE | 620 维观测消费方 |
| `simple/jaka_rl/motion_buffer.py` | ✅ | 已上传（1465 行） |
| `simple/jaka_rl/state_processor.py` | ✅ | 已上传（323 行） |
| `simple/jaka_rl/__init__.py` | ✅ | 已上传，但 import 就炸（缺上面几项） |
| `simple/jaka_rl/action_trunk_reconstruct.py` | ⚠️ 位置不对 | 文件在 `implement_action_analysis/` 根目录；文档按包路径引用，`python -m simple.jaka_rl.action_trunk_reconstruct` 找不到 |

**影响**：仓库里的 `motion_buffer.py` 只能当"参考实现/语义基准"阅读，
**不能直接当运行时用**。真正在跑的接收端来自 SIMPLE 项目
（`implementation_analysis.md` 里的 `src/SIMPLE-jaka-main/`），本仓库没有上传；
现在本地副本在 `../SIMPLE-jaka/src/simple/`。

**版本漂移**：本仓库的 vendored 拷贝比 `../SIMPLE-jaka` 的版本旧（`motion_buffer.py`
1465 vs **1665** 行）。SIMPLE 新版多了：播放时钟按实测墙钟间隔推进（修 50 Hz 软实时
循环下的延迟漂移）、`diagnostics()` 与 `enable_diagnostics` 开关、
`RealtimeMotionBufferVla.clear()` 重置播放时钟。`state_processor.py` 只是多了一段
`enable_diagnostics=True`。`__init__.py` 无差异。

其余缺失的 SIMPLE 文件（文档引用、代码没有）：

- `simple/cli/record_jaka_zmq.py`（录制主程序，`JAKA_RECORDED_DATA_REFERENCE.md` 的主角）
- `simple/cli/teleop_jaka_mf.py`（`jaka_policy.py` 的 docstring 指向它）
- `simple/datasets/jaka_lerobot.py`（落盘格式、`OpenHLMRootVel`）
- `simple/interfaces/jaka_zmq_pub.py`（:28711/:28712 的发布端）
- `simple/robots/jaka.py`（`waist_imu` 观测）
- `simple/tasks/jaka_vla_cameras.py`（相机分辨率真源）

---

## 3. 训练数据（全部不在仓库）

`data/*` 被 `.gitignore` 排除，以下都没有：

| 内容 | 路径 | 状态 |
| --- | --- | --- |
| 原始录制（3 个 session，47 episodes / 11657 frames） | `data/teleop_jaka_mf/20260915-142439/level-0` 等 | ➖ 不需要（后续实机采集；合并前的原始 session 未公开） |
| 合并后的训练集 | `data/teleop_jaka_mf/simple/JakaTabletopPickTeleop-v0/level-0` | ✅ **2026-11-07 已从 HF `XuJinfan/openhlm-data` 下载**（226 MB，47 eps / 11657 frames / 30 fps，state 30 / actions 40） |
| 合并脚本 | `examples/jaka/merge_teleop_jaka_mf.py` | ✅（但 `DEFAULT_SOURCE_DIRS` / `DEFAULT_HF_HOME` 写死 `/workspace/...`） |
| 训练超参 | `config.py::jaka_tabletop_pick` | ✅（`pytorch_weight_path` 写死 `/workspace/openpi-assets/...`） |

合并数据集已在本地，**重算 norm stats 已实测可行**（CPU 约 60 s，1457 个 batch）。
但发现一个**语义不一致**，重训前必须处理：

| 项 | 实测 |
| --- | --- |
| 当前 `config.py::jaka_tabletop_pick` | `use_delta_joint_actions=False`（绝对动作用于训练） |
| 仓库/checkpoint 里的 `norm_stats.json` | **delta 动作统计**：`mean(actions - state)` 与提交值逐维吻合（误差 <1.3e-5），绝对动作均值则差到 1.34 |
| 后果 | 训练时 27 个关节维度的动作会被按"delta 的均值/方差"归一化，中心偏了一个 state 均值。归一化可逆，所以推理仍自洽，但训练收敛/稳定性会受影响 |

**处理建议（重训时二选一）**：

1. 保持 `use_delta_joint_actions=False`，重新跑
   `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" ./.venv/bin/python scripts/compute_norm_stats.py --config-name jaka_tabletop_pick`
   覆盖成绝对动作统计（**本机已验证可跑**）；
2. 或者把 config 改成 `use_delta_joint_actions=True`，让训练/推理都走 `DeltaActions` + `AbsoluteActions`，
   与现有统计匹配。

> ⚠️ 只做部署时**不要重算**：已发布的 30000 checkpoint 的 `assets/` 里带的就是这份 delta 统计，
> 服务端加载 checkpoint 内的 assets，训练/推理统计一致。重算会改变推理行为。

---

## 4. 文档引用但仓库里没有的资产 / 配置 / 脚本

来自 `JAKA_RECORDED_DATA_REFERENCE.md`、`implementation_analysis.md`：

| 类别 | 路径 | 说明 |
| --- | --- | --- |
| 策略配置 | `data/jaka_mf/teleop_jaka_mf.yaml` | `motion_backend: zmq_vla` 的宿主配置 |
| 策略配置 | `data/jaka_mf/latest56k_pico_dr.yaml` | `anchor_body_name` / `future_steps` / 顺序表 |
| 参考运动 | `data/jaka_mf/motion_mf.npz`（及 `motion_mf` 系列） | 对账真值 |
| 机器人 MJCF | `data/robots/jaka/Khan_mini_simplified_new_bigfeet.xml` | free joint / waist 链定义 |
| 分析文档 | `implement_action_analysis/JAKA_QUICKSTART.md` | 被引用，未上传 |
| 分析文档 | `implement_action_analysis/TRAINING_DATA_FORMAT.md` | 被引用，未上传 |
| 脚本 | `scripts/replay_jaka_action_mujoco.py` | 文档自己标注**已失效**（import 不存在的函数） |
| 脚本 | `scripts/verify_jaka_action_recon_error.py` | action 与真值对账脚本 |
| C++ 参考实现 | `src/KpiDeployReal/include/RealtimeMotionBufferVla.hpp`、`src/RealtimeMotionBuffer_vla.cpp`、`src/FSMMimicJakaMiniZmq.cpp` | `implementation_analysis.md` 的"语义基准" |
| 上游项目 | `src/SIMPLE-jaka-main/**` | 接收端真正运行的代码 |
| 外部仓库 | sim2real-jaka（`pico_retarget_pub.py`、`xrobot_to_jaka.json`、`sim2real/config/robots/jaka.py`） | 只在文档里以绝对路径 `/home/xu/...` 出现 |

另外两处文档自己已声明"任何 checkout 都不存在"，可忽略：
`doc/FSMMimicJakaMiniZmq.cpp`、`doc/RealtimeMotionBuffer.cpp`。

`gear_sonic_deploy/.../streamed_motion_merger.hpp` 实际存在，路径是
`gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/input_interface/streamed_motion_merger.hpp`，文档用 `...` 省略了。

---

## 5. 文档与实际不一致的地方（建议修）

| # | 文档 | 问题 |
| --- | --- | --- |
| 1 | `doc/07` §0 | HF 上文件叫 `upload`（7,473,121,280 B），不是 `30000.tar`（7,473,099,660 B） |
| 2 | `doc/06` 速查表 | ~~写 state/head 端口 `28702/28703`~~ → **已修为 28711/28712**（以 `../SIMPLE-jaka` 的 `jaka_zmq_pub.py` 与 motion buffer 为准） |
| 3 | `JAKA_RECORDED_DATA_REFERENCE.md` §9 | 链接 `action_trunk_reconstruct.py` 指向 `src/simple/jaka_rl/`，实际在 `implement_action_analysis/` 根目录 |
| 4 | 同文 §7.2 | 自检命令 `PYTHONPATH=src python -m simple.jaka_rl.action_trunk_reconstruct` 跑不通（文件不在包里 + 缺 `simple.jaka_rl.math`） |
| 5 | `implementation_analysis.md` §4.1/§8 | 声称修改了 `src/SIMPLE-jaka-main/...`，但该目录不在本仓库；读者无法核对"旧类零改动" |

---

## 6. 最小可跑清单

**只做部署（不改代码）**：

1. HF 下载 `upload` → 解包出 `30000/`（含 `model.safetensors` + `assets/`）
2. `serve_policy.py --env JAKA --num-steps 10`
3. `main.py --env jaka_tabletop_pick ...`（`PYTHONPATH` 指向 GR00T 根）
4. 下游接收端：需要 **SIMPLE-jaka-main**（外部项目），本仓库的 `motion_buffer.py` 不可直接运行

**要训练/复现**：还需要 §3 的原始录制 + 合并数据集 + `pi05_base_pytorch` 基座权重
（本机已按 [08-local-machine.md](08-local-machine.md) 下载转换到
`~/.cache/openpi/openpi-assets/checkpoints/pi05_base_pytorch`）。

---

## 7. 结论：OpenHLM 内还缺什么

按新范围（实机采集 + 不经过 SIMPLE），OpenHLM 仓库内**训练与 VLA 推理都不缺代码**，
只剩两件"配置/一致性"层面的事：

| # | 事项 | 状态 |
| --- | --- | --- |
| 1 | 训练集 `teleop_jaka_mf/simple/JakaTabletopPickTeleop-v0/level-0`（47 eps） | ✅ 已下载到 `data/`（226 MB） |
| 2 | norm stats 与 `use_delta_joint_actions=False` 语义不一致 | ⚠️ **重训前重算或改配置**，见 §3 |
| 3 | `config.py` 的 `pytorch_weight_path` 写死 `/workspace/...` | ℹ️ 训练机路径本来就对；只有本机临时训练才需要 `--pytorch-weight-path` 覆盖 |
| 4 | 部署/推理：server + client + checkpoint + 资产 | ✅ 全部就位并实测 |
| 5 | 机器人侧动作接收方（协议 v1，28701） | 不在本仓库；既不走 SIMPLE，需要机器人侧自己实现/集成 |

历史遗留（已不影响当前流程）：

- `../SIMPLE-jaka` 相关的两个快照文件已同步（`motion_buffer.py` 1665 行 + `implementation_analysis.md` 592 行）；
- `data/jaka_mf/*.yaml`、`data/robots/jaka/*.xml`、`JAKA_QUICKSTART.md`、`TRAINING_DATA_FORMAT.md`
  在 SIMPLE 仓库里，OpenHLM 本身不需要它们来训练/推理；
- `scripts/verify_jaka_action_recon_error.py`、`src/KpiDeployReal/**` 属于历史分析材料，未上传。

若要长期维护，仍建议把原始 3 个 session 的录制数据归档（重算/对比数据分布时会用到），
但按当前"实机采集"的路线不是必须。
