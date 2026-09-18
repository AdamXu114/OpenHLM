# 02 数据

## 2.1 数据流总览

```
遥操作录制                   合并 + 重排                       训练读取
─────────────────►  data/teleop_jaka_mf/<timestamp>/level-0/
                    （每个 session 一个独立 LeRobot v2.1 数据集）
                              │
                              │  examples/jaka/merge_teleop_jaka_mf.py
                              ▼
                    data/teleop_jaka_mf/simple/JakaTabletopPickTeleop-v0/level-0
                    （训练用的 repo_id，已重排成"手臂优先"顺序）
                              │
                              │  scripts/compute_norm_stats.py
                              ▼
                    assets/jaka_tabletop_pick/.../norm_stats.json
```

**关键环境变量**（不设会报错，vendored lerobot 不认 `LEROBOT_HOME`）：

```bash
export HF_LEROBOT_HOME=/workspace/OpenHLM/data
```

## 2.2 维度约定

| | 维度 | 组成 |
| --- | --- | --- |
| 录制时 state | 30 | dof27 + rpy3 |
| 录制时 actions | **40** | 上面 30 + `anchor_lin_vel` 3 + `anchor_pos_w` 3 + `anchor_quat_w` 4 |
| 训练用 state | 30 | 重排后 |
| 训练用 actions | **33** | 重排后 30 + `base_vel` 3（anchor 尾巴被 `JakaOutputs[:33]` 丢掉） |

> 注意：state 里 **没有速度**；actions 的 33 维是 dof27 + rpy3 + base_vel3。
> `base_vel` 是当前机器人坐标系下的相邻帧速度，采集时就记好了，训练原样穿过。

### 排列顺序

**录制时（磁盘上，MuJoCo 关节序）**：

```
[0:6]   leg_left    hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
[6:12]  leg_right   同上
[12:13] waist       waist_yaw
[13:19] arm_left    shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_yaw
[19:25] arm_right   同上
[25:27] neck        neck_yaw, neck_pitch
[27:30] root        root_roll, root_pitch, yaw_vel
```

**重排后（训练用，"手臂优先"）**：

```
arm_left(6), arm_right(6), leg_left(6), leg_right(6), waist(1), neck(2), root(3)
```

> 这套顺序是**硬编码**在 `reorder()` 里的（`examples/jaka/merge_teleop_jaka_mf.py:181`）。
> 顺序同时作用于 `state` 和 `actions[:30]`；末尾 10 维 anchor 动作永远原样追加、不参与重排。
> 脚本会用 `SOURCE_STATE_NAMES` 对照 `meta/info.json` 校验，来源数据格式不对会**直接报错**而不是静默错位。

## 2.3 合并 + 重排

```bash
cd /workspace/OpenHLM/src/openpi4OpenHLM

# 1) 先 dry-run 看维度、episode 数、排列是否正确
/workspace/OpenHLM/src/openpi4OpenHLM/.venv/bin/python \
    examples/jaka/merge_teleop_jaka_mf.py --dry-run

# 2) 确认无误后真正写入（会删掉同名输出目录重建）
/workspace/OpenHLM/src/openpi4OpenHLM/.venv/bin/python \
    examples/jaka/merge_teleop_jaka_mf.py --overwrite
```

常用参数：

| 参数 | 说明 |
| --- | --- |
| `--source-dirs <dir>...` | 源 `level-0` 目录列表（默认已填好，见脚本 `DEFAULT_SOURCE_DIRS`） |
| `--output-repo-id <id>` | 输出 repo_id，**必须和 `config.py` 里 `LeRobotJakaDataConfig.repo_id` 一致** |
| `--hf-home <dir>` | 默认 `/workspace/OpenHLM/data` |
| `--max-episodes-per-session <n>` | 每个 session 只取前 n 条，调试用 |
| `--identity` | 保持录制原序不重排，用来和源数据 diff 校验 |
| `--dry-run` | 只打印不写盘 |
| `--overwrite` | 输出已存在时删除重建 |

脚本是 **pyarrow 直接改写**：`head_image_left` 列里是 PNG 字节（`struct<bytes, path>`），按字节整段拷贝，不解码；只动 `state` / `actions` 和记账列（`index` / `episode_index` / `task_index`）。

历史数据（2026-09-15 三个 session）：3 + 5 + 39 = **47 episodes / 11657 frames**。

## 2.4 重算 norm stats（**每次动过数据格式就必须重算**）<a id="recompute-norm-stats"></a>

```bash
cd /workspace/OpenHLM/src/openpi4OpenHLM

# 必须在 CPU 上跑！
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
    ./.venv/bin/python scripts/compute_norm_stats.py --config-name jaka_tabletop_pick
```

> ⚠️ **一定要加 `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=""`。**
> 否则 jax 会预分配整张卡（在 20 GB 的预约上直接 OOM）。

其他参数：`--max-frames <n>` 限制统计帧数（大数据集时用）。

**什么时候要重算**：重排顺序改了 / 数据格式改了 / 换数据集 / 增删 session —— 只要
`assets/jaka_tabletop_pick/<...>/norm_stats.json` 和当前数据对不上，就重算。

## 2.5 写合并脚本时踩过的坑（复用注意）

- `LeRobotDataset.create(root=X)`：`X` 被当作**数据集目录本身**，**不会**自动拼 `repo_id`；
  只有默认的 `HF_LEROBOT_HOME / repo_id` 形式才会拼。
- `add_frame` 的 `timestamp` 不接受普通 float —— 它按 shape `(1,)` 的 float32 ndarray 校验。
  直接**不传**即可，自动推导的 `frame_index / fps` 和这些源数据的 timestamp 逐位相同。
