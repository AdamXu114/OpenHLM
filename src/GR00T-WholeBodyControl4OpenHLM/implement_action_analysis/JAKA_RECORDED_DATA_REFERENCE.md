# Jaka 录制数据参考（`record_jaka_zmq` 落盘格式）

本文只描述 `simple.cli.record_jaka_zmq` 落盘的数据：每一维**怎么算出来的、在哪个参考系里**，
以及**哪些量能从这份数据重建、哪些不能**（§7.1）。
训练侧的格式（π0.5 / OpenHLM 的 34 维布局等）见 `TRAINING_DATA_FORMAT.md`；录制操作流程见
`JAKA_QUICKSTART.md` §3。

---

## 0. 速览

- 一个 episode = **一帧一行**；录制循环每拍从三路 ZMQ 各取**最新一帧**拼成一行。
- 每行三个 key：`head_image_left`（原分辨率 RGB 图）、`state`(30)、`actions`(40)。
- 单位：关节 **rad**、位置 **m**、关节/角速度 **rad/s**、线速度 **m/s**。
- 四元数**一律 `wxyz`**（标量在前，MuJoCo 约定）。
- `timestamp` 列是 LeRobot 合成的 `frame_index / fps`，**不是**墙钟时间（见 §5）。
- ⚠️ **机器人 root 存的是 `base_link`，而策略观测用的是 `waist_yaw_Link`**（`waist_imu` site）——
  两者差一个腰关节角（见 §6.3）。这是本数据集最容易踩的参考系坑。

---

## 1. 数据从哪来

录制进程不参与控制，只订阅三路 ZMQ，全部 **latest-only**（每拍丢弃积压、只留最新一帧）：

| 流 | 端口 | 发布方 | 原生频率 | 落到数据集的字段 |
|---|---|---|---|---|
| `state` | `28711` | `JakaTeleopZmqPublisher._publish_state`（[jaka_zmq_pub.py:162](src/simple/interfaces/jaka_zmq_pub.py#L162)） | 每个控制拍一次（`rl_rate` = 50 Hz） | `state`(30) |
| `camera` | `28712` | 同一 publisher 的**独立相机线程**（[jaka_zmq_pub.py:231](src/simple/interfaces/jaka_zmq_pub.py#L231)） | `camera_hz` = 30 Hz | `head_image_left` |
| `action`（参考运动） | `28701` | `pico_retarget_pub`（pico hub） | hub 的 `--publish_hz` | `actions`(40) |

要点：

- **三路不是同一采样时刻**，只是"同一墙钟瞬间各自的最新值"。相机线程按自己的 `time.monotonic()`
  节拍发送，与 50 Hz 控制拍网格无关。
- `action` 是**参考运动**（冻结的 MF 策略要跟踪的目标），**不是策略输出**。录制端**直连**
  :28701（`LatestMotionClient`，[record_jaka_zmq.py:193](src/simple/cli/record_jaka_zmq.py#L193)），
  不经 `RealtimeMotionBuffer`、不做插值、不应用 `_delay_ns` 回看 —— 对齐的是"此刻"。
- 录制循环以 `--frequency`（默认 30）运行，每拍取完三路就 `sleep` 到下一拍，
  所以**真实帧间隔 ≥ 1/30 且有抖动**（见 §5）。

---

## 2. 落盘结构

```
<save-dir>/<run-id>/level-<dr_level>/
├── meta/{info.json, episodes.jsonl, tasks.jsonl, episodes_stats.jsonl}
├── data/chunk-XXX/episode_XXXXXX.parquet      # 唯一权威数据（图像 PNG 字节内嵌其中）
└── videos/chunk-XXX/head_image_left/episode_XXXXXX.mp4   # 仅供浏览的副本
```

- `run-id` = `--run-name`，未指定则时间戳 `20260915-134512`；**撞名自动加 `-2`/`-3` 后缀**，
  已有数据集永不被覆盖或复用（[record_jaka_zmq.py:335](src/simple/cli/record_jaka_zmq.py#L335)）。
- `level-<dr_level>` 来自 `--dr-level`。
- **数据集在第一帧相机到达时才创建**，并用该帧的实际 HWC 作为 `head_image_left` 的声明 shape
  （[record_jaka_zmq.py:498](src/simple/cli/record_jaka_zmq.py#L498)）。分辨率的真源是
  sim 的相机 cfg（`tasks/jaka_vla_cameras.py`），当前为 **224×224**。中途 shape 变化会被跳过并提示重启。
- `LeRobotDataset.create(..., use_videos=False)`：图像以 **PNG 字节内嵌在 parquet**
  （LeRobot 的 `embed_images`）。`videos/*.mp4` 是保存完 episode 后另外编码的浏览副本，
  随后**删掉冗余的 `images/` 逐帧 PNG 目录**。**读数据只读 parquet 即可**。
- 每帧的 `task` 字符串 = `--desc`，落在 `meta/tasks.jsonl`，按 `task_index` 关联。

parquet 的列（一次 `save_episode()` = 一个文件的全部行，行序即时间序）：

| 列 | 类型 | 说明 |
|---|---|---|
| `head_image_left` | struct（含内嵌 PNG） | RGB，HWC uint8，原分辨率 |
| `state` | fixed_size_list&lt;float32&gt;[30] | §3 |
| `actions` | fixed_size_list&lt;float32&gt;[40] | §4 |
| `timestamp` | float | LeRobot 合成：`frame_index / fps`（**非墙钟**） |
| `frame_index` / `episode_index` / `index` / `task_index` | int | LeRobot 元数据 |

> 工程备注：`init_openhlm_exporter` 会打两个 monkeypatch —— 离线版 `get_safe_version`，
> 以及容忍 LeRobot「`*.mp4` 计数」断言（因为我们 `use_videos=False` 却又保留了浏览用 mp4）。
> 后者**只吞那一种断言**，parquet 计数不符仍会抛。见
> [jaka_lerobot.py:156-236](src/simple/datasets/jaka_lerobot.py#L156-L236)。

---

## 3. `state` (30) —— 机器人自身状态

```
[0:27]  27 个关节位置（实际值，非目标值），MJCF(JAKA) 顺序，rad
[27]    root_roll      ┐
[28]    root_pitch     ├ 由 base_link 世界四元数算出
[29]    yaw_vel        ┘  = 绕世界 z 的角速率，rad/s
```

**来源**：`state` 流里的 `joint_pos`（27）与 `body_quat_w[0]`。

- `joint_pos` = `robot.get_robot_qpos()`，按 `JAKA_JOINT_NAMES`（MuJoCo 顺序）排列
  （[jaka_zmq_pub.py:165](src/simple/interfaces/jaka_zmq_pub.py#L165)）。是**真实关节角**，
  不是 PD 目标。顺序即 `simple.jaka_rl.config.JOINT_NAMES`，也即 `info.json` 里 `state.names[0:27]`。
- `body_quat_w` = `mjData.xquat[body_ids]`，`body_ids` 按 `JAKA_BODY_NAMES`
  （[jaka_lerobot.py:26-36](src/simple/datasets/jaka_lerobot.py#L26)）解析 —— ⚠️ 这个列表是
  **npz/policy 顺序**（`waist_yaw_Link` 在 **3**），**不是** MJCF 顺序（那种顺序下它在 13）。
  `base_link` 在两种顺序里都是 0，所以 **`body_quat_w[0]` 就是 `base_link`**
  （[jaka_zmq_pub.py:168-170](src/simple/interfaces/jaka_zmq_pub.py#L168)）。见 §6.4。
- `base_link` 挂着 free joint（`base_joint` type=`free`，[MJCF:59](data/robots/jaka/Khan_mini_simplified_new_bigfeet.xml#L59)），
  所以这个四元数 = **机器人基座在 MuJoCo 世界系里的姿态**。

**计算方式**（`OpenHLMRootVel`，[jaka_lerobot.py:407-444](src/simple/datasets/jaka_lerobot.py#L407)）：

```python
w, x, y, z = quat_wxyz
roll, pitch, raw_yaw = Rotation.from_quat([x, y, z, w]).as_euler("xyz")
dyaw = (raw_yaw - prev_raw_yaw + pi) % (2*pi) - pi      # 去掉 ±π 跳变
yaw_vel = dyaw / dt                                     # dt = 本拍与上一拍的墙钟差
```

- `as_euler("xyz")` 用的是 **scipy 外旋 xyz**（= 内旋 zyx），等价于
  `R = Rz(yaw)·Ry(pitch)·Rx(roll)`。它有精确逆变换 `quat_from_rpy`
  （[action_trunk_reconstruct.py:47](src/simple/jaka_rl/action_trunk_reconstruct.py#L47)），
  两者往返误差 ~1e-8（已实测）。
- 该 `yaw` 分量与 `simple.jaka_rl.math.yaw_from_quat` **完全相等**（实测差 0），
  即与策略侧 `yaw_quat()` 的"yaw"是同一个量。
- **`dt` 用的是录制循环的墙钟 `time.time_ns()`，不是 state 流自己的 `publish_t_ns`**（见 §5）。
- **第一帧 `yaw_vel` 恒为 0.0**：`reset()` 在每段 episode 开始时调用（[record_jaka_zmq.py:469](src/simple/cli/record_jaka_zmq.py#L469)），
  首帧只记 `prev`、不产生速率。
- ⚠️ `OpenHLMRootVel.__init__` 收 `yaw_vel_ema_alpha=0.1` 但**从未使用** —— 存的名字是
  `_yaw_vel_ema`，实际写进去的是**原始速率**，没有 EMA、没有 clip。
  （`action_trunk_reconstruct.py` 的 docstring 说 "EMA'd/clipped"，那是过时描述。）
- ⚠️ 因此 `yaw_vel` 可以是**尖峰**：实测某 episode 最大 **19.7 rad/s**（某拍 `dt` 特别小所致）。
- **绝对 yaw 不存**，且**无法从本数据集恢复**（只有速率）。要绝对朝向只能积 `yaw_vel`（会漂移），
  或回到仿真重放。

---

## 4. `actions` (40) —— pico 参考运动（冻结策略的跟踪目标）

```
[0:27]   27 个参考关节位置，MJCF(JAKA) 顺序，rad  ← 与 state 的关节顺序相同
[27]     root_roll    ┐
[28]     root_pitch   ├ 由 anchor（waist_yaw_Link）世界四元数算出
[29]     yaw_vel      ┘
[30:33]  anchor 线速度 xyz，**anchor 机体系**，m/s，已 clip 到 ±2.0
[33:36]  anchor_pos_w xyz，**原始世界系**（DEBUG）
[36:40]  anchor_quat_w wxyz，**原始世界系**（DEBUG）
```

**anchor 是谁**：`motion.anchor_body_name`，默认 **`waist_yaw_Link`**
（[latest56k_pico_dr.yaml:263](data/jaka_mf/latest56k_pico_dr.yaml#L263)）—— 即策略用来构建
`root_pos_diff_b` / `anchor_ori` 的那个参考体。

**全部 40 维取自同一帧**，即该拍从 :28701 收到的最新 pico payload。

```python
# [0:27]：关节重排到 JAKA_JOINT_NAMES 顺序（按名字匹配，不靠位置）
ref_joints = latest.joint_pos[[mj_names.index(n) for n in JAKA_JOINT_NAMES]]

# [27:30]：与 state 同款 OpenHLMRootVel，但喂的是 anchor 四元数
action_root_calc(ref_anchor_quat, t_ns)          # 与 state 用同一个 t_ns

# [30:33]：anchor 机体系线速度
vel_body = quat_rotate_inverse_numpy(ref_anchor_quat, (pos_now - pos_prev))
lin_vel  = clip(vel_body / dt, -2.0, +2.0)       # dt 来自 pico 流自己的 publish_t_ns
```

（[jaka_lerobot.py:530-590](src/simple/datasets/jaka_lerobot.py#L530)，调用点
[record_jaka_zmq.py:522](src/simple/cli/record_jaka_zmq.py#L522)）

关键细节：

- `[0:27]` 是**重定向 IK 解出来的参考关节角**（GMR 在只有机器人的模型上解 `configuration_data.qpos`），
  是**指令**而非实测关节 —— 与 `state[0:27]` 的语义不同。`--skip_retarget` 模式下这一列会是 NaN。
- `quat_rotate_inverse_numpy(q, v) == R(q)ᵀ·v`（已实测，误差 1e-15）。所以 `[30:33]` 是
  **世界系位移转进 anchor 当前机体系**，再除以 `dt`。**不是** anchor 相对自己过去的位移。
- 差分用的是**最近两帧收到的 pico payload**（`prev_body_pos_w` / `prev_timestamp_ns`），
  所以 `dt` 是 **pico 的发布周期**，**不是** 1/30 的录制周期。
- `[33:40]` 是**原样搬运**的原始世界系 anchor 位姿，落盘时不做任何变换
  （DEBUG 用：可直接与 `motion_mf.npz` 对账，也用于量化 `yaw_vel` 积分漂移）。
  **重建时应优先用它们，而不是靠积分 `yaw_vel`。**
- ⚠️ **若该拍还没收到任何 pico 帧**，`reference_action_live_latest` 返回中位兜底
  （`joint_pos` 全 0、`quat=[1,0,0,0]`、`lin_vel=0`、`pos_w=0`），会写出**一行全零 action**。
  数据清洗时值得过滤。注意 hub 在操作者**第一次按下 X 之前一个包都不发**
  （`_live_started` 闸门），所以"刚启动就开录"很容易整段都是这种行。
- ⚠️ anchor 索引是**按名字在 `body_names_simulation` 里的位置**去索引 pico 流的 `body_pos_w`
  （[jaka_lerobot.py:563-564](src/simple/datasets/jaka_lerobot.py#L563)）——
  录制端传的是策略 yaml 的 `body_names_simulation`（**MJCF 顺序，`waist_yaw_Link` = 13**），
  而 hub 发布的 body 数组正好也是 MJCF 顺序（`sim2real/config/robots/jaka.py` 的 `JAKA_BODY_NAMES`），
  两边一致才取对。实测佐证：录制到的 `actions[:,35]`（anchor 世界 z）在整个 episode 上
  mean ≈ 0.831、p95 ≈ 0.845，贴近腰高（参考运动里 `waist_yaw_Link` z ≈ 0.84），
  而**远离**同位置若取错成 `Neck_pitch_Link` 的高度（≈ 0.99）。**换 hub 或改配置时务必重验。**

---

## 5. 时间语义（重要）

- **每一行只有一个时间基准**：录制循环里的 `t_ns = int(time.time()*1e9)`
  （[record_jaka_zmq.py:516](src/simple/cli/record_jaka_zmq.py#L516)），**同时**喂给
  `state_root_calc` 和 `action_root_calc`，好让机器人和参考的 roll/pitch/yaw_vel 共享同一时钟基准。
- **这个 `t_ns` 没有落盘**。数据集里的 `timestamp` 是 `frame_index / fps` 合成值（已实测：
  0, 1/30, 2/30, …）。**真实帧间隔（含抖动）不可恢复**，只能按标称 `dt = 1/fps` 近似。
- 后果：用标称 `dt` 去积分 `yaw_vel` 会漂移 —— `yaw_vel` 是按**真实抖动间隔**算的真速率。
  代码注释给出的量级是 **≈2.6 deg/s**（[jaka_lerobot.py:411-416](src/simple/datasets/jaka_lerobot.py#L411)）。
- `state` 流的 `smplx_t_ns` 是**仿真时间** `mjData.time`（[jaka_zmq_pub.py:174](src/simple/interfaces/jaka_zmq_pub.py#L174)），
  `publish_t_ns` 才是本机墙钟 `time.time_ns()`。录制端用 `publish_t_ns`（缺省才退到 `smplx_t_ns`），
  两个时钟域不要混用。

---

## 6. 参考系总表

### 6.1 四个系

| 名字 | 是什么 | 谁在用 |
|---|---|---|
| **MuJoCo 世界系** | 仿真的世界原点，z 朝上，含**桌子/物体/房间**；机器人在其中有自己的 spawn 位姿（该任务的 `robot_region` 给 `x ≈ -1.45`，桌面中心 `x = 0.6`）。注意 `body_*_w` 是世界系，而 payload 里另一个 key `"qpos"` 是**挂载 frame 局部**的 | `state` 的 root 旋转；`body_*_w` 全体 |
| **`base_link`** | 机器人浮动基座体，挂 free joint，因此 `qpos[0:3]` / `qpos[3:7]` 就是它的世界位姿 | **`state[27:30]` 的 roll/pitch/yaw_vel** |
| **`waist_yaw_Link`（= `waist_imu` site）** | `base_link` 的**直系子体**，`pos="0 0 0.22385"`，`waist_imu` site 在其原点且姿态为单位阵 | **策略观测的机器人 root**（gravity / anchor_ori / align_quat）；`actions` 的 anchor |
| **pico 参考世界系（GMR 重定向系）** | `pico_retarget_pub` 里 GMR 的**只有机器人 + 地面**的 MJCF（`GMR/assets/Khan_jaka/jakamini.xml`）的世界。原点 = **操作者 live 开始那一刻的骨盆位置**（`xrobot_to_jaka.json` 把 `base_link ← Pelvis` 零偏移对齐），地面在 `z=-0.05`。与机器人仿真世界**不是同一个系**：yaw 有任意偏置、平移有任意偏置（**平移偏置全流程不做对齐**，靠策略只用相对位移化解） | `actions[33:40]` 原样保存 |

`actions` 的三个系（同一帧内）：
`[0:27]` 关节角（无系）、`[27:30]` anchor 姿态角、`[30:33]` **anchor 机体系**线速度、
`[33:40]` anchor **pico 世界系**位姿。

### 6.2 为什么策略观测非要用 `waist_yaw_Link`

策略的 state getter 优先读 **`waist_imu` site 的 `framequat` / `gyro` 传感器**，
读不到才退回 `qpos[3:7]` / `qvel[3:6]`
（[jaka.py:231-258](src/simple/robots/jaka.py#L231)，`prepare_obs`）：

```xml
<body name="waist_yaw_Link" pos="0 0 0.22385">
  <site name="waist_imu" pos="0 0 0" quat="1 0 0 0"/>
  ...
  <joint name="waist_yaw_joint" pos="0 0 0" axis="0 0 1" .../>
```
```xml
<gyro name="waist_gyro" site="waist_imu"/>
<framequat name="waist_quat" objtype="site" objname="waist_imu"/>
```

site 在 body 原点、姿态单位阵 ⇒ `root_quat_w` 就是 **`waist_yaw_Link` 的世界姿态**，
`root_ang_vel_b` 是**该体系**下的陀螺读数。

### 6.3 ⚠️ `base_link` 与 `waist_yaw_Link` 的关系

MJCF 里 `waist_yaw_Link` 是 `base_link` 的直系子体，铰链轴 `0 0 1`、位于子体系原点，故

```
R_waist = R_base · Rz(q_waist)          # q_waist = waist_yaw_joint 角
yaw_waist = yaw_base + q_waist
```

（已数值验证，误差 5.6e-16）

**因此**：

- 录制存入 `state[27:30]` 的 roll/pitch/yaw_vel 是 **`base_link`** 的；
- 策略观测里 `gravity` / `anchor_ori` / `align_quat` 用的机器人 root 是 **`waist_yaw_Link`** 的；
- 两者**在腰关节非零时不等**，差的就是 `q_waist`（= `state[12]`，JAKA 顺序下 `waist_yaw_joint`）。

**好消息**：投影重力对 yaw 免疫，且有精确换算

```
gravity_waist = Rz(-q_waist) · gravity_base
```

（已数值验证）。所以**重力项是可以从本数据集精确重建的**，见 §7。

**坏消息**：`base_link` 的**绝对 yaw 没有被记录**（只存了 yaw_vel），所以任何需要
机器人**完整四元数**的量（`anchor_ori`、`align_quat`）都**无法**从本数据集精确重建。

### 6.4 ⚠️ 两条流的 body 顺序**不一样**

| 流 | body 列表来源 | `base_link` | `waist_yaw_Link` |
|---|---|---|---|
| state `:28711` | `jaka_lerobot.JAKA_BODY_NAMES` = **npz/policy 顺序** | 0 | **3** |
| pico `:28701` | hub 的 `robot_cfg.body_names` = **MJCF 顺序** | 0 | **13** |

`base_link` 两边都是 0（所以 `body_quat_w[0]` 恒为 `base_link`，这是安全的），
但**别的位置做任何逐位比较都是错的**。录制端靠"按名字查位置"分别处理了这两条流，
所以落盘数据本身是对的 —— 只是**读的人**要清楚 `state` 侧和 `action` 侧的编号体系不同。

---

## 7. 从录制数据重建策略观测

冻结策略的观测是 **620 维**（MF v1，`jaka_frame_stack_mf`，ONNX 输入名 `obs`）：

```
[  0:155]  command      = root_pos_diff_b(15) + root_z_mf(5) + ref_joint_pos(135)
[155:185]  anchor_ori   = rot6d × 5
[185:200]  gravity      × 5 帧堆叠
[200:215]  base_ang_vel × 5 帧堆叠   （×0.25）
[215:350]  dof_pos      × 5 帧堆叠   （q − q_default，policy 顺序）
[350:485]  dof_vel      × 5 帧堆叠   （×0.05，policy 顺序）
[485:620]  last_action  × 5 帧堆叠
```

每个 5 帧块内是 **最旧 → 最新**（`[x_{t-4}, …, x_t]`）展开。详见
[jaka_mf.py:254-262](src/simple/jaka_rl/observations/jaka_mf.py#L254)。

### 7.1 可重建性（结论先行）

| 观测块 | 能否从本数据集重建 | 依据 |
|---|---|---|
| `command` 全部 155 维 | ✅ **精确** | `actions[:, 0:27]` + `actions[:, 33:40]`（原始世界 anchor 位姿） |
| `anchor_ori` 30 维 | ⚠️ 仅差 yaw 漂移 | 需要机器人 `waist_yaw_Link` **完整四元数**，数据集只存了 roll/pitch/yaw_vel |
| `gravity` 15 维 | ✅ **精确** | `Rz(-state[12]) · gravity_from(state[27:29])`（§6.3） |
| `base_ang_vel` 15 维 | ❌ | 数据集里**根本没有角速度**（只有 yaw_vel） |
| `dof_pos` 135 维 | ✅ **精确** | `state[:, 0:27]`（需 MuJoCo→policy 顺序重排 + 减 `default_joint_pos`） |
| `dof_vel` 135 维 | ❌ | 数据集里**没有关节速度**（可差分，但有噪声/非标称 dt） |
| `last_action` 135 维 | ❌ | 策略自身输出的历史，录制时不存在 |

**结论**：本数据集是给 **VLA 训练**用的（state + image + 参考动作），
**不是**完整 MF 策略观测的快照。要完整 620 维观测，只能**重跑仿真/策略**
（`teleop_jaka_mf.py` + 同一 `--policy-config`）或让录制端额外存这些量。
`state` / `actions` 自洽可用，但别指望把它们拼成策略观测。

### 7.2 参考侧重建（`command`，精确）

对每个采样时刻，取其**后 5 帧**（`future_steps=[0..4]`）作为窗口：

```python
anchor_pos  = actions[k, 33:36]      # 原始世界系，精确
anchor_quat = actions[k, 36:40]      # 原始世界系 wxyz，精确

root_pos_diff_b = R(anchor_quat[0])ᵀ · (anchor_pos[k] - anchor_pos[0])   # 15
root_z_mf       = anchor_pos[k, 2]                                        # 5，绝对世界 z
ref_joint_pos   = actions[k, 0:27] 重排 MuJoCo→policy 顺序                # 135
```

- `root_pos_diff_b` 用的是**窗口第 0 帧**的 anchor 四元数（不是各未来帧自己的系）。
- `root_pos_diff_b` 对世界平移不变，所以 pico 世界系与仿真世界系的**任意平移差不影响它** ——
  这也是全流程**从不做平移对齐**的原因。
- `root_z_mf` 是**绝对 z**，落在 pico/GMR 世界里（地面 `z=-0.05`，两者都是 z 朝上的地面系），
  所以它给的是"相对重定向世界地面的高度"。两个世界的 z 轴约定一致，但**绝对高度基准不同**，
  跨世界直接用这一项时要留意。
- 现成实现：`mf_command()` / `reconstruct_anchor_motion()`
  （[action_trunk_reconstruct.py](src/simple/jaka_rl/action_trunk_reconstruct.py)）。
  自检可跑：`PYTHONPATH=src python -m simple.jaka_rl.action_trunk_reconstruct`
  （实测：位置误差 3e-8 m，155 维 command 误差 4e-9）。注意自检里 `fps` 硬编码 **50**，
  而录制数据是 **30**，套用时要改。

### 7.3 `anchor_ori`（差一个 yaw）

```
ori_b = conj(q_robot_waist) ⊗ (align_quat ⊗ q_ref_anchor_k)
```
`align_quat = yaw(robot_waist) ⊗ yaw(ref_anchor)⁻¹`，是**纯绕世界 z** 的对齐，
由 motion buffer 在参考流「空→非空」边沿重算一次
（[motion_buffer.py:327-353](src/simple/jaka_rl/motion_buffer.py#L327)）。

- 参考侧的 `q_ref_anchor` 可直接取 `actions[:, 36:40]`（精确）。
- 机器人侧的 `q_robot_waist` **不在数据集里** ⇒ 只能：积分 `state[:,29]` 得到近似 yaw
  （会漂移），或假设初始对齐、只关心增量。
- ⚠️ `command` 用的是**未对齐**的原始参考四元数，只有 `anchor_ori` 才应用 `align_quat`。
- ⚠️ `align_quat` 只在参考流「空→非空」边沿盖一次章。hub 启动时会先发一段
  **冻结的 idle 快照（世界 yaw ≈ 0）**，所以实际盖章时刻是 **hub/teleop 启动**，
  **不是**操作者按下 X 开始驱动的时刻。录制数据里 `actions[33:40]` 的原始朝向也带着这个
  未对齐的初始 yaw。

---

## 8. 已知坑（写代码前先看）

1. **`timestamp` 是假的** —— 不是墙钟，真实帧间隔不可恢复（§5）。
2. **state 的 root 是 `base_link`，策略观测的 root 是 `waist_yaw_Link`** —— 差一个 `q_waist`（§6.3）。
3. **`yaw_vel` 无 EMA、无 clip**，可能有尖峰（实测 19.7 rad/s）；`_yaw_vel_ema` 这个名字是误导。
   首帧恒为 0。
4. **绝对 yaw 不在数据集里**，也恢复不出来。
5. **没有关节速度、没有角速度** —— 别把它们当"差分一下就有"，抖动 `dt` 会让差分不可控。
6. **pico 未就绪时会产生全零 action 行**（§4），训练前建议过滤。
7. **anchor 索引是按位置**索引 pico 流的 body 数组，依赖 hub 的 body 顺序与配置一致（§4）。
8. **`scripts/replay_jaka_action_mujoco.py` 已失效** —— 它 import 的
   `reconstruct_base_qpos` 在 `action_trunk_reconstruct.py` 里**并不存在**，跑会直接 ImportError。
   需要"从 action 重建轨迹"请用 `mf_command()` / `reconstruct_anchor_motion()`。
9. 数据集创建时**第一帧相机到达前** `exporter is None`，此前的帧不落盘；episode 结束
   若一帧都没收到会打印 "Stop toggled with no frames — episode skipped"。
10. 代码注释里反复引用的 `doc/FSMMimicJakaMiniZmq.cpp` / `doc/RealtimeMotionBuffer.cpp`
    **在任何 checkout 里都不存在**，别去找；yaw 对齐的现有说明只在
    [/home/xu/code/copy/simple_jaka_yaw_align_20260914_live_edge/NOTES.md](../copy/simple_jaka_yaw_align_20260914_live_edge/NOTES.md)
    （那是**另一个变体**的备份，触发条件不同，别当成当前行为）。
11. sim2real 参考仓库里**没有** `align_quat` —— 这套 yaw 对齐是 SIMPLE / C++ 部署侧新增的，
    两边的 `motion_buffer` 并不等价。

---

## 9. 相关文件

| 文件 | 作用 |
|---|---|
| [src/simple/cli/record_jaka_zmq.py](src/simple/cli/record_jaka_zmq.py) | 录制主程序（本文档主角） |
| [src/simple/datasets/jaka_lerobot.py](src/simple/datasets/jaka_lerobot.py) | 落盘格式、`OpenHLMRootVel`、`reference_action_live_latest` |
| [src/simple/interfaces/jaka_zmq_pub.py](src/simple/interfaces/jaka_zmq_pub.py) | state / camera 发布端（`base_link` 就是这里定的） |
| [src/simple/jaka_rl/config.py](src/simple/jaka_rl/config.py) | 关节/body 名字与顺序、ZMQ 端口、payload key |
| [src/simple/jaka_rl/motion_buffer.py](src/simple/jaka_rl/motion_buffer.py) | 策略侧的参考流缓冲、`align_quat` |
| [src/simple/jaka_rl/observations/jaka_mf.py](src/simple/jaka_rl/observations/jaka_mf.py) | 620 维观测的确切定义 |
| [src/simple/jaka_rl/action_trunk_reconstruct.py](src/simple/jaka_rl/action_trunk_reconstruct.py) | 从 action trunk 反推参考轨迹与 command |
| [data/jaka_mf/latest56k_pico_dr.yaml](data/jaka_mf/latest56k_pico_dr.yaml) | 策略配置（`future_steps`、`anchor_body_name`、顺序表） |
| [scripts/verify_jaka_action_recon_error.py](scripts/verify_jaka_action_recon_error.py) | 录制的 action 与 `motion_mf.npz` 真值的对账脚本 |
| `/home/xu/code/cxr/sim2real-jaka/sim2real/teleop/pico_retarget_pub.py` | **在用的** pico hub（`_build_payload` :475），定义 `actions` 的来源与 GMR 世界系 |
| [JAKA_QUICKSTART.md](JAKA_QUICKSTART.md) · [TRAINING_DATA_FORMAT.md](TRAINING_DATA_FORMAT.md) | 操作流程 / 训练侧格式 |
