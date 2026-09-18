# Implementation Analysis: SIMPLE-jaka-main VLA 部署参考轨迹缓冲改造

> 本文档面向后续接手的 code agent：说明本次在 `src/SIMPLE-jaka-main/` 中做的全部修改、
> 修改动机、对应的 sonic 端（GR00T-WholeBodyControl4OpenHLM）设计哲学、接口契约与
> 调试/排查指南。阅读本文后可无缝继续修改、完善与 bug 排查。

---

## 1. 背景与目标

### 1.1 整体架构

OpenHLM 流水线采用 **高层 VLA + 低层 tracker 策略** 的分层架构：

```
┌────────────────────────────────────────────────────────────┐
│ 高层 π0.5 VLA (openpi4OpenHLM)                             │
│   serve_policy.py: WebSocket + msgpack-numpy (:8000)       │
│   输出 27 维关节动作块 (action_horizon=50)                  │
└──────────────────────┬─────────────────────────────────────┘
                       │ WebSocket
┌──────────────────────▼─────────────────────────────────────┐
│ openpi-eval 客户端 (30Hz 控制环, 每块 25 步开环执行)         │
│   推理停顿 ~120ms+ (阻塞式) 时不下发任何数据                  │
│   每个动作步打包为二进制协议 v1 消息, ZMQ PUB 下发            │
└──────────────────────┬─────────────────────────────────────┘
                       │ ZMQ PUB/SUB (二进制 v1, 2 帧滑动窗口)
┌──────────────────────▼─────────────────────────────────────┐
│ 参考轨迹缓冲 (RealtimeMotionBuffer / RealtimeMotionBufferVla)│
│   接收 30Hz 动作流 → 缓冲/插值 → 供 tracker 观测解算          │
└──────────────────────┬─────────────────────────────────────┘
                       │ get_obs() → MotionData (5 未来帧)
┌──────────────────────▼─────────────────────────────────────┐
│ JAKA MF tracker 策略 (jaka_frame_stack_mf, 620 维观测)      │
│   command(155) + anchor_ori(30) + history(5×87) → 27 维动作 │
└────────────────────────────────────────────────────────────┘
```

### 1.2 本次改造解决的问题

openpi-eval 客户端是 **30 Hz 下发 + 阻塞式推理停顿**（每次重推理约 120ms+ 无数据）。
原版 `RealtimeMotionBuffer`（墙钟时间轴）在停顿期与恢复期存在两个语义问题：

1. **停顿期**：播放点 `P = wall_now - delay` 随墙钟继续前进，把缓冲里剩余的
   120ms 尾巴逐步播完，未来帧渐进 clamp 塌缩——参考轨迹不冻结。
2. **恢复期**：新帧时间戳 = 墙钟到达时刻，与旧末帧之间留下一个等于停顿时长的
   **时间轴空洞**；播放窗口扫过空洞时在"旧保持帧 ↔ 新轨迹首帧"之间做跨大间隙
   lerp/slerp——产生 ~停顿时长 的缓入（ease-in），新动作块的起始轨迹被时间拉伸失真。

改造目标：让参考轨迹在 VLA 推理停顿期间 **位级冻结**、恢复时 **无混合直入新块**、
播放延迟 **恒为 120ms 不随停顿累积**——即 sonic 端"游标钳制 + 帧号连续"的语义，
但保留时间域插值重采样能力。

---

## 2. 参考设计哲学（sonic 端）

sonic 端 = `src/GR00T-WholeBodyControl4OpenHLM` 的 `g1_deploy_onnx_ref`（C++）
中的 `MotionSequence "streamed"` 播放机制：

| 维度 | sonic "streamed" | 原版 RMB（墙钟版） |
|---|---|---|
| 索引轴 | 发送端全局 `frame_index`（单调） | 到达墙钟时刻 |
| 停顿期 | 游标被"预留窗口=11 帧"钳制 → **参考轨迹位级冻结** | 播放点继续前进 → 渐进冻结/播完尾巴 |
| 恢复期 | 窗口滑动 + `frame_offset_adjustment` 原位续接，**无插值无混合** | 跨大间隙插值 → 缓入失真 |
| 延迟 | 恒 ~12 帧（30Hz≈0.4s），不累积 | 120ms（但恢复期有额外失真） |
| 帧率适配 | 无重采样（阶梯参考） | 时间域 lerp/slerp 插值（30→50Hz 平滑） |

**设计哲学提炼**（本次 Python/C++ 两端的共同改造原则）：

1. **数据驱动的播放时钟**：播放进度不由墙钟直接驱动，而是 `P = min(P + dt, newest - delay)`
   ——"最新数据 − 固定延迟"是播放点的天花板。数据停则播放停，数据行则播放行。
2. **停顿期时间轴暂停 + 恢复时重锚定**：停顿的墙钟间隙**不允许**出现在数据时间轴上。
   恢复后的第一帧重锚为"上一帧 + 名义帧周期"，从而恢复期只是"一次正常帧距插值"。
3. **cleanup 跟随播放时钟**：旧帧清理线 = `P - history`，而非墙钟。保证停顿期不删除
   尚未播放的保留尾巴（等价 sonic 的 reserve 窗口）。
4. **冗余帧去重**：发送端 2 帧滑动窗口 `[i-1, i]` 与接收端 frame_index 去重配合，
   单条消息丢失不产生帧空洞。

---

## 3. C++ 参考实现（先行完成，语义基准）

`src/KpiDeployReal/` 中已完成同语义的 C++ 移植（本次 Python 版的对标基准）：

- `src/KpiDeployReal/include/RealtimeMotionBufferVla.hpp` — 独立新类声明
- `src/KpiDeployReal/src/RealtimeMotionBuffer_vla.cpp` — 实现（5 项核心修改）

C++ 版 5 项核心修改（Python 版逐条对应，见第 4 节）：

1. 双协议入口（JSON 兼容 pico + 二进制 v1 适配 openpi-eval）
2. frame_index 去重（滑动窗重叠帧只插一次）
3. 时间戳重锚定（`compute_anchor_ts_locked`，gap > 阈值 → `back + nominal`）
4. 数据驱动播放时钟（`get_obs` 中 `P = min(P+dt, newest-delay)`，"钉 cap"）
5. cleanup 基于播放时钟 P（`cutoff = P - history`）

---

## 4. Python 迁移实现（本次修改内容）

### 4.1 修改文件清单

| 文件 | 修改 | 影响面 |
|---|---|---|
| `src/simple/jaka_rl/motion_buffer.py` | 新增 `RealtimeMotionBufferVla` 类（约 500 行）+ 模块级 `_resolve_default_posture` / `_decode_binary_v1` 辅助函数；`__all__` 增加导出。**旧 `RealtimeMotionBuffer` 类零改动** | 纯新增 |
| `src/simple/jaka_rl/state_processor.py` | ① import 新类；② `motion_buffer` 类型注解改为 union；③ `_init_motion_backend` 新增 `motion_backend == "zmq_vla"` 分支（实例化新类）；④ `_update_motion_data` 分支改为 `in ("zmq", "zmq_vla")` | 新增后端 |
| `src/simple/jaka_rl/__init__.py` | 导出 `RealtimeMotionBufferVla` | 导出 |

**未修改**：`observations/jaka_mf.py`（消费接口完全兼容，见 5.1）、旧类 `RealtimeMotionBuffer`、
`cli/teleop_jaka_mf.py`（通过新增兼容属性适配，见 4.6）。

### 4.2 类设计决策（已与需求方逐项确认）

| 决策点 | 结论 | 理由 |
|---|---|---|
| 类结构 | **独立新类** `RealtimeMotionBufferVla`，旧类零改动 | 镜像 C++ 迁移；pico 遥操作/录制链路零风险 |
| 二进制关节序 | **已是 SIM/JAKA 序**（openpi 客户端已重排），入库不重排 | `jaka_mf` 的 `motion_joint_indices` 解算按原样工作 |
| 二进制 body 语义 | 流只携带 **anchor（waist_yaw_Link）位姿**；入库只填 anchor 槽位，其余 body 用默认姿态 FK 值填充 | `jaka_mf` 只读 anchor body；`MotionData` 是多 body 结构必须填满 |
| 接入方式 | `motion_backend: "zmq_vla"` 新后端值 | 一个配置键切换，旧 `"zmq"` 不动 |
| 特性范围 | **最小 VLA 核心接口**（无 toggle_data_collection 录制电平、无 npz replay 回退、无完整 diagnostics） | VLA 部署不需要录制；空流回退为默认 FK 站姿 |
| 配置 | 端口沿用 `motion_zmq_connect`；新增可选键 `motion_nominal_frame_s`（默认 1/30）、`motion_gap_threshold_s`（默认 0.05）；dt/tolerance 沿用 `motion_dt_s`/`motion_tolerance_s` | 最小配置面 |

### 4.3 五项核心修改（与 C++ 逐条对应）

#### [1] 双协议入口 — `_start_motion_stream` 的 `_stream_loop`

```python
raw = sock.recv(flags=zmq.NOBLOCK)     # 注意: 旧类用 recv_string, 二进制非 UTF-8
if raw and raw[:1] == b"{":
    self._handle_json_message(raw.decode("utf-8"))   # JSON: pico 遥操作兼容
else:
    self._handle_binary_message(raw)                 # 二进制 v1: openpi-eval
```

二进制 v1 线格式（`_decode_binary_v1` 模块级函数）：

```
[可选 topic "pose" 4B][1280B JSON 头, null 填充][按头中字段顺序拼接的二进制负载]
头: {"v":1,"endian":"le","fields":[{"name":"joint_pos","dtype":"f32","shape":[N,27]}, ...]}
必需字段: joint_pos(N,27) f32 | frame_index(N,) i64 | body_pos_w(N,3) f32 | body_quat_w(N,4) f32(wxyz)
忽略字段: joint_vel(N,27) f32 | action_hand_left/right
```

要点：头取首个 `\x00` 前的有效 JSON 段；大小端用 `np.dtype(...).newbyteorder(">"/"<")` 处理；
版本仅接受 v1；形状严格校验（帧数一致、关节数 == `len(joint_names)`），失败打
`binary decode failed` / `joint dim mismatch` / `body dim mismatch` 日志并丢弃该消息。

#### [2] frame_index 去重 — `_handle_binary_message`

```python
new_rows = [i for i in range(n) if frame_index[i] > self._last_frame_index]
if not new_rows: return                 # 2 帧滑动窗重叠帧 → 静默丢弃
self._last_frame_index = int(frame_index.max())
# 多新帧(仅首包可能): 时间戳按 anchor - (cnt-1-j)*nominal 错开, 最新帧落在 anchor 上
```

`_last_frame_index` 初始 -1、**在 clear() 中刻意保留**（客户端帧号单调，clear 后旧帧仍需去重，与 C++ 一致）。

#### [3] 时间戳重锚定 — `_compute_anchor_ts_locked`（JSON/二进制共用）

```python
wall = time.time_ns()
if not self._timestamps_ns:
    anchor = 0                          # 数据时间轴原点
else:
    delta = wall - self._last_arrival_wall_ns
    if delta > self._gap_threshold_ns:          # >50ms → 判定为 VLA 推理停顿
        anchor = self._timestamps_ns[-1] + self._nominal_frame_ns   # 时间轴暂停, 重锚 33.3ms
    else:
        anchor = self._timestamps_ns[-1] + delta                    # 正常流按到达间隔推进
self._last_arrival_wall_ns = wall
```

**核心思想**：数据时间轴在停顿期间"暂停"，停顿的墙钟间隙**不进入**数据轴 → 恢复期
不存在大间隙插值区间。

#### [4] 数据驱动播放时钟 — `get_obs`（核心语义）

```python
if not self._playback_initialized:
    self._playback_time_ns = newest - self._delay_ns        # 首个数据: 对齐到 delay 之后
    self._playback_initialized = True
else:
    self._playback_time_ns += self._dt_ns                    # 每 policy tick +20ms
    cap = newest - self._delay_ns                            # 天花板 = 最新帧 - 120ms
    if self._playback_time_ns > cap:
        self._playback_time_ns = cap                         # "钉 cap": 停顿期 P 冻结
target_times_ns = (self._playback_time_ns + self._future_steps_ns).reshape(1, -1)
```

- `_delay_ns = max_future_step*dt + tolerance = 4×20ms + 40ms = 120ms`；
- 不变量：`P ≤ newest - 120ms` → 5 个未来采样目标（P+{0,20,40,60,80}ms）永远
  ≤ 最新帧 → 前视永不塌缩；
- 假设 `get_obs()` 每个控制 tick（50Hz, 20ms）恰好被调用一次（`state_processor.update`
  → `_update_motion_data` 链路保证）；
- `clear()` 时 `_playback_time_ns/_playback_initialized` 重置，下一批数据重新对齐。

#### [5] cleanup 基于播放时钟 — `get_obs` 内

```python
cutoff_ns = self._playback_time_ns - self._history_ns       # 用 P 而非墙钟
self._cleanup_locked(cutoff_ns)                             # 始终保留至少 1 帧
```

停顿期 cutoff 冻结 → 未播放的 120ms 保留尾巴不被删除 → 恢复后先播完尾巴再进新块。

### 4.4 其他保留功能（从旧类移植，`jaka_mf` 依赖）

- **首帧 yaw 对齐**：`_update_align_quat` 在空→非空边沿用机器人实况 `mj_data` 的
  anchor 姿态计算 `align_quat = yaw(robot) * yaw(ref)^-1`，随 `MotionData.align_quat`
  传给 obs（`jaka_mf._compute_anchor_ori` 消费）。`clear()` 时重置对齐。
- **默认姿态 FK**：`_resolve_default_posture` 在 **SCRATCH** `MjData` 上 FK
  `default_qpos`（不可写实况 mj_data，避免重置场景位姿）；FK 失败回退全零并打 error 日志。
- **接口**：`ready()` / `clear()` / `close()` / `latest_timestamp_ns`（数据时间轴）/
  `stale_ms()` / `playback_time_ns()`。
- **兼容属性**：`latest_toggle_data_collection` 恒返回 `None` —— `teleop_jaka_mf`
  主循环在 `reset_on_record_end=true` 时访问该属性（`level is not None` 才触发 reset），
  VLA 流无录制电平，返回 None 即该触发条件天然失效（避免 AttributeError）。

### 4.5 修改后的行为语义（tick 级结论）

以 `delay=120ms`、`dt=20ms`、30Hz 流、停顿 120ms（=6 tick）为基准（与 C++ 版推演一致）：

| 阶段 | 行为 |
|---|---|
| 停顿期 | newest/cap/P/cutoff/buffer 全部冻结 → 5 个未来参考帧**位级一致**，机器人停在停顿开始时的执行姿态（= 头帧前 120ms 处） |
| 恢复期 | 首帧重锚为 `上帧+33.3ms`；P 以 20ms/tick 扫过保留尾巴（~6 tick），随后以**一次正常 33.3ms 插值**跨过块边界直入新块，零缓入失真 |
| 稳态 | P 钉在 cap 上，播放净速率 = 30Hz 到达速率；`newest - P` 恒 = 120ms，**不随停顿次数累积** |
| 空流 | 回退为默认 FK 站姿窗口（与旧类空缓冲语义一致） |

### 4.6 state_processor 接入点

```python
elif motion_backend == "zmq_vla":
    self.motion_buffer = RealtimeMotionBufferVla(
        joint_names=self.joint_names,
        body_names=self._body_names or [],
        future_steps=self.motion_future_steps,
        mj_model=self._mj_model, mj_data=self._mj_data, default_qpos=self._default_qpos,
        motion_zmq_connect=self.motion_config.get("motion_zmq_connect", "tcp://127.0.0.1:28701"),
        motion_zmq_hwm=int(self.motion_config.get("motion_zmq_hwm", 1)),
        dt_s=float(self.motion_config.get("motion_dt_s", 0.02)),
        tolerance_s=float(self.motion_config.get("motion_tolerance_s", 0.04)),
        nominal_frame_s=float(self.motion_config.get("motion_nominal_frame_s", 1.0/30.0)),
        gap_threshold_s=float(self.motion_config.get("motion_gap_threshold_s", 0.05)),
    )
    self.motion_joint_names = list(self.motion_buffer.joint_names)
    self.motion_body_names = list(self.motion_buffer.body_names)
```

`_update_motion_data`：`elif self.motion_backend in ("zmq", "zmq_vla"):`
→ `self.motion_data = self.motion_buffer.get_obs()`（`_using_zmq_replay()` 恒 False，
npz replay 回退只属于旧 "zmq" 路径，`zmq_vla` 空流即默认站姿）。

---

## 5. 接口契约（后续 agent 必须遵守）

### 5.1 MotionData 消费契约（`jaka_frame_stack_mf`）

`get_obs()` 返回 `MotionData`，字段与旧类完全一致：

```
joint_pos      (1, 5, 27)   SIM/JAKA 关节序（客户端已排好, 缓冲不重排）
joint_vel      (1, 5, 27)   全零（二进制流 joint_vel 解析后丢弃, 与 C++ 一致）
body_pos_w     (1, 5, nb, 3)  anchor 槽位=流内位姿, 其余 body=默认 FK 姿态
body_lin_vel_w (1, 5, nb, 3) 全零
body_quat_w    (1, 5, nb, 4) wxyz, 同上填充规则
body_ang_vel_w (1, 5, nb, 3) 全零
motion_id / step / timestamps_ns  (1,5) 模板
align_quat     (1, 4)  首帧 yaw 对齐结果(空→非空边沿重算)
```

`jaka_mf` 消费点：`motion_data.body_pos_w[0,:,anchor_idx]`、
`motion_data.body_quat_w[0,:,anchor_idx]`、`motion_data.joint_pos[0][:, self._motion_joint_indices]`
（obs 侧再做 SIM→IsaacLab 重排）、`motion_data.align_quat`。**任何修改不得破坏这些键/形状。**

### 5.2 配置契约（`data/jaka_mf/teleop_jaka_mf.yaml` 的 `motion:` 段）

```yaml
motion:
  motion_backend: zmq_vla            # 新后端(旧 "zmq" 仍指向旧类, pico 遥操作不变)
  motion_zmq_connect: "tcp://127.0.0.1:28701"   # 沿用, 双协议自动识别
  motion_dt_s: 0.02                  # 沿用
  motion_tolerance_s: 0.04           # 沿用
  motion_nominal_frame_s: 0.0333     # 可选, 默认 1/30
  motion_gap_threshold_s: 0.05       # 可选, 默认 0.05
```

### 5.3 上游（openpi-eval 客户端）契约

- 30 Hz 下发，二进制 v1，2 帧滑动窗口 `[i-1, i]`（frame_index 单调递增、跨动作块连续）；
- `joint_pos` 27 维 **SIM/JAKA 序**；`body_pos_w (N,3)` / `body_quat_w (N,4) wxyz` 为
  **anchor（waist_yaw_Link）** 位姿；
- 字段名/形状不符时新类打 warning 丢弃（不崩溃），但策略将拿不到参考——排查时先看该日志。

---

## 6. 调试与排查指南

### 6.1 诊断接口

| 接口 | 语义 | 用途 |
|---|---|---|
| `stale_ms()` | 距上次到达的墙钟毫秒（0=无数据） | 停顿检测：VLA 推理间隙应周期性上升到 ~推理时长 |
| `playback_time_ns()` | 当前播放点 P（数据时间轴） | 停顿期应**冻结**；恢复后应逐步扫过保留尾巴 |
| `latest_timestamp_ns` | 最新帧数据时间（重锚定后） | **核心不变量**：`latest - playback == delay(120ms)` 恒成立；若随停顿漂移增大 → 重锚定/时钟逻辑被破坏 |
| `ready()` | 缓冲非空 | 空流回退判定 |

### 6.2 日志消息与故障模式

| 日志 | 含义 | 排查方向 |
|---|---|---|
| `binary decode failed: ...` | 二进制消息不合法（版本/缺字段/长度） | 客户端打包格式与 v1 不符 |
| `joint dim mismatch: (N,27) != (N, J)` | 关节数不符 | 客户端动作维度/顺序错误 |
| `body dim mismatch` | body 字段形状不是 (N,3)/(N,4) | 客户端 body 打包形状错误 |
| `default posture FK failed` | 默认姿态回退全零 | 空流时参考将塌到原点（危险），检查 mj_model/joint_names |
| `align_quat unavailable` | 读不到实况机器人 anchor 姿态 | mj_data 未接入或 body 名不符；align_quat 恒等 |
| 无日志但参考不动 | 流被去重/丢弃 | 检查 `_last_frame_index` 是否被客户端更小的帧号卡住 |

### 6.3 常见陷阱（后续修改务必注意）

1. **`get_obs()` 调用频率 = 播放时钟速率**：`P += _dt_ns` 每调用一次。若将来改动让
   `get_obs()` 每 tick 被调用多次（如诊断轮询），P 会加速冲到 cap——诊断读取请用
   `playback_time_ns()` 只读接口，不要重复调 `get_obs()`。
2. **clear() 不重置 `_last_frame_index`**：客户端帧号单调，重置会导致旧帧被当作新帧
   重复插入（时间戳乱序）。若要支持"客户端重启帧号归零"，需在 clear 时同步重置
   `_last_frame_index = -1`（并接受乱序插入的 bisect 路径）。
3. **JSON 路径同样走重锚定**：pico 遥操作若经 `zmq_vla` 后端运行，>50ms 的操作员暂停
   也会触发重锚定——这是期望行为，但别误以为 JSON 路径是"墙钟语义"。
4. **`_playback_time_ns` 无锁**：仅 policy 线程（get_obs）读写；任何其他线程访问
   必须通过 `playback_time_ns()` 并自担同步。
5. **`timestamps_ns` 是数据时间轴**：`MotionData.timestamps_ns` 已不是墙钟，任何把它
   与 `time.time_ns()` 混用的下游逻辑都会出错（当前下游无此用法）。

---

## 7. 已知限制与后续工作（TODO）

- [ ] **长停顿语义**：停顿 > delay 时机器人冻结在"停顿开始时的执行姿态"（= 头帧前
      120ms），而非最后一帧姿态——这是"前视永不塌缩"约束的必然结果，与 sonic 冻结在
      reserve 边界同构。若需"走完尾巴再停"，需把 cap 从 `newest-delay` 改为
      `newest - max_step*dt`（牺牲前视完整性）。
- [ ] **joint_vel 未存储**：二进制流携带 joint_vel 但被丢弃（`MotionData.joint_vel`
      恒零，与 C++ `InterpolatedFrame` 无速度字段一致）。若未来 tracker 观测需要
      参考速度，需扩展帧存储 + 插值。
- [ ] **`paused`（space 键）语义**：`zmq_vla` 后端下 teleop 的暂停键不会冻结播放时钟
      （get_obs 每 tick 照常推进）。VLA 部署不依赖暂停，如需支持需在
      `state_processor.update` 的 paused 分支处理。
- [ ] **最小诊断**：未移植旧类的完整 `diagnostics()`（payloads/buffered/window_frames/
      clamp 计数等）。若排查需要，可把旧类的诊断计数器平移到新类。
- [ ] **录制兼容**：`zmq_vla` 后端不适合 `record_jaka_zmq` 工作流（录制走 pico 直连，
      与参考缓冲无关；但 `jaka_lerobot.reference_action_live_latest` 依赖旧类的
      `get_latest_frame()`，新类未实现——若要在 VLA 模式下录制，需补齐）。

---

## 8. 相关文件索引（跨仓库）

| 文件 | 角色 |
|---|---|
| `src/KpiDeployReal/include/RealtimeMotionBufferVla.hpp` | C++ 参考实现头（语义基准） |
| `src/KpiDeployReal/src/RealtimeMotionBuffer_vla.cpp` | C++ 参考实现（5 项核心修改的 C++ 版） |
| `src/KpiDeployReal/src/FSMMimicJakaMiniZmq.cpp` | C++ 侧消费方（`get_obs()` → 620 维 obs） |
| `src/SIMPLE-jaka-main/src/simple/jaka_rl/motion_buffer.py` | **本次修改**：`RealtimeMotionBufferVla`（Python 版）+ 旧类（未动） |
| `src/SIMPLE-jaka-main/src/simple/jaka_rl/state_processor.py` | **本次修改**：`zmq_vla` 后端接入 |
| `src/SIMPLE-jaka-main/src/simple/jaka_rl/observations/jaka_mf.py` | 消费方（620 维 obs 解算，未改动，契约见 5.1） |
| `src/SIMPLE-jaka-main/src/simple/jaka_rl/__init__.py` | **本次修改**：导出 |
| `src/GR00T-WholeBodyControl4OpenHLM/gear_sonic_deploy/.../streamed_motion_merger.hpp` 等 | sonic 端设计哲学来源（第 2 节） |
| `src/openpi4OpenHLM/...` / `openpi-eval/` | 上游 VLA 服务器与 30Hz 客户端（契约见 5.3） |

---

*文档生成于 OpenHLM 工作区; 若 C++/Python 两端行为出现分歧, 以 `RealtimeMotionBuffer_vla.cpp` 的语义为基准对齐。*
