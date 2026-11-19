# Implementation Analysis: SIMPLE-jaka-main VLA 部署参考轨迹缓冲改造

> **同步说明（2026-11-07）**：本文与 `src/simple/jaka_rl/{motion_buffer,state_processor,__init__}.py`
> 已从本机 `../SIMPLE-jaka`（commit `d581fc5`）同步，是比仓库旧快照更新的版本。
> 文中 `src/simple/...`、`data/jaka_mf/...` 等路径按 **SIMPLE 仓库根目录**书写；
> 本仓库内的快照位于 `implement_action_analysis/src/simple/`，不是可直接运行的包
> （缺 `config.py` / `math.py` / `motion.py` 等同级模块）。


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
| `src/simple/jaka_rl/base_policy.py` | ⑤ `motion_backend` 的 `Literal` 加 `"zmq_vla"`；⑥ `prepare_policy_config` 的注入分支 `if mb == "zmq":` → `if mb in ("zmq", "zmq_vla"):` | **打通链路的关键**（见 5.2） |
| `data/jaka_mf/teleop_jaka_mf.yaml` | `motion_backend: zmq` → `zmq_vla`（改回 `zmq` 即恢复 pico 遥操作） | 配置切换 |

⑤⑥ 是 2026-09-19 补的：在此之前 `zmq_vla` 虽然能构造出 buffer，但
`prepare_policy_config` 只对 `"zmq"` 注入 ZMQ 连接/频率参数，于是走 `zmq_vla` 时
**teleop 配置里的 `motion_zmq_connect` 被静默忽略**（永远连 `state_processor` 里
硬编码的 28701），且 `motion_dt_s` 停在默认 `0.02`——`rl_rate = 50` 时刚好相等所以看不
出问题，一旦 `rl_rate ≠ 50` 就退化成 6.3 陷阱 1 的延迟无界增长。

**未修改**：`observations/jaka_mf.py`（消费接口完全兼容，见 5.1）、旧类 `RealtimeMotionBuffer`、
`cli/teleop_jaka_mf.py`（通过新增兼容属性适配，见 4.6；`zmq_vla` 下无需改动）。

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

### 4.7 怎么测（本机仿真，无需 VLA 服务器 / 机器人）

默认数据源是 **`data/simple/JakaTabletopPickTeleop-v0/level-0`**（47 集真录数据，
3.4~18.7 s；即 VLA 训练用的那份规范数据集）。两个终端：

```bash
# 终端 B —— 被测的接收端（motion_backend 已设为 zmq_vla），先起来
python -m simple.cli.teleop_jaka_mf simple/JakaTabletopPickTeleop-v0 \
    --headless --controller keyboard --debug-log /tmp/vla.jsonl
#   等到打印 [zmq] state … Hz（约 13 s）

# 终端 A —— 仿真 openpi-eval 下发（默认就是上面那份数据集）
python scripts/test_vla_motion_stream.py --episode 0
```

**顺序很重要**：ZMQ PUB 不缓存，订阅者没连上前发的消息直接丢弃，而 teleop 启动到进控制
循环要 ~13 s——一段 6 s 的数据在它连上前就发完了。所以要**先 B 后 A**；想反过来就先开 A
并加 `--loop`。

脚本本身只做一件事：**把录制数据按 openpi-eval 的下发模式发到 :28701**。
`--debug-log` 的 JSONL（`ref_joint` / `joint_qpos` / `base_pos` …）就是对账接收端的入口。

延时/异常场景的旋钮：`--inference-ms`（推理停顿，默认 120）、`--inference-jitter-ms`、
`--drop-every`（丢帧）、`--final-stall-s`（VLA 挂掉、彻底停发）、`--loop`、
`--hold-s`（数据放完后保持末帧多久）、`--anchor-source integrated`（复现 openpi-eval
的积分路径）。

⚠️ **`[send]` 显示 ~27 Hz 而不是 30 Hz 是正常的**，不是丢帧：`control_hz` 是"非推理拍"的
节拍，而每 25 帧里有 1 拍要阻塞推理时间，那张单子算进帧率里 ——
`25 / (24×33.3ms + 120ms) ≈ 27.2 Hz`。`openpi-eval` 的循环结构完全相同（推理拍那次
`env.step()` 花掉 120ms+ 之后 `elapsed > target_dt`，不再补 sleep），所以**真机也到不了
30 Hz**。每次运行结束脚本会打印这个上限。它同时意味着**参考被放慢约 9%**
（27.2/30）——见第 7 节 TODO。

#### 用**真的** OpenHLM env 当发送端

上面的 `test_vla_motion_stream.py` 是按 openpi-eval 逻辑**复刻**的发送端（便于在本机无
OpenHLM 依赖时跑）。要验证**真代码**，可以直接用本目录的 `jaka_tabletop_env.py`：它的
`step()` 就是那条打包路径，喂录制的 33 维 action 即可（规范集的 `actions[:, 0:33]` 正好
就是它要的策略序 action）：

```python
from jaka_tabletop_env import JakaTabletopEnv
env = JakaTabletopEnv(mock=False, publish=True, control_hz=30)   # PUB :28701
env.step(actions33[i])                                            # 逐帧下发
```

2026-09-19 实测（199 帧 / 6.6 s，与 4.7 同样的两终端跑法，只是发送端换成真 env）：

| 检查项 | 真 `JakaTabletopEnv` | 复刻发送端 |
|---|---|---|
| 参考链路 ref→录制轨迹 | mean **0.0170** / p95 0.056 rad | 0.0162 / 0.048 rad |
| 闭环相关系数 | mean 0.70 | 0.73 |
| 稳定性 | base z 0.596 → 0.562，**站住** | 站住 |
| 接收端错误 | 0（无 `decode failed` / `dim mismatch`）| 0 |

两条路径的数字一致 ⇒ 复刻发送端是忠实的，真 env 的线格式也确实被接收端接受。
（观测通道同样验过：`get_observation()` 从 SIMPLE 的 :28711/:28712 取到
`(30,) float32` 状态与 `(224,224,3) uint8` 真图，非全零/非全黑。）

### 4.8 端到端验证记录（2026-09-19，本机仿真）

按 4.7 的两终端跑法，`data/simple/JakaTabletopPickTeleop-v0/level-0` episode 0
（199 帧 / 6.6 s）：

| 检查项 | 结果 |
|---|---|
| buffer 构造 / 协议解析 | `default standing posture via MuJoCo FK`；无 `binary decode failed` / `dim mismatch` |
| 控制循环 | 45.4–49.5 Hz，无异常掉帧 |
| **参考链路正确性** | teleop 实际用的 `ref_joint` vs 录制轨迹：mean **0.016 rad**、p95 0.048 rad |
| 闭环跟踪 | 25 个活动关节 ref-vs-robot 相关系数 mean **0.73**；关节误差 mean 0.60 rad |
| 稳定性 | base z 0.596 → min 0.555，**站住了** |
| 时序不变量 † | `newest - P` 下界恒 120.0 ms，锯齿上包络跨停顿不抬高；停顿冻结 100 ms（推理 120 ms）|
| 停顿一致性 † | 参考帧推进 0.47 帧/拍，理论 0.63——差额正好由 14 次停顿（每次 ~5 拍不推进）解释 |

† 这两行（以及 6.3 陷阱 1 的读数）是当时用脚本里一个**进程内监视器**（按 50 Hz 调
`get_obs()` 并统计）量的。那个监视器 2026-09-19 已从脚本中移除——脚本现在只做下发，
这些指标要复现得另写探针。其余各行都来自 `--debug-log` 的 JSONL，随时可复现。

同一套跑法在 `data/teleop_jaka_mf/20260915-145329`（旧格式数据集，163 帧 / 5.4 s）上
也通过：参考链路 mean 0.018 rad、相关系数 mean 0.65、关节误差 0.70 rad、站住。

⚠️ 闭环跟踪误差 0.6~0.7 rad 偏大，但**没有 pico 路径的基线可比**，无法区分是策略本身在
该场景的跟踪水平还是 VLA 链路引入的——要定位需用 `scripts/fake_pico_motion_pub.py`
回放同一条 motion 跑对照。

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

### 5.2 配置契约

**开关在 `data/jaka_mf/teleop_jaka_mf.yaml`（平铺键，不是 `motion:` 段）**：

```yaml
motion_backend: zmq_vla                        # 部署用；改回 zmq 即恢复 pico 遥操作
motion_zmq_connect: tcp://127.0.0.1:28701      # 经 prepare_policy_config 注入（见下）
```

链路是：`teleop yaml` → `JakaMFAgent(motion_backend=..., motion_zmq_connect=...)`
→ `BasePolicyArgs` → `BasePolicy.prepare_policy_config` 把它写进
`policy_config["motion"]` → `StateProcessor._init_motion_backend` 读取。

⚠️ **`prepare_policy_config` 只对 `"zmq"` 与 `"zmq_vla"` 注入**（4.1 的 ⑤⑥）；不注入时
下面这些键会退到 `state_processor` 里 `.get(..., 默认值)` 的硬编码值：`motion_zmq_connect`
→ `tcp://127.0.0.1:28701`、`motion_zmq_hwm` → `1`、`motion_dt_s` → `0.02`、
`motion_tolerance_s` → `0.04`。

注入的是这四个；另外两个 VLA 专属键**没有 CLI 入口**，只能用默认值，或直接写进
**policy yaml 的 `motion:` 段**（那才是 `StateProcessor` 真正读的字典）：

```yaml
motion:
  motion_nominal_frame_s: 0.0333     # 可选, 默认 1/30（= openpi-eval 的 control_hz）
  motion_gap_threshold_s: 0.05       # 可选, 默认 0.05
```

### 5.3 上游（openpi-eval 客户端）契约

**动作流（openpi-eval → SIMPLE，:28701）**

- 30 Hz 下发，二进制 v1，2 帧滑动窗口 `[i-1, i]`（frame_index 单调递增、跨动作块连续）；
- `joint_pos` 27 维 **SIM/JAKA 序**；`body_pos_w (N,3)` / `body_quat_w (N,4) wxyz` 为
  **anchor（waist_yaw_Link）** 位姿；
- 字段名/形状不符时新类打 warning 丢弃（不崩溃），但策略将拿不到参考——排查时先看该日志。

**观测流（SIMPLE → openpi-eval，:28711 / :28712）**

eval 端的 `JakaTabletopEnv.get_observation()` 直接订阅 SIMPLE 的
`JakaTeleopZmqPublisher`（`jaka_zmq_pub.py`）——**不是**另起一套，两边端口/编码已对齐
（2026-09-19；原先 eval 侧写的是占位的 28702/28703 + msgpack + topic 前缀，没有发布端）：

| 通道 | 端口 | 编码 | 内容 |
|---|---|---|---|
| state | **28711**（`state_zmq_bind`）| **裸 JSON**，无 topic 前缀 | `publish_t_ns` / `smplx_t_ns` / `paused` / `seq` / `joint_pos`(27, **JAKA/MuJoCo 序**) / `body_pos_w` / `body_quat_w` / `qpos` |
| camera | **28712**（`camera_zmq_bind`）| **裸字节**，无 topic 前缀 | `struct.pack("iii", w, h, c)` + HWC uint8 **RGB** |

两个 socket 都用 `SUBSCRIBE b""` + `CONFLATE=1`（发布端是 latest-only，慢消费者要丢旧帧
而不是排队）。eval 侧由 payload 现场派生两个量：

- `state[27:30]` = `roll, pitch` 取 **`body_quat_w[0]`（`base_link`）** 的 scipy 外旋 xyz；
  `yaw_vel` = 绕世界 z 的原始角速率（包裹差分 / `publish_t_ns` 间隔，首帧为 0）——
  与录制端 `OpenHLMRootVel` 同款算法；
- `state[0:27]` = `joint_pos` 按 `PERM_SIM_TO_POLICY` 重排到策略序（eval 侧做，SIMPLE 发的是 JAKA 序）。

⚠️ 改 `data/jaka_mf/teleop_jaka_mf.yaml` 的 `state_zmq_bind` / `camera_zmq_bind` 时，
eval 侧 `main.py` 的 `jaka_state_port` / `jaka_head_image_port` 要同步改。

### 5.4 上游代码位置

真正在用的 OpenHLM 侧代码（**要同步到部署机的那两份**）也放在本目录下：

| 文件 | 角色 |
|---|---|
| `docs/openhlm_vla/main.py` | eval 主循环，`--env {sonic_g1, jaka_tabletop_pick}` 二选一；Jaka 走 33 维 action / 30 维 state / 单目 |
| `docs/openhlm_vla/jaka_tabletop_env.py` | Jaka 部署环境：动作打包上 PUB :28701，观测从 :28711/:28712 取 |

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

   同一个假设还有一个**更隐蔽的失败面：调用频率必须等于 `1/dt_s`**，也就是控制循环
   真的跑满 `rl_rate`。`P` 每次**固定**推进 `dt_s`，而数据按真实帧率到达；循环一旦
   跑慢，`P` 就追不上 `newest`，`newest - P` **无界增长**（cleanup 基于 `P`，缓冲随之
   膨胀），参考越播越滞后，且**全程没有任何日志**。实测
   （2026-09-19 用 `scripts/test_vla_motion_stream.py` 实测；该脚本一度带过一个进程内
   监视器来量这些数字，现已移除，下面的读数是历史记录）：

   | 条件 | `newest - P` 稳态区间 |
   |---|---|
   | 47 Hz 循环 + `dt_s = 0.02` | 120 → **232 ms 且持续爬升** |
   | 47 Hz 循环 + `dt_s = 1/47` | 120 → 161 ms（正常锯齿） |

   而 `teleop_jaka_mf` 无头模式实测只跑到 **45.4–49.5 Hz**（从不到 50），即长期处在
   第一行。按 47/50 估算延迟每秒涨约 60 ms——跑 5 分钟就滞后 ~18 s。
   排查时盯 `newest - P` 锯齿的**上包络是否逐段抬高**（不抬高才是稳态）。
2. **clear() 不重置 `_last_frame_index`**：客户端帧号单调，重置会导致旧帧被当作新帧
   重复插入（时间戳乱序）。若要支持"客户端重启帧号归零"，需在 clear 时同步重置
   `_last_frame_index = -1`（并接受乱序插入的 bisect 路径）。
3. **JSON 路径同样走重锚定**：pico 遥操作若经 `zmq_vla` 后端运行，>50ms 的操作员暂停
   也会触发重锚定——这是期望行为，但别误以为 JSON 路径是"墙钟语义"。
4. **`_playback_time_ns` 无锁**：仅 policy 线程（get_obs）读写；任何其他线程访问
   必须通过 `playback_time_ns()` 并自担同步。
5. **`timestamps_ns` 是数据时间轴**：`MotionData.timestamps_ns` 已不是墙钟，任何把它
   与 `time.time_ns()` 混用的下游逻辑都会出错（当前下游无此用法）。
6. **⚠️ 两份数据集的 `actions[0:27]` 关节序不一样**，喂参考时**按名字判、别按位置猜**：

   | 数据集 | `actions[0:27]` 顺序 | 与什么一致 |
   |---|---|---|
   | `data/simple/JakaTabletopPickTeleop-v0/level-0`（**处理后的规范集，直接喂 VLA 训练**）| **OpenPI 策略序**（手臂在前）| **就是 VLA 的输出空间**；= `jaka_tabletop_env.POLICY_JOINT_NAMES` |
   | `data/teleop_jaka_mf/<run>/level-0`（`record_jaka_zmq` 直出）| **JAKA/MuJoCo 序**（腿在前）| = `simple.jaka_rl.config.JOINT_NAMES` |

   依据是各自 `meta/info.json` 的 `features.actions.names`。**搞错不会报任何错**——只是
   27 个参考关节整体错位，机器人会直接摔倒（实测：错误序下关节误差 8.58 rad、base z
   掉到 0.128 摔倒；修正后 0.60 rad、站住）。`test_vla_motion_stream.py` 因此按
   `meta/info.json` 按名字判定并在启动时打印 `[data] actions 关节序: …`；
   **接别的数据源时先看这一行**。
   （注：`config.POLICY_JOINT_NAMES` 是 IsaacLab/npz 序（左右腿交替），与上面两种**都不同**，
   只是重名——它对应 `NPZ_JOINT_NAMES`，不要拿来当"策略序"用。）

7. **规范集的 33 维 action 就是 `actions[:, 0:33]` 原样**（承陷阱 6）。既然那份数据是
   VLA 的训练数据，VLA 的输出空间就等于它——所以模拟 VLA 时**不做任何重排**：

   ```
   action[0:27] = actions[0:27]   # 策略序关节，原样
   action[27:30]= actions[27:30]  # roll(绝对) / pitch(绝对) / yaw_vel
   action[30:33]= actions[30:33]  # anchor 机体系线速度
   ```

   **`actions[33:40]`（anchor 世界位姿）是 DEBUG 列，VLA 不输出**——这正是
   `--anchor-source` 那个旋钮的由来：

   * `recorded`（脚本默认）直接发 `[33:40]` 的真值 ⇒ 参考轨迹 === 录制轨迹，
     但用了一条 **VLA 根本产生不了**的通道，属于理想化参考；
   * `integrated` 只用 VLA 真会输出的 `[30:33]`，按
      `JakaTabletopEnv.step()` 那样积分出 anchor 位姿 ⇒ **与真实部署逐位一致**，
     代价是引入实测 **mean 5.6 cm / max 8.0 cm** 的 anchor 漂移（见第 7 节 TODO）。

   想要"VLA 真会看到什么"就用 `--anchor-source integrated`；想先排除漂移这个变量、
   单纯验证缓冲时序就用默认的 `recorded`。

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
- [ ] **控制循环跑不满 `rl_rate` 时播放延迟无界增长**（机理与实测见 6.3 陷阱 1）。
      `P` 每次固定推进 `dt_s`，而 teleop 实测只到 45.4–49.5 Hz（配置宣称 50），
      于是 `newest - P` 持续爬升、缓冲同步膨胀，**无任何日志**。
      可选修法：把 `get_obs()` 里的 `self._playback_time_ns += self._dt_ns` 换成
      "按真实墙钟间隔推进"——记下上次调用时刻，`step = min(max(now - last, dt/4), 4*dt)`；
      停顿期仍由 cap 钉住，**冻结语义不变**，且顺带让 `get_obs()` 对重复调用免疫
      （即 6.3 陷阱 1 的前半段也一并消掉）。**代价是与 C++
      `RealtimeMotionBuffer_vla.cpp` 的语义分歧**，故 2026-09-19 决定先记录、暂不改。
- [ ] **参考被放慢约 9%（属上游客户端，非本类）**：`openpi-eval` 每 25 拍阻塞一次推理
      （约 120ms），那一次 `env.step()` 只发一帧却吃掉一整个节拍且不再补 sleep，于是实际
      下发速率是 `25/(24×33.3ms + 120ms) ≈ 27.2 Hz` 而非 30 Hz。接收端按**到达速率**播放
      （`P` 被 `newest - delay` 钳住），所以**参考轨迹按 0.91× 速度播放**——整条 VLA 链路
      执行的动作比训练数据的时序慢约 10%。这是 openpi-eval 的循环结构决定的，不是缓冲的
      问题；要真正 30 Hz 需要缩短推理时间，或把"节拍"与"推理"解耦。实测见 4.7。

- [ ] **openpi-eval 的 anchor 积分路径有数厘米漂移**（属上游客户端，非本类）：
      若按 `JakaTabletopEnv.step()` 原样从 `action[30:33]` 的机体系线速度积分 anchor
      世界位姿，相对录制真值的偏差实测 **mean 5.6 cm / max 8.0 cm**（线速度按 pico
      发布周期算出、却按 1/30 积分，且被 clip 到 ±2.0，实测峰值 2.07 m/s 确实触顶）。
      用录制数据做参考时建议直接发真值 anchor 位姿
      （`test_vla_motion_stream.py --anchor-source recorded`，也是该脚本默认）。

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
| `src/SIMPLE-jaka-main/src/simple/jaka_rl/base_policy.py` | **本次修改**：`Literal` 加 `zmq_vla` + 注入分支放行（打通链路，见 4.1 ⑤⑥） |
| `src/SIMPLE-jaka-main/data/jaka_mf/teleop_jaka_mf.yaml` | **本次修改**：`motion_backend: zmq_vla` |
| `src/SIMPLE-jaka-main/scripts/test_vla_motion_stream.py` | **本次新增**：仿真 openpi-eval 下发（录制 parquet 回放 + 停顿/丢帧/挂掉注入）。默认数据源 `data/simple/JakaTabletopPickTeleop-v0/level-0`；用法见 4.7 |
| `data/simple/JakaTabletopPickTeleop-v0/level-0` | 47 集规范数据集（VLA 训练用），`actions` 为 **OpenPI 策略序**（见 6.3 陷阱 6）|
| `src/GR00T-WholeBodyControl4OpenHLM/gear_sonic_deploy/.../streamed_motion_merger.hpp` 等 | sonic 端设计哲学来源（第 2 节） |
| `src/openpi4OpenHLM/...` / `openpi-eval/` | 上游 VLA 服务器与 30Hz 客户端（契约见 5.3） |

---

*文档生成于 OpenHLM 工作区; 若 C++/Python 两端行为出现分歧, 以 `RealtimeMotionBuffer_vla.cpp` 的语义为基准对齐。*
