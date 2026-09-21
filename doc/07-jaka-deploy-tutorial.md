# 07 Jaka 部署教程（最终部署命令）

**部署机：新机器，原生运行（不在 docker 里），无需 GPU 预约。** 仓库在 `~/Workspace/OpenHLM`。
策略服务端与部署客户端在**同一台机器**上跑，因此不需要 `--remote-host` 指向别的机器。

链路：`serve_policy.py` ──websocket :8000──► `main.py`(`--env jaka_tabletop_pick`) ──ZMQ :28701──►
`RealtimeMotionBufferVla`（下游接收方，常驻，不用动）。

> 本文是**照着敲就能跑**的最终版；原理、anchor 约定、参数表见 [04-deployment.md](04-deployment.md)。
> 只有改了发布协议才需要跑 [04-deployment.md §4.4](04-deployment.md) 的协议对拍测试。

---

## 0. 前置检查

无需 GPU 预约（那是训练机 / 容器 `xujinfan_dev` 的规矩，见 [05-troubleshooting.md](05-troubleshooting.md)）。

```bash
# 确认仓库在
ls ~/Workspace/OpenHLM/src/openpi4OpenHLM/scripts/serve_policy.py

# 确认 checkpoint 在
ls ~/Workspace/OpenHLM/src/openpi4OpenHLM/checkpoints/jaka_tabletop_pick/jaka_tabletop_pick_v1/
# 期望看到 5000 10000 15000 20000 25000 30000；30000 是默认部署步数

# 若是新机器首次拉 ckpt：从 HF 公开仓库 XuJinfan/jaka-tabletop-pick-v1 取 30000.tar，
# 在 checkpoints/jaka_tabletop_pick/jaka_tabletop_pick_v1/ 下解包（tar 顶层就是 30000/，且已含 assets/）
ls -l ~/Workspace/OpenHLM/src/openpi4OpenHLM/checkpoints/jaka_tabletop_pick/jaka_tabletop_pick_v1/30000/model.safetensors
# 期望 7,473,099,660 字节
```

需要的文件（都已就位，无需改动）：


| 作用           | 文件                                                                                        |
| ------------ | ----------------------------------------------------------------------------------------- |
| 策略服务端        | `~/Workspace/OpenHLM/src/openpi4OpenHLM/scripts/serve_policy.py`                          |
| 部署客户端        | `~/Workspace/OpenHLM/src/GR00T-WholeBodyControl4OpenHLM/openpi-eval/main.py`              |
| Jaka 环境（发动作） | `~/Workspace/OpenHLM/src/GR00T-WholeBodyControl4OpenHLM/openpi-eval/jaka_tabletop_env.py` |


Python 解释器统一用 openpi 的 venv：
`~/Workspace/OpenHLM/src/openpi4OpenHLM/.venv/bin/python`

部署**不需要** `pi05_base` / `pi05_base_pytorch` 权重，也不需要 LeRobot 数据集——
服务端只读 `<ckpt>/model.safetensors` 和 `<ckpt>/assets/<asset_id>/norm_stats.json`。
启动时打印的 `/workspace/openpi-assets/...` 那条配置是惰性的，可以无视。
首次启动会从 GCS 下载 `paligemma_tokenizer.model`（4 MB），**需要联网一次**。

---

## 1. 起 tmux（两个进程都要保活）

```bash
tmux new -s jakadep
```

`Ctrl-b` 松手再按 `c` 开第二个窗口（`Ctrl-b` + `n`/`p` 切窗口）。脱离：`Ctrl-b` 松手再按 `d`。
**不要按 `Ctrl-d` / `exit`。** 本地桌面直接开两个终端也可以，只是 SSH 断线会一起断。

---

## 2. 窗口 1：策略服务端

```bash
cd ~/Workspace/OpenHLM/src/openpi4OpenHLM
./.venv/bin/python scripts/serve_policy.py --env JAKA --num-steps 10
```

`--env JAKA` 等价于 `--env jaka_tabletop_pick`，checkpoint 走
`serve_policy.py` 里 `DEFAULT_CHECKPOINT[EnvMode.JAKA]` 的默认值
（`checkpoints/jaka_tabletop_pick/jaka_tabletop_pick_v1/30000`）。

**换 checkpoint 步数**时显式指定，别改代码：

```bash
./.venv/bin/python scripts/serve_policy.py --env JAKA --num-steps 10 \
  policy:checkpoint \
  --policy.config=jaka_tabletop_pick \
  --policy.dir=checkpoints/jaka_tabletop_pick/jaka_tabletop_pick_v1/30000
```

等到出现监听日志（默认 `0.0.0.0:8000`）再进下一步。

> `--num-steps 10` 与服务端自检有关，**不影响客户端推理**，保持默认即可。

---

## 3. 窗口 2：部署客户端（推理 + 发动作）

```bash
cd ~/Workspace/OpenHLM/src/GR00T-WholeBodyControl4OpenHLM/openpi-eval
export PYTHONPATH=~/Workspace/OpenHLM/src/GR00T-WholeBodyControl4OpenHLM

~/Workspace/OpenHLM/src/openpi4OpenHLM/.venv/bin/python main.py \
  --env jaka_tabletop_pick \
  --remote-host 127.0.0.1 --remote-port 8000 \
  --jaka-motion-zmq-port 28701 \
  --max-steps 100000 \
  <<< $'s\n'
```

同机部署，`--remote-host 127.0.0.1` 即可（`--remote-port` 与服务端 `--port` 一致，默认都是 8000）。

两个必须的要素：


| 要素                                                                  | 原因                                                                                                          |
| ------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| `PYTHONPATH=~/Workspace/OpenHLM/src/GR00T-WholeBodyControl4OpenHLM` | `sonic_g1_env` 会 import `gear_sonic`，openpi 的 venv 里没有；不加这句 main.py 在 import 阶段就死（原始生产环境用的是 `.venv_teleop`） |
| 喂 stdin（`<<< $'s\n'`）                                               | 主循环阻塞在 `input()` 等 's'，喂一个 `s` 才开始发帧                                                                        |


> SSH 远程连这台机器时**再加** `--no-opencv-visualize`：无 X display 时 cv2 在 `imshow` 上直接 abort，
> 而该参数**默认是 True**。本地桌面有显示则不用加。想手动控制就把 `<<< $'s\n'` 去掉、直接在终端敲 `s` 回车。

常用可选项：


| 参数                                           | 说明                                         |
| -------------------------------------------- | ------------------------------------------ |
| `--instruction "<文本>"`                       | 语言指令，默认是紫色软指那条                             |
| `--max-steps <n>`                            | 跑多少步（30 Hz，1000 步 ≈ 33 s）                  |
| `--control-hz 30` / `--open-loop-horizon 25` | 控制频率 / 开环步数（策略 chunk 是 50，最多开环 25）         |
| `--save-video --video-save-dir <dir>`        | 存 MP4（单路 head 图，480×480）                   |
| `--jaka-initial-anchor-pos "0.0 0.0 0.83"`   | anchor 初始位置，**z 有意义**（`root_z_mf` 是绝对世界 z） |


跑起来后应该看到：连上策略服务端的日志、每个 chunk（50 步）一次约 70–80 ms 的推理、
30 Hz 往 `tcp://*:28701` 发协议 v1 二进制帧。

### 跑多久 / 为什么会自己停

`--max-steps 1000` @ 30 Hz ≈ **35 秒**就跑满一轮，这不是崩溃。跑满后代码会问
`Do one more eval? (enter y or n)`；用 `<<< $'s\n'` 启动时 stdin 里只剩一个空行
（bash 会给 here-string 再补一个换行，所以实际是 "s" + 空行两行），
空行 ≠ `y` → 走正常退出路径（重置 → `Final reset complete.`）。要跑久一点就调大：

```bash
--max-steps 100000        # ≈ 55 分钟
```

想每轮结束手动决定是否继续，就用 `<<<` 的形式、在终端里交互敲 `s`，结束时敲 `y`。

停止：客户端窗口 `Ctrl+C`（回初始位姿）→ 服务端窗口 `Ctrl+C`。

---

## 4. 下游接收方

不用重启、不用改配置，确保 `RealtimeMotionBufferVla`（`data/jaka_mf/teleop_jaka_mf.yaml` 里
`motion_backend: zmq_vla`）订阅 `tcp://127.0.0.1:28701` 即可。健康检查看接收端日志：

- 不应出现 `binary decode failed` / `joint dim mismatch` / `body dim mismatch`
- `stale_ms()` 在 chunk 边界涨到 ~120 ms、chunk 内部回落
- `latest_timestamp_ns - playback_time_ns ≈ 120 ms` 且**不随停顿漂移**

---

## 5. 端口速查


| 端口    | 方向        | 说明                                     |
| ----- | --------- | -------------------------------------- |
| 8000  | 客户端 → 服务端 | websocket 策略推理                         |
| 28701 | 客户端 → 下游  | 协议 v1 二进制动作帧（`--jaka-motion-zmq-port`） |
| 28711 | 观测 → 客户端  | `jaka_state`（占位通道，**仓库内暂无发布方**）        |
| 28712 | 观测 → 客户端  | `jaka_head`（占位通道，**仓库内暂无发布方**）         |


---

## 6. 坑（都会静默失败，务必看）

1. **首次连接必失败。** 服务端在 infer 期间阻塞事件循环，每次**重启服务端后第一个**客户端会撞上
  20 s ping 超时。**重连一次即可**，不是配置问题。
2. **不要靠重启客户端来"复位"。** `frame_index` 一旦回退会被接收端静默去重，参考轨迹**永久冻结**。
  `JakaTabletopEnv.reset()` 已刻意不复位它，别改。
   **注意 `Ctrl+C` 结束客户端进程后重新启动，`frame_index` 会从 0 重新开始**——
   这时必须**同时重启下游接收方**（它的去重基线跨自己的 `clear()` 保留），
   否则新发的帧全被丢掉、机器人不动。因此宁可开始就把 `--max-steps` 开大，别中途重启。
3. **推理停顿不能被帧填满。** 每块 25 步开环 ≈120 ms 无数据是**期望行为**，接收端按
  "间隙 > 50 ms 即停顿" 重锚时间轴。**不要**加 filler 帧。
4. **观测通道目前是占位。** 28711/28712 没有发布方时，`get_observation()` 退化为全零 state + 黑图——
  链路照跑、动作照发，但策略看不到画面。真机部署需要机器人侧提供这两路 msgpack 发布。
5. **视频宽度。** jaka 只有 1 路 head 图，`VideoWriter` 宽度按单图算；改 `num_cameras` 会让写帧静默失败。

