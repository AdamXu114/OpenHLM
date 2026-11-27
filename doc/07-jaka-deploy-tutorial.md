# 07 Jaka 部署教程（最终部署命令）

**部署机：新机器，原生运行（不在 docker 里），无需 GPU 预约。** 仓库在 `~/workspace/OpenHLM`。
策略服务端与部署客户端在**同一台机器**上跑，因此不需要 `--remote-host` 指向别的机器。

`main.py` 向 `serve_policy.py` 的 WebSocket 8000 请求动作，在部署 PC 的 28701 发布轨迹。机器人进入 SuperZmq 后订阅轨迹，并在 28711 发布实测 state。头部相机另在 28712 发布 RGB。

> 本文是**照着敲就能跑**的最终版；原理、anchor 约定、参数表见 [04-deployment.md](04-deployment.md)。
> 只有改了发布协议才需要跑 [04-deployment.md §4.4](04-deployment.md) 的协议对拍测试。

---

## 0. 前置检查

无需 GPU 预约（那是训练机 / 容器 `xujinfan_dev` 的规矩，见 [05-troubleshooting.md](05-troubleshooting.md)）。

```bash
# 确认仓库在
ls ~/workspace/OpenHLM/src/openpi4OpenHLM/scripts/serve_policy.py

# 检查本机的 base_link checkpoint
ls -l ~/workspace/OpenHLM/src/openpi4OpenHLM/checkpoints/jaka_tabletop_pick/jaka_pick_real_v2/30000/model.safetensors
```

需要的文件（都已就位，无需改动）：


| 作用           | 文件                                                                                        |
| ------------ | ----------------------------------------------------------------------------------------- |
| 策略服务端        | `~/workspace/OpenHLM/src/openpi4OpenHLM/scripts/serve_policy.py`                          |
| 部署客户端        | `~/workspace/OpenHLM/src/GR00T-WholeBodyControl4OpenHLM/openpi-eval/main.py`              |
| Jaka 环境（发动作） | `~/workspace/OpenHLM/src/GR00T-WholeBodyControl4OpenHLM/openpi-eval/jaka_tabletop_env.py` |


Python 解释器统一用 openpi 的 venv：
`~/workspace/OpenHLM/src/openpi4OpenHLM/.venv/bin/python`

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
cd ~/workspace/OpenHLM/src/openpi4OpenHLM
./.venv/bin/python scripts/serve_policy.py --env JAKA --num-steps 10 \
  policy:checkpoint --policy.config jaka_tabletop_pick \
  --policy.dir checkpoints/jaka_tabletop_pick/jaka_pick_real_v2/30000
```

等到出现监听日志（默认 `0.0.0.0:8000`）再进下一步。

> `--num-steps` 设置服务端每次生成 action chunk 的采样步数。它会影响推理耗时，默认值为 10。

---

## 3. 窗口 2：部署客户端（推理 + 发动作）

```bash
cd ~/workspace/OpenHLM/src/GR00T-WholeBodyControl4OpenHLM/openpi-eval
export PYTHONPATH=~/workspace/OpenHLM/src/GR00T-WholeBodyControl4OpenHLM

~/workspace/OpenHLM/src/openpi4OpenHLM/.venv/bin/python main.py \
  --env jaka_tabletop_pick \
  --remote-host 127.0.0.1 --remote-port 8000 \
  --jaka-motion-zmq-port 28701 \
  --jaka-state-address "${ROBOT_IP:?set robot IP}" --jaka-head-image-address "$ROBOT_IP" \
  --instruction "pick up the box" \
  --max-steps 100000
```

同机部署，`--remote-host 127.0.0.1` 即可（`--remote-port` 与服务端 `--port` 一致，默认都是 8000）。

`PYTHONPATH` 中的 `GR00T-WholeBodyControl4OpenHLM` 提供 `gear_sonic`。客户端等待终端输入 `s` 回车后开始发送参考。远程 SSH 无显示器时加 `--no-opencv-visualize`。

常用可选项：


| 参数                                           | 说明                                         |
| -------------------------------------------- | ------------------------------------------ |
| `--instruction "<文本>"`                       | 语言指令，JAKA 默认为 `pick up the box`                             |
| `--max-steps <n>`                            | 跑多少步（30 Hz，1000 步 ≈ 33 s）                  |
| `--control-hz 30` / `--open-loop-horizon 25` | 控制频率 / 开环步数（策略 chunk 是 50，最多开环 25）         |
| `--save-video --video-save-dir <dir>`        | 存 MP4（单路 head 图，480×480）                   |
| `--jaka-initial-anchor-pos X Y Z`（默认 `0 0 0.604`）   | anchor 初始位置，**z 有意义**（`root_z_mf` 是绝对世界 z） |


跑起来后应该看到：连上策略服务端的日志、默认每执行 25 步重新推理一次、
30 Hz 往 `tcp://*:28701` 发协议 v1 二进制帧。

### 跑多久 / 为什么会自己停

30 Hz 下执行 1000 步约需 33 秒，再加上各次同步推理的停顿。跑满后终端会询问 `Do one more eval? (enter y or n)`。输入 `y` 开始下一轮，其他输入结束并关闭 JAKA 客户端。如果停顿超过控制器的参考超时时间，先核对新的起始姿态，在控制器退出并重入 SuperZmq 后再发新轨迹。

停止：客户端窗口 `Ctrl+C`（结束本轮参考发布）→ 服务端窗口 `Ctrl+C`。

---

## 4. 下游接收方

在控制器实际读取的 `KpiConfig.yaml` 中，将 `jakaminimimic_superzmq.motion_backend` 设为 `vla_pose_v1`，`zmq_connect` 指向部署 PC 的 `tcp://<PC-IP>:28701`。启动相机并让机器人进入 SuperZmq，客户端收到有效 state 与 RGB 后才开始推理。相机发布器使用 RGB。模型权重由控制器的 `policy_path` 选择，复现 180k EMA 结果须选择相应的 ONNX 和配套部署参数。

每个动作包注明 `anchor_body=base_link`、JAKA 关节顺序和发布器 `stream_id`。相邻滑窗通常重复一帧。若启用控制器的 policy diagnostic，`latest_frame_index` 和 `valid_receive_age_ns` 来自同一次窗口采样，首次收到有效包前均为 −1。`stream_hold_reason` 说明参考被锁定的原因。

---

## 5. 端口速查


| 端口    | 方向        | 说明                                     |
| ----- | --------- | -------------------------------------- |
| 8000  | 客户端 → 服务端 | websocket 策略推理                         |
| 28701 | 客户端 → 下游  | 协议 v1 二进制动作帧（`--jaka-motion-zmq-port`） |
| 28711 | 观测 → 客户端  | 实测关节与 base 四元数（StatePublisher，JSON）        |
| 28712 | 观测 → 客户端  | 头部 RGB（cam_pub，12 字节图像头 + uint8 像素）         |


---

## 6. 坑（都会静默失败，务必看）

1. 服务端首次推理可能需要预热。如遇 WebSocket 超时，先检查服务端是否已完成加载，再重新连接。
2. 同一发布器的 reset 保留编号与积分 pose。新实例会更换 `stream_id` 并从新提供的 base 初值开始。
3. 默认每执行 25 步同步推理下一块动作。停顿期间保留实际执行的帧序号。
4. 实机模式等待新鲜的 state 和 head。报 `No fresh ...` 时，按提示检查该发布源地址、数据格式与发送进度。
5. **视频宽度。** jaka 只有 1 路 head 图，`VideoWriter` 宽度按单图算；改 `num_cameras` 会让写帧静默失败。

