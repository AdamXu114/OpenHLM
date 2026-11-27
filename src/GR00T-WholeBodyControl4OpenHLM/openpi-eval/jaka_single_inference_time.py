"""Measure repeated inference requests to a running JAKA policy server.

Sends a synthetic observation (black 224x224 RGB image, zero 30-D state) through
the deployment websocket/msgpack protocol. It does not connect to or command the
robot. Requests continue until Ctrl+C; --warmup excludes the first compile call.

Run from openpi-eval with the OpenPI venv:
    ../../openpi4OpenHLM/.venv/bin/python jaka_single_inference_time.py
"""

import dataclasses
import time

import numpy as np
import tyro
import websockets.sync.client
from openpi_client import msgpack_numpy


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 8000
    timeout_s: float = 600.0
    prompt: str = "pick up the box"
    warmup: bool = False  # send one unmeasured request first, useful after server restart
    interval_s: float = 0.5  # pause between requests; set 0 to run as fast as possible


def recv_with_progress(conn, timeout_s: float, label: str) -> bytes | str:
    start = time.perf_counter()
    while True:
        remaining = timeout_s - (time.perf_counter() - start)
        if remaining <= 0:
            raise TimeoutError(f"timed out waiting for {label} after {timeout_s:.0f}s")
        try:
            return conn.recv(min(15.0, remaining))
        except TimeoutError:
            print(f"  waiting for {label}: {time.perf_counter() - start:.0f}s", flush=True)


def request_once(conn, packed_request: bytes, timeout_s: float, label: str):
    start = time.perf_counter()
    conn.send(packed_request)
    response = recv_with_progress(conn, timeout_s, label)
    elapsed = time.perf_counter() - start
    if isinstance(response, str):
        raise RuntimeError(f"policy server returned an error: {response[:2000]}")
    result = msgpack_numpy.unpackb(response)
    if "actions" not in result:
        raise RuntimeError(f"policy response has no 'actions'; got keys {sorted(result)}")
    return elapsed, np.asarray(result["actions"])


def main(args: Args) -> None:
    if args.interval_s < 0:
        raise ValueError("--interval-s must be >= 0")

    uri = f"ws://{args.host}:{args.port}"
    request = {
        "head_image_left": np.zeros((224, 224, 3), dtype=np.uint8),
        "state": np.zeros(30, dtype=np.float32),
        "prompt": args.prompt,
    }
    packed = msgpack_numpy.Packer().pack(request)
    latencies_ms = []

    print(f"Connecting to {uri} (timeout {args.timeout_s:.0f}s)...", flush=True)
    start = time.perf_counter()
    try:
        with websockets.sync.client.connect(
            uri,
            compression=None,
            max_size=None,
            open_timeout=args.timeout_s,
            ping_interval=None,
        ) as conn:
            print(f"WebSocket connected in {time.perf_counter() - start:.3f}s", flush=True)
            metadata = recv_with_progress(conn, args.timeout_s, "server metadata")
            print(f"Server metadata received ({len(metadata)} bytes)", flush=True)

            if args.warmup:
                warmup_s, _ = request_once(conn, packed, args.timeout_s, "warm-up inference")
                print(f"Warm-up (not included): {warmup_s * 1000:.1f} ms", flush=True)

            print(f"Measuring continuously; press Ctrl+C to stop (interval={args.interval_s:g}s).")
            while True:
                index = len(latencies_ms) + 1
                elapsed, actions = request_once(conn, packed, args.timeout_s, f"inference #{index}")
                latency_ms = elapsed * 1000.0
                latencies_ms.append(latency_ms)
                print(f"#{index:05d}  {latency_ms:8.1f} ms  actions={actions.shape}", flush=True)
                if args.interval_s:
                    time.sleep(args.interval_s)
    except KeyboardInterrupt:
        print("\nStopped by Ctrl+C.", flush=True)

    if latencies_ms:
        values = np.asarray(latencies_ms, dtype=np.float64)
        print(
            f"Summary ({len(values)} measured requests): "
            f"mean={values.mean():.1f} ms, median={np.median(values):.1f} ms, "
            f"p95={np.percentile(values, 95):.1f} ms, "
            f"min={values.min():.1f} ms, max={values.max():.1f} ms"
        )
    else:
        print("No measured inference completed.")


if __name__ == "__main__":
    main(tyro.cli(Args))
