"""Measure how long the policy server actually takes -- first inference included.

Why this exists: every other client (``main.py``, ``jaka_policy_dataset_eval.py``) talks to
``serve_policy.py`` through openpi's ``WebsocketClientPolicy``, which hard-codes a **10 s** handshake
deadline and retries **only** on ``ConnectionRefusedError``. A server still compiling its first
inference cannot serve the HTTP upgrade at all, so every attempt dies with ``TimeoutError`` (no
handshake response) or ``InvalidMessage`` (connection closed) -- identical symptoms to a server that
is genuinely broken. Neither client can distinguish the two, because both give up long before the
first inference finishes.

This probe connects with a generous timeout, keepalive OFF, and a dummy observation, printing the
elapsed time of each phase. One run separates the cases:

* handshake OK + a slow first infer, then fast repeats -> the server is healthy, the warmup is just
  long. ``jaka_policy_dataset_eval.py`` absorbs this automatically (its warmup phase), and any other
  client should simply be pointed at a server that has already served one request.
* handshake never completes even with a huge timeout -> the server's event loop really is stuck;
  the server-side log (or a stack dump) is the next thing to look at.
* a server error string comes back -> the server raised during inference and told us why.

Each phase is deadline-bounded and ticks every 15 s, so a server that accepts the handshake and then
never answers shows up as a count-up ("still waiting on infer 0 (45s)") instead of a silent hang.

No dataset, no robot, no ZMQ -- just a dummy request. Also prints any proxy environment variables:
the websockets client honors ``ws_proxy``/``all_proxy``, so an exported proxy can silently redirect
a connection that looks like it is going to 127.0.0.1.

Run with the server already up in another window::

    cd ~/Workspace/OpenHLM/src/GR00T-WholeBodyControl4OpenHLM/openpi-eval
    export PYTHONPATH=~/Workspace/OpenHLM/src/GR00T-WholeBodyControl4OpenHLM
    ~/Workspace/OpenHLM/src/openpi4OpenHLM/.venv/bin/python jaka_policy_server_probe.py

The dummy observation is a black image and zeros for state, so the action values are meaningless --
only the timings and the error strings matter.
"""

# flake8: noqa: E402
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import dataclasses
import time

import numpy as np
import tyro
import websockets.sync.client
from openpi_client import msgpack_numpy

DEPLOY_PROMPT = "Pick up the purple soft finger on the table and place it on the mouse pad."


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 8000
    timeout_s: float = 600.0   # per phase; the first infer can compile for minutes
    repeats: int = 3           # >1 shows steady-state rate after the first one


def recv_bounded(conn, timeout_s: float, label: str, tick_s: float = 15.0):
    """``recv`` bounded by a deadline, with a heartbeat -- this is the phase being measured.

    A timed-out ``recv`` does not discard the message (websockets puts the frames it already read
    back on the queue), so ticking is safe and the next call still returns it. Kept local instead of
    imported from ``jaka_policy_dataset_eval`` so this probe keeps running even when the rest of the
    stack will not import -- which is a large part of why it exists.
    """
    t0 = time.time()
    while True:
        left = timeout_s - (time.time() - t0)
        if left <= 0:
            raise TimeoutError(f"no reply for {label} within {timeout_s:.0f}s")
        try:
            return conn.recv(min(tick_s, left))
        except TimeoutError:
            print(f"      ... still waiting on {label} ({time.time() - t0:.0f}s)", flush=True)


def main(args: Args):
    proxy_env = {k: v for k, v in os.environ.items() if "proxy" in k.lower()}
    print("proxy env : " + (str(proxy_env) if proxy_env else "(none)"))

    uri = f"ws://{args.host}:{args.port}"
    packer = msgpack_numpy.Packer()
    request = {
        "head_image_left": np.zeros((224, 224, 3), np.uint8),   # black image
        "state": np.zeros(30, np.float32),
        "prompt": DEPLOY_PROMPT,
    }

    t0 = time.time()
    print(f"connecting: {uri}   handshake timeout {args.timeout_s:.0f}s, keepalive off")
    with websockets.sync.client.connect(
        uri, compression=None, max_size=None,
        open_timeout=args.timeout_s, ping_interval=None,   # no ping: it must NOT be killed mid-compile
    ) as conn:
        print(f"  handshake         {time.time() - t0:8.1f}s")
        metadata = recv_bounded(conn, 30.0, "server metadata")
        print(f"  server metadata   {time.time() - t0:8.1f}s   ({len(metadata)} bytes)")

        for i in range(max(args.repeats, 1)):
            t = time.time()
            conn.send(packer.pack(request))
            response = recv_bounded(conn, args.timeout_s, f"infer {i}")
            dt = time.time() - t
            if isinstance(response, str):
                # The server reports exceptions during infer as a text frame.
                print(f"  infer {i}: SERVER ERROR after {dt:.1f}s\n    {response[:2000]}")
                break
            out = msgpack_numpy.unpackb(response)
            if "actions" not in out:
                print(f"  infer {i}: {dt * 1000:8.1f} ms   no 'actions' key; got {sorted(out)}")
                continue
            act = np.asarray(out["actions"])
            print(f"  infer {i}: {dt * 1000:8.1f} ms   actions {act.shape} {act.dtype}   "
                  f"absmax {np.abs(act).max():.3f}")

    print(f"total {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main(tyro.cli(Args))
