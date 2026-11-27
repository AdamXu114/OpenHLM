"""Tiny periodic rate/gap probe for the deployment loops (main.py / jaka_openloop_dataset.py).

Every ``interval_s`` seconds it prints one line per channel: achieved rate (count / window)
and the maximum gap between consecutive events. No per-event durations are reported.
Channels fed by the env's own receive counters (image / state) are added with
:meth:`note_external`.

Usage::

    stats = PeriodicStats(interval_s=3.0)
    stats.record("infer")        # one event
    stats.record("publish")
    snap = env.rx_snapshot(reset=True)
    stats.note_external("image", snap["image"]["count"], snap["image"]["max_gap_s"])
    stats.maybe_print()          # prints at most once per interval

    stats.flush()                # print the tail window before exiting
"""

from __future__ import annotations

import time


class PeriodicStats:
    def __init__(self, interval_s: float = 3.0) -> None:
        self.interval_s = float(interval_s)
        self._t0 = time.monotonic()
        self._last_print = self._t0
        self._channels: dict[str, dict] = {}   # name -> count / last / max_gap
        self._external: dict[str, dict] = {}   # name -> count / max_gap

    # ------------------------------------------------------------------ feed
    def record(self, name: str) -> None:
        """Count one event of ``name`` and track the gap since the previous one."""
        now = time.monotonic()
        ch = self._channels.setdefault(name, {"count": 0, "last": None, "max_gap_s": 0.0})
        if ch["last"] is not None:
            ch["max_gap_s"] = max(ch["max_gap_s"], now - ch["last"])
        ch["last"] = now
        ch["count"] += 1

    def note_external(self, name: str, count: int, max_gap_s: float) -> None:
        ext = self._external.setdefault(name, {"count": 0, "max_gap_s": 0.0})
        ext["count"] += int(count)
        ext["max_gap_s"] = max(ext["max_gap_s"], float(max_gap_s))

    # ----------------------------------------------------------------- print
    def _format(self, window_s: float) -> str:
        parts = []
        for name, ch in self._channels.items():
            if ch["count"] == 0:
                continue
            parts.append(
                f"{name} {ch['count'] / window_s:5.1f}/s (max gap {ch['max_gap_s'] * 1000.0:5.0f} ms)"
            )
        for name, ext in self._external.items():
            if ext["count"] == 0:
                continue
            parts.append(
                f"{name} {ext['count'] / window_s:5.1f}/s (max gap {ext['max_gap_s'] * 1000.0:5.0f} ms)"
            )
        return " | ".join(parts)

    def _reset_window(self) -> None:
        for ch in self._channels.values():
            ch.update(count=0, max_gap_s=0.0)
        for ext in self._external.values():
            ext.update(count=0, max_gap_s=0.0)

    def maybe_print(self, force: bool = False) -> None:
        now = time.monotonic()
        window = now - self._last_print
        if not force and window < self.interval_s:
            return
        text = self._format(window)
        if text:
            print(f"[stats {now - self._t0:7.1f}s] {text}", flush=True)
        self._reset_window()
        self._last_print = now

    def flush(self) -> None:
        self.maybe_print(force=True)
