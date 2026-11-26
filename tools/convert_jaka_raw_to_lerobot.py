"""Convert Khan raw JSON/MP4 episodes to the existing OpenHLM Jaka contract.

Default: forward action velocities on the exported fixed-fps row timeline;
state observations retain their recorded causal values. Raw poses and camera
rows are not resampled, synchronized, or shifted. Actual recording cadence is
reported separately: nominal-time differentiation is a playback contract, not
a claim of hardware synchronization. The output is already arms-first; do not
reorder it with the old merge script. LeRobot is imported after validation.

The video-reference -> decoded RGB -> LeRobot image-feature path follows
OpenHLM's examples/unitree_g1/convert_g1_data_to_lerobot.py (upstream commit
4a8d4cd0ecd8313bb8dbb10d2afac2c2ec9ed1b4), adapted to Khan's single head camera.
Raw episodes need only MP4 + JSON frame references; this converter materializes
the camera observations as embedded PNG images in the LeRobot v2.1 dataset.
Native image resolution is retained; OpenPI's model transforms resize with pad.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

from jaka_record_lib import JAKA_JOINT_NAMES, _OPENHLM_ACTION_NAMES, _OPENHLM_STATE_NAMES

PERMUTATION = np.array(list(range(13, 25)) + list(range(12)) + [12, 25, 26])
POLICY_STATE_NAMES = [JAKA_JOINT_NAMES[i] for i in PERMUTATION] + _OPENHLM_STATE_NAMES[27:]
POLICY_ACTION_NAMES = POLICY_STATE_NAMES + _OPENHLM_ACTION_NAMES[30:]


@dataclass
class Episode:
    path: Path
    info: dict[str, Any]
    rows: list[dict[str, Any]]
    task: str
    frequency: int
    states: np.ndarray
    actions: np.ndarray
    summary: dict[str, Any]
    image_shape: tuple[int, int, int]


def _vector(value: Any, size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,) or not np.isfinite(result).all() or np.any(np.abs(result) > np.finfo(np.float32).max):
        raise ValueError(f"{name}: expected finite shape ({size},), got {result.shape}")
    return result


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name}: expected an integer")
    return value


def _quat(value: Any, name: str) -> np.ndarray:
    q = _vector(value, 4, name)
    norm = float(np.linalg.norm(q))
    if not np.isfinite(norm) or norm < 1e-8:
        raise ValueError(f"{name}: zero quaternion")
    return q / norm


def _yaw(quaternions: np.ndarray) -> np.ndarray:
    w, x, y, z = quaternions.T
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _inverse_rotate(q: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    w, xyz = q[:, :1], q[:, 1:]
    return vectors * (2 * w * w - 1) - 2 * w * np.cross(xyz, vectors) + 2 * xyz * np.sum(xyz * vectors, axis=1, keepdims=True)


def _record_intervals(rows: list[dict]) -> np.ndarray:
    stamps = []
    for index, row in enumerate(rows):
        try:
            stamps.append(_integer(row["raw"]["record_monotonic_ns"], f"row {index} record_monotonic_ns"))
        except (KeyError, TypeError) as exc:
            raise ValueError(f"row {index}: recorded-time differences require raw.record_monotonic_ns") from exc
    dt = np.array([b - a for a, b in zip(stamps, stamps[1:])], dtype=np.float64) * 1e-9
    if np.any(dt <= 0):
        raise ValueError("record_monotonic_ns must strictly increase")
    return dt


def recompute_velocities(rows: list[dict], states: np.ndarray, actions: np.ndarray, mode: str,
                         *, frequency: int | None = None, state_mode: str = "recorded",
                         timebase: str = "nominal") -> tuple[np.ndarray, np.ndarray]:
    """Apply independent causal-state and action-label derivative contracts.

    Forward action uses the starting row's orientation and describes the next
    exported interval; backward uses the ending row's orientation. Nominal
    action dt=1/frequency matches fixed-fps playback without changing row order.
    Record-time action differences estimate velocity on actual selected-row
    times; they are valid labels but are not an exact inverse of fixed-dt replay.
    State is retained or differentiated backward on recorder monotonic time;
    future state information is never introduced. No derivative is clipped.
    """
    if mode not in {"recorded", "backward", "forward"}:
        raise ValueError(f"Unknown action velocity mode: {mode}")
    if state_mode not in {"recorded", "backward"}:
        raise ValueError("State velocity mode must be recorded or causal backward")
    if timebase not in {"nominal", "record"}:
        raise ValueError(f"Unknown action velocity timebase: {timebase}")
    out_states, out_actions = states.copy(), actions.copy()
    if mode != "recorded":
        action_q, positions = [], []
        for index, row in enumerate(rows):
            try:
                motion = row["raw"]["motion"]
                action_q.append(_quat(motion["anchor_quat_wxyz"], f"row {index} action quaternion"))
                positions.append(_vector(motion["anchor_pos_w"], 3, f"row {index} anchor position"))
            except (KeyError, TypeError) as exc:
                raise ValueError(f"row {index}: {mode} action differences require raw.motion poses; use recorded to preserve legacy values") from exc
        if timebase == "nominal":
            if not isinstance(frequency, int) or isinstance(frequency, bool) or frequency <= 0:
                raise ValueError("Nominal action differences require a positive integer frequency")
            dt = np.full(max(0, len(rows) - 1), 1.0 / frequency, dtype=np.float64)
        else:
            dt = _record_intervals(rows)
        action_q, positions = np.asarray(action_q), np.asarray(positions)
        out_actions[:, 29:33] = 0
        target = slice(1, None) if mode == "backward" else slice(None, -1)
        dyaw = (np.diff(_yaw(action_q)) + np.pi) % (2 * np.pi) - np.pi
        out_actions[target, 29] = dyaw / dt
        out_actions[target, 30:33] = _inverse_rotate(action_q[target], np.diff(positions, axis=0)) / dt[:, None]
    if state_mode == "backward":
        state_q = []
        for index, row in enumerate(rows):
            try:
                state_q.append(_quat(row["raw"]["state"]["base_quat_wxyz"], f"row {index} state quaternion"))
            except (KeyError, TypeError) as exc:
                raise ValueError(f"row {index}: backward state differences require raw.state.base_quat_wxyz") from exc
        dt = _record_intervals(rows)
        out_states[:, 29] = 0
        dyaw = (np.diff(_yaw(np.asarray(state_q))) + np.pi) % (2 * np.pi) - np.pi
        out_states[1:, 29] = dyaw / dt
    if not np.isfinite(out_states).all() or not np.isfinite(out_actions).all():
        raise ValueError("Velocity recomputation produced non-finite values")
    if np.any(np.abs(out_states) > np.finfo(np.float32).max) or np.any(np.abs(out_actions) > np.finfo(np.float32).max):
        raise ValueError("Velocity recomputation exceeds float32 range")
    return out_states, out_actions


class VideoReader:
    """Read every referenced frame; reject missing, reordered, or partial video."""
    def __init__(self, episode_path: Path):
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("OpenCV is required even for --dry-run video validation") from exc
        self.cv2 = cv2
        self.root = episode_path.resolve()
        self.capture = None
        self.path = None
        self.next_index = 0

    def read(self, row: dict) -> np.ndarray:
        ref = row.get("head_image_left")
        if not isinstance(ref, dict):
            raise ValueError("Missing head_image_left video reference")
        relative = ref.get("video_path")
        if not isinstance(relative, str):
            raise ValueError("Invalid video_path")
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("video_path must remain inside its episode directory")
        index = _integer(ref.get("frame_index"), "head_image_left.frame_index")
        if index != self.next_index:
            raise ValueError(f"Video frame references must be contiguous from 0: expected {self.next_index}, got {index}")
        if self.capture is None:
            self.path = path
            self.capture = self.cv2.VideoCapture(str(path))
            if not self.capture.isOpened():
                raise ValueError(f"Cannot open video: {path}")
        elif path != self.path:
            raise ValueError("One head video per episode is required")
        ok, bgr = self.capture.read()
        if not ok or bgr is None:
            raise ValueError(f"Cannot decode video frame {index}: {path}")
        self.next_index += 1
        return self.cv2.cvtColor(bgr, self.cv2.COLOR_BGR2RGB)

    def check_end(self) -> None:
        if self.capture is not None:
            ok, _ = self.capture.read()
            if ok:
                raise ValueError("Video contains extra unreferenced frames")

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()


def _xr_source_descriptor(metadata: dict, xr_sources, row_index: int) -> tuple[dict, str | None]:
    """Resolve static XR metadata without expanding or mutating recorded rows."""
    if metadata.get("version") != 2 and "metadata_id" not in metadata:
        return metadata, None  # Legacy v1 inline diagnostics remain optional.
    reference = metadata.get("metadata_id")
    prefix = f"row {row_index}: xr_sample.metadata_id"
    if not isinstance(reference, str) or not reference:
        raise ValueError(f"{prefix} must reference an info.xr_sources descriptor")
    if not isinstance(xr_sources, dict) or reference not in xr_sources:
        raise ValueError(f"{prefix} {reference!r} is missing from info.xr_sources")
    descriptor = xr_sources[reference]
    if not isinstance(descriptor, dict):
        raise ValueError(f"{prefix} {reference!r} resolves to an invalid XR descriptor")
    names = descriptor.get("joint_names")
    if (not isinstance(names, list) or len(names) != 24
            or not all(isinstance(name, str) and name for name in names)
            or len(set(names)) != 24):
        raise ValueError(f"{prefix} {reference!r}: XR descriptor needs 24 unique joint_names")
    mode = descriptor.get("diagnostics_mode")
    if mode is not None and mode not in ("light", "full"):
        raise ValueError(f"{prefix}: diagnostics_mode must be light or full")
    head_index = descriptor.get("head_joint_index")
    if head_index is not None and (isinstance(head_index, bool)
            or not isinstance(head_index, int) or not 0 <= head_index < len(names)
            or names[head_index] != "Head"):
        raise ValueError(f"{prefix}: head_joint_index must identify the named Head joint")
    if mode == "light" and any(key in metadata for key in (
            "raw_body_poses", "joint_timestamps_raw", "body_timestamp_ns")):
        raise ValueError(f"{prefix}: light diagnostics must omit full XR fields")
    return descriptor, reference


def _xr_scalar_clock_diagnostics(records: list[tuple]) -> dict:
    """Count optional clocks without converting absolute integer stamps to float."""
    result = {"positive": 0, "zero": 0, "negative": 0, "missing": 0, "invalid": 0,
              "increased_pairs": 0, "repeated_pairs": 0, "regressed_pairs": 0}
    deltas = []
    previous = (None, None, None)
    for value, session, domain in records:
        prior, prior_session, prior_domain = previous
        valid = isinstance(value, int) and not isinstance(value, bool)
        if value is None:
            result["missing"] += 1
        elif not valid:
            result["invalid"] += 1
        else:
            result["positive" if value > 0 else "zero" if value == 0 else "negative"] += 1
            if (value > 0 and isinstance(prior, int) and not isinstance(prior, bool) and prior > 0
                    and session is not None and session == prior_session and domain == prior_domain):
                delta = value - prior
                if delta > 0:
                    result["increased_pairs"] += 1
                    deltas.append(delta)
                else:
                    result["repeated_pairs" if delta == 0 else "regressed_pairs"] += 1
        # Missing or malformed input breaks adjacency, as does a domain change.
        previous = (value if valid else None, session, domain)
    deltas.sort()
    count = len(deltas)
    result["positive_delta_raw_min_p50_p95_max"] = (
        [deltas[0], deltas[(count + 1) // 2 - 1], deltas[(95 * count + 99) // 100 - 1], deltas[-1]]
        if count else [])
    return result


def _xr_joint_timestamp_diagnostics(samples: list[dict], xr_sources=None) -> dict:
    """Describe optional SDK clocks without assigning them capture-time meaning.

    Legacy inline diagnostics are optional, but explicit references must resolve
    to intact episode metadata. Keep timestamps as Python integers; subtract
    before computing order statistics. No per-row static metadata copies.
    """
    records, joints = [], {}
    head_records, global_records = [], []
    mode_counts = {"light": 0, "full": 0, "legacy_unspecified": 0}
    omitted = {"joint_timestamps_raw": 0, "body_timestamp_ns": 0, "raw_body_poses": 0}
    invalid_vectors = missing_vectors = 0
    present = valid_vectors = invalid_metadata = all_positive_equal = 0
    for row_index, sample in enumerate(samples):
        metadata = sample.get("xr_sample")
        present += metadata is not None
        if not isinstance(metadata, dict):
            invalid_metadata += metadata is not None
            records.append((None, [], None, None, None))
            head_records.append((None, None, None))
            global_records.append((None, None, None))
            continue
        descriptor, metadata_id = _xr_source_descriptor(metadata, xr_sources, row_index)
        mode = descriptor.get("diagnostics_mode")
        mode = mode if isinstance(mode, str) and mode in ("light", "full") else "legacy_unspecified"
        mode_counts[mode] += 1
        if mode == "light":
            for key in omitted:
                omitted[key] += key not in metadata
        names = descriptor.get("joint_names")
        names_valid = (isinstance(names, list) and len(names) == 24
                       and all(isinstance(name, str) and name for name in names)
                       and len(set(names)) == 24)
        stamps = metadata.get("joint_timestamps_raw")
        stamps = stamps if isinstance(stamps, list) else []
        vector_valid = (names_valid and len(stamps) == 24
                        and all(isinstance(value, int) and not isinstance(value, bool) for value in stamps))
        valid_vectors += vector_valid
        if "joint_timestamps_raw" not in metadata:
            missing_vectors += mode != "light"
        else:
            invalid_vectors += not vector_valid
        if vector_valid and all(value > 0 for value in stamps) and len(set(stamps)) == 1:
            all_positive_equal += 1
        names = tuple(names) if names_valid else None
        if names is not None:
            for index, name in enumerate(names):
                joints.setdefault((index, name), {
                    "index": index, "name": name, "positive": 0, "zero": 0,
                    "negative": 0, "missing": 0, "omitted_by_mode": 0, "repeated_pairs": 0,
                    "regressed_pairs": 0, "increased_pairs": 0, "_deltas": [],
                })
        session = sample.get("reference_session_id")
        session = session if isinstance(session, str) and session else None
        # Explicit descriptors are separate clock/layout domains even when the
        # joint names happen to match. Legacy inline clocks also respect their
        # declared SDK/unit/context; do not infer a mapping between domains.
        domain = (metadata_id, descriptor.get("context"), descriptor.get("unit"), descriptor.get("sdk"))
        records.append((names, stamps, session, domain, mode))
        head_index = descriptor.get("head_joint_index")
        if "head_joint_index" not in descriptor and names is not None and "Head" in names:
            head_index = names.index("Head")
        head_value = metadata.get("head_timestamp_raw")
        if ("head_timestamp_raw" not in metadata and isinstance(head_index, int)
                and not isinstance(head_index, bool) and 0 <= head_index < len(stamps)):
            head_value = stamps[head_index]
        head_records.append((head_value, session, (domain, head_index)))
        global_records.append((metadata.get("global_timestamp_ns"), session, domain))
    previous = (None, [], None, None)
    for names, stamps, session, context, mode in records:
        previous_names, previous_stamps, previous_session, previous_context = previous
        comparable = (names is not None and names == previous_names
                      and len(stamps) == len(previous_stamps) == 24
                      and session is not None and session == previous_session
                      and context == previous_context)
        for (index, name), stats in joints.items():
            value = stamps[index] if names is not None and names[index] == name and index < len(stamps) else None
            if not isinstance(value, int) or isinstance(value, bool):
                stats["omitted_by_mode" if mode == "light" else "missing"] += 1
                continue
            stats["positive" if value > 0 else "zero" if value == 0 else "negative"] += 1
            if not comparable or names[index] != name or value <= 0:
                continue
            prior = previous_stamps[index]
            if not isinstance(prior, int) or isinstance(prior, bool) or prior <= 0:
                continue
            delta = value - prior
            if delta > 0:
                stats["increased_pairs"] += 1
                stats["_deltas"].append(delta)
            else:
                stats["repeated_pairs" if delta == 0 else "regressed_pairs"] += 1
        # Missing/malformed rows break comparisons; never bridge over them.
        previous = (names, stamps, session, context)
    for stats in joints.values():
        deltas = sorted(stats.pop("_deltas"))
        count = len(deltas)
        stats["positive_delta_raw_min_p50_p95_max"] = (
            [deltas[0], deltas[(count + 1) // 2 - 1], deltas[(95 * count + 99) // 100 - 1], deltas[-1]]
            if count else [])
    return {
        "rows_with_metadata": present, "rows_without_metadata": len(samples) - present,
        "rows_with_invalid_metadata_object": invalid_metadata,
        "rows_with_valid_vector": valid_vectors,
        "rows_with_invalid_vector": invalid_vectors,
        "rows_with_missing_vector_not_light": missing_vectors,
        "diagnostics_mode_rows": mode_counts,
        "fields_intentionally_omitted_in_light_rows": omitted,
        "head_timestamp_raw": _xr_scalar_clock_diagnostics(head_records),
        "global_timestamp_ns": _xr_scalar_clock_diagnostics(global_records),
        "rows_all_joint_stamps_equal_and_positive": all_positive_equal,
        "unit": "sdk_raw_unverified",
        "semantics": "Light records Head only, full additionally records the joint vector; absent mode is legacy. Head units/epoch and capture-time meaning are unverified. Global is a source packet wall clock. Neither clock is used for label dt or treated as a whole-body capture clock.",
        "comparison_policy": "Adjacent rows only, matching 24 joint names, known publisher session and static metadata/clock domain, positive integer pair. Missing includes absent, malformed or differently named joint data.",
        "metadata_resolution": "Read static descriptors from info.xr_sources without expanding rows; legacy v1 inline metadata remains supported. Explicit references must resolve.",
        "percentile_method": "nearest_rank on exact Python-integer positive deltas; absolute timestamps never converted to float",
        "joints": [joints[key] for key in sorted(joints)],
    }


def _timing_summary(rows: list[dict], fps: int, xr_sources=None) -> dict:
    wall = [_integer(row.get("timestamp_ns"), "timestamp_ns") for row in rows]
    wall_nonincreasing = sum(b <= a for a, b in zip(wall, wall[1:]))
    mono = [(row.get("raw") or {}).get("record_monotonic_ns") for row in rows]
    if any(value is not None for value in mono) and not all(isinstance(value, int) and not isinstance(value, bool) for value in mono):
        raise ValueError("record_monotonic_ns is incomplete or invalid")
    clock = "record_monotonic_ns" if all(value is not None for value in mono) else "timestamp_ns"
    stamps = mono if clock == "record_monotonic_ns" else wall
    intervals = np.array([b - a for a, b in zip(stamps, stamps[1:])], dtype=np.float64) * 1e-9
    if np.any(intervals <= 0):
        raise ValueError(f"{clock} must strictly increase; no automatic sorting or deduplication")
    samples, source_times, source_domains = [], [], []
    for row in rows:
        sample = (row.get("raw") or {}).get("motion") or {}
        if not isinstance(sample, dict):
            raise ValueError("raw.motion must be a metadata object")
        for key in ("seq", "source_t_ns", "smplx_t_ns", "receive_monotonic_ns"):
            if sample.get(key) is not None:
                _integer(sample[key], f"raw.motion.{key}")
        # Preserve zero/missing XR stamps as unknown. Do not replace them with a
        # publisher/receiver clock and then call it a body-sample identity.
        if "smplx_t_ns" in sample:
            source = sample["smplx_t_ns"]
        elif sample.get("xr_sample") is not None or sample.get("source_clock") == "publisher_wall_or_receiver_fallback":
            source = None  # Light rows may intentionally omit an invalid Body timestamp.
        else:
            source = sample.get("source_t_ns")  # Truly legacy, without explicit XR metadata.
        source_times.append(source if source is not None and source > 0 else None)
        source_domains.append((sample.get("source_clock"), sample.get("paused"),
                               sample.get("reference_session_id")))
        samples.append(sample)
    repeated_seq = repeated_source = source_regressions = repeated_messages = repeated_pose = 0
    for i, (a, b) in enumerate(zip(samples, samples[1:])):
        if a.get("seq") is not None and b.get("seq") == a["seq"]:
            repeated_seq += 1
        receive = a.get("receive_monotonic_ns")
        if receive is not None and receive > 0 and b.get("receive_monotonic_ns") == receive:
            repeated_messages += 1
        if (source_times[i] is not None and source_times[i + 1] is not None
                and source_domains[i] == source_domains[i + 1]):
            repeated_source += int(source_times[i + 1] == source_times[i])
            source_regressions += int(source_times[i + 1] < source_times[i])
        if all(key in sample for sample in (a, b) for key in ("anchor_pos_w", "anchor_quat_wxyz")):
            p0 = _vector(a["anchor_pos_w"], 3, "raw.motion.anchor_pos_w")
            p1 = _vector(b["anchor_pos_w"], 3, "raw.motion.anchor_pos_w")
            q0 = _quat(a["anchor_quat_wxyz"], "raw.motion.anchor_quat_wxyz")
            q1 = _quat(b["anchor_quat_wxyz"], "raw.motion.anchor_quat_wxyz")
            repeated_pose += int(np.array_equal(p0, p1) and
                                 (np.allclose(q0, q1, rtol=0, atol=1e-12) or np.allclose(q0, -q1, rtol=0, atol=1e-12)))
    return {
        "clock": clock,
        "interval_ms_min_p50_p95_max": (np.percentile(intervals * 1000, [0, 50, 95, 100]).tolist() if len(intervals) else []),
        "max_relative_interval_deviation": float(np.max(np.abs(intervals * fps - 1))) if len(intervals) else 0,
        "observed_duration_s": float(intervals.sum()),
        "exported_duration_s": (len(rows) - 1) / fps,
        "wall_nonincreasing_pairs": int(wall_nonincreasing),
        "repeated_motion_seq_pairs": repeated_seq,
        "reused_motion_message_pairs": repeated_messages,
        "repeated_motion_pose_pairs": repeated_pose,
        "repeated_motion_source_time_pairs": repeated_source,
        "motion_source_time_regressions": source_regressions,
        "motion_source_time_unknown_rows": sum(stamp is None for stamp in source_times),
        "source_time_metric": "smplx_t_ns when present, otherwise source_t_ns only for legacy rows without XR metadata or an explicit publication fallback; omitted/zero Body stamps stay unknown; positive values only in a matching clock/session domain",
        "repeated_sample_policy": "Retain selected row poses and their intervals; an unchanged pose yields zero difference. No deduplication or XR freshness inference.",
        "xr_joint_timestamp_diagnostics": _xr_joint_timestamp_diagnostics(samples, xr_sources),
        "row_timestamps_ns": wall,
        "row_record_monotonic_ns": mono,
    }


def inspect_episode(path: Path, *, velocity_mode: str = "forward", legacy_anchor: str | None = None,
                    allow_repeated_source: bool = False, max_interval_deviation: float | None = None,
                    state_velocity_mode: str = "recorded", velocity_timebase: str = "nominal") -> Episode:
    json_path = path / "data.json"
    contents = json_path.read_bytes()
    payload = json.loads(contents)
    if not isinstance(payload, dict):
        raise ValueError(f"{json_path}: expected an episode object")
    info, rows = payload.get("info", {}), payload.get("data")
    if not isinstance(info, dict) or not isinstance(rows, list) or not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{json_path}: empty or missing data")
    if any(row.get("raw") is not None and not isinstance(row["raw"], dict) for row in rows):
        raise ValueError("raw must be a metadata object")
    quality = info.get("quality") or {}
    if not isinstance(quality, dict):
        raise ValueError("info.quality must be an object")
    if quality.get("complete") is False or quality.get("write_errors"):
        raise ValueError("Writer marked this episode incomplete or reported write errors")
    if quality.get("frames_dropped", 0):
        raise ValueError("Writer reported dropped frames; refusing to silently compress the episode")
    if quality.get("requires_review"):
        print(f"WARNING: {path} is marked requires_review; inspect its quality metadata before training", file=sys.stderr)
    anchor = info.get("anchor_body_name")
    if not anchor:
        if str(info.get("schema_version", "")).startswith("2."):
            raise ValueError("Schema v2 must explicitly identify anchor_body_name")
        anchor = legacy_anchor
        if anchor is None:
            raise ValueError("Unmarked legacy anchor: specify --legacy-anchor only after verifying the source")
    if anchor != "base_link":
        raise ValueError(f"Anchor {anchor!r} cannot be converted to base_link by relabeling; re-record or perform a verified FK conversion")
    for key, expected in (("state_dim", 30), ("action_dim", 40)):
        if info.get(key) != expected:
            raise ValueError(f"Expected info.{key}={expected}")
    for key, expected in (("state_names", _OPENHLM_STATE_NAMES), ("action_names", _OPENHLM_ACTION_NAMES)):
        if info.get(key) != expected:
            raise ValueError(f"info.{key} must describe the raw legs-first Khan layout")
    fps = _integer(info.get("frequency"), "info.frequency")
    if fps <= 0:
        raise ValueError("info.frequency must be positive")
    indices = [_integer(row.get("idx"), "row.idx") for row in rows]
    if indices != list(range(len(rows))):
        raise ValueError("idx must be contiguous from 0; refusing to silently compress a dropped-frame gap")
    states = np.stack([_vector(row.get("state"), 30, f"row {i} state") for i, row in enumerate(rows)])
    actions = np.stack([_vector(row.get("actions"), 40, f"row {i} actions") for i, row in enumerate(rows)])
    timing = _timing_summary(rows, fps, info.get("xr_sources"))
    if max_interval_deviation is not None and timing["max_relative_interval_deviation"] > max_interval_deviation:
        raise ValueError("Sample interval deviation exceeds the explicitly configured limit")
    # Repeated source IDs are diagnostics, not a veto on differentiating the
    # selected reference trajectory. A held pose has a legitimate zero delta.
    # allow_repeated_source remains accepted for old callers but is a no-op.
    states, actions = recompute_velocities(
        rows, states, actions, velocity_mode, frequency=fps,
        state_mode=state_velocity_mode, timebase=velocity_timebase,
    )
    task = payload.get("text", {}).get("goal") or payload.get("text", {}).get("desc")
    if not isinstance(task, str) or not task.strip():
        raise ValueError("A nonempty task goal/description is required")
    reader, shape = VideoReader(path), None
    try:
        for row in rows:
            image = reader.read(row)
            if shape is not None and image.shape != shape:
                raise ValueError("Image shape changes within an episode")
            shape = image.shape
        reader.check_end()
    finally:
        reader.close()
    summary = {"source_episode": str(path.resolve()), "source_json_sha256": hashlib.sha256(contents).hexdigest(),
               "source_info": info, "frames": len(rows), "task": task, "timing": timing,
               "legacy_anchor_assertion": legacy_anchor if not info.get("anchor_body_name") else None}
    return Episode(path, info, rows, task, fps, states.astype(np.float32), actions.astype(np.float32), summary, tuple(shape))


def reorder(states: np.ndarray, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.concatenate((states[:, PERMUTATION], states[:, 27:]), axis=1), np.concatenate((actions[:, PERMUTATION], actions[:, 27:]), axis=1)


def find_episodes(data_dir: Path) -> list[Path]:
    if not data_dir.is_dir():
        raise ValueError(f"Not a directory: {data_dir}")
    paths = [data_dir] if data_dir.name.startswith("episode_") else sorted(p for p in data_dir.rglob("episode_*") if p.is_dir())
    if not paths:
        raise ValueError(f"No episode directories under {data_dir}")
    for path in paths:
        if not (path / "data.json").is_file():
            raise ValueError(f"Incomplete episode missing data.json: {path}")
    return paths


def _lerobot_class():
    environment_hint = (
        "Use OpenHLM's pinned LeRobot commit "
        "0cf864870cf29f4738d3ade893e6fd13fbd7cdb5 with its dependencies "
        "(Python package 0.1.0, dataset format v2.1). "
        "LeRobot 0.3.3 is also supported for standalone conversion; "
        "no training-environment upgrade is required. --dry-run needs no LeRobot"
    )
    current = "lerobot.datasets.lerobot_dataset"
    legacy = "lerobot.common.datasets.lerobot_dataset"
    try:
        module = importlib.import_module(current)
    except ModuleNotFoundError as exc:
        if exc.name not in {"lerobot", "lerobot.datasets", current}:
            raise RuntimeError(f"LeRobot dependency {exc.name!r} is missing. {environment_hint}") from exc
        try:
            module = importlib.import_module(legacy)
        except ModuleNotFoundError as legacy_exc:
            raise RuntimeError(f"Cannot load LeRobot (missing {legacy_exc.name!r}). {environment_hint}") from legacy_exc
        except ImportError as legacy_exc:
            raise RuntimeError(f"LeRobot dependency/API import failed: {legacy_exc}. {environment_hint}") from legacy_exc
    except ImportError as exc:
        raise RuntimeError(f"LeRobot dependency/API import failed: {exc}. {environment_hint}") from exc
    version = getattr(module, "CODEBASE_VERSION", None)
    if version != "v2.1":
        raise RuntimeError(f"This OpenHLM dataset contract requires LeRobot format v2.1, got {version!r}. "
                           f"This is the dataset format, not the Python package version. {environment_hint}. No output was written")
    return module.LeRobotDataset


def convert(args: argparse.Namespace) -> dict:
    output = args.output.resolve()
    if output.exists():
        raise ValueError(f"Output already exists; refusing to overwrite: {output}")
    task_override = getattr(args, "task", None)
    if task_override is not None:
        if not isinstance(task_override, str) or not task_override.strip():
            raise ValueError("--task must be a nonempty instruction")
        task_override = task_override.strip()
    state_mode = getattr(args, "state_velocity_mode", "recorded")
    timebase = getattr(args, "velocity_timebase", "nominal")
    paths = find_episodes(args.data_dir)
    episodes = [inspect_episode(path, velocity_mode=args.velocity_mode, legacy_anchor=args.legacy_anchor,
                                allow_repeated_source=args.allow_repeated_source,
                                max_interval_deviation=args.max_interval_deviation,
                                state_velocity_mode=state_mode, velocity_timebase=timebase) for path in paths]
    for episode in episodes:
        episode.summary["source_task"] = episode.task
        episode.summary["task_overridden"] = task_override is not None
        if task_override is not None:
            episode.task = task_override
        episode.summary["task"] = episode.task
    if args.allow_repeated_source:
        print("NOTE: --allow-repeated-source is deprecated; selected hold rows are always retained and source timestamps are diagnostic only.", file=sys.stderr)
    fps, shape = episodes[0].frequency, episodes[0].image_shape
    if any(episode.frequency != fps or episode.image_shape != shape for episode in episodes):
        raise ValueError("All episodes must share frequency and head image shape; no implicit resizing or resampling")
    action_recorded = args.velocity_mode == "recorded"
    action_semantics = {
        "mode": args.velocity_mode,
        "timebase": "preserved_from_recording" if action_recorded else
                    ("exported nominal row timeline (dt=1/frequency)" if timebase == "nominal" else "raw.record_monotonic_ns differences"),
        "dt_seconds": 1.0 / fps if not action_recorded and timebase == "nominal" else None,
        "orientation": "preserved_from_recording" if action_recorded else
                       ("current row anchor quaternion, interval start" if args.velocity_mode == "forward" else
                        "current row anchor quaternion, interval end"),
        "interval": "preserved_from_recording" if action_recorded else
                    ("row i to i+1" if args.velocity_mode == "forward" else "row i-1 to i"),
        "boundary": "preserved_from_recording" if action_recorded else
                    ("last row yaw/linear velocities set to zero" if args.velocity_mode == "forward" else
                     "first row yaw/linear velocities set to zero"),
        "clipping": "preserved_from_recording" if action_recorded else "none",
        "pose_roundtrip": "Forward nominal labels are discrete inverses of start-pose publication then fixed-dt integration, given matching initial pose, action roll/pitch and raw quaternions. No hardware timing equivalence is implied.",
    }
    state_semantics = {
        "mode": state_mode,
        "timebase": "preserved_from_recording (see each source_info.velocity_semantics)" if state_mode == "recorded" else "raw.record_monotonic_ns differences",
        "interval": "preserved causal observation" if state_mode == "recorded" else "row i-1 to i",
        "boundary": "preserved_from_recording" if state_mode == "recorded" else "first row yaw velocity set to zero",
        "clipping": "preserved_from_recording" if state_mode == "recorded" else "none",
        "uses_future_state": False,
    }
    manifest = {
        "schema_version": "2.0.0", "status": "validated" if args.dry_run else "writing",
        "anchor_body_name": "base_link", "joint_order": "arms_first", "state_dim": 30, "action_dim": 40,
        "repo_id": args.repo_id, "frequency": fps, "image_shape": list(shape), "velocity_mode": args.velocity_mode,
        "task_override": task_override,
        "state_velocity_mode": state_mode, "velocity_timebase": timebase,
        "velocity_semantics": {"action": action_semantics, "state": state_semantics},
        "time_policy": "Exported row i has nominal time i/frequency. Preserve row/image order; no source synchronization, timestamp shift, pose interpolation, or resampling. Actual recording times/cadence are retained below and can differ from the exported timeline.",
        "allow_repeated_source": args.allow_repeated_source,
        "allow_repeated_source_deprecated": True,
        "max_interval_deviation": args.max_interval_deviation,
        "episodes": [episode.summary for episode in episodes],
    }
    for episode in episodes:
        stats = episode.summary["timing"]
        print(f"Validated {episode.path}: {len(episode.rows)} frames @ {fps} Hz; "
              f"interval ms min/p50/p95/max={stats['interval_ms_min_p50_p95_max']}; "
              f"repeated seq/source/pose={stats['repeated_motion_seq_pairs']}/{stats['repeated_motion_source_time_pairs']}/{stats['repeated_motion_pose_pairs']}; "
              f"unknown source time rows={stats['motion_source_time_unknown_rows']}", file=sys.stderr)
        print(f"  Original span={stats['observed_duration_s']:.6f}s; exported span={stats['exported_duration_s']:.6f}s. "
              "The output uses declared-fps sample order, not a synchronized timeline.", file=sys.stderr)
    if args.dry_run:
        return manifest
    dataset_class = _lerobot_class()
    dataset = dataset_class.create(repo_id=args.repo_id, root=output, robot_type="jaka", fps=fps, use_videos=False,
                                   features={"head_image_left": {"dtype": "image", "shape": shape, "names": ["height", "width", "channel"]},
                                             "state": {"dtype": "float32", "shape": (30,), "names": POLICY_STATE_NAMES},
                                             "actions": {"dtype": "float32", "shape": (40,), "names": POLICY_ACTION_NAMES}})
    manifest_path = output / "conversion_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    separate_task = "task" in inspect.signature(dataset.add_frame).parameters
    try:
        for episode in episodes:
            states, actions = reorder(episode.states, episode.actions)
            reader = VideoReader(episode.path)
            try:
                for i, row in enumerate(episode.rows):
                    frame = {"head_image_left": reader.read(row), "state": states[i], "actions": actions[i]}
                    if separate_task:
                        dataset.add_frame(frame, task=episode.task)
                    else:
                        dataset.add_frame({**frame, "task": episode.task})
                reader.check_end()
            finally:
                reader.close()
            dataset.save_episode()
        finalize = getattr(dataset, "finalize", None)
        if callable(finalize):
            finalize()
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
        raise
    manifest["status"] = "complete"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--task", help="Explicitly override the instruction for all exported episodes; preserve source JSON and record the original task in the manifest")
    parser.add_argument("--dry-run", action="store_true", help="Validate all JSON and decode all video frames without importing LeRobot or writing output")
    parser.add_argument("--velocity-mode", choices=("recorded", "backward", "forward"), default="forward",
                        help="ACTION yaw/linear labels only: default forward; recorded preserves online values")
    parser.add_argument("--state-velocity-mode", choices=("recorded", "backward"), default="recorded",
                        help="State yaw: preserve recorded causal value or recompute backward on recorder monotonic time; never forward")
    parser.add_argument("--velocity-timebase", choices=("nominal", "record"), default="nominal",
                        help="Recomputed ACTION dt: nominal=1/frequency for fixed-step pose replay (default); record=actual recorded-row intervals for elapsed-time velocity estimates")
    parser.add_argument("--legacy-anchor", choices=("base_link", "waist_yaw_Link"), help="Explicitly identify an unmarked legacy source; waist data will be rejected")
    parser.add_argument("--allow-repeated-source", action="store_true", help="Deprecated compatibility no-op: held row poses are retained; unknown/duplicate XR stamps never define derivative dt")
    parser.add_argument("--max-interval-deviation", type=float, help="Optional explicit maximum relative deviation from 1/frequency; otherwise report all intervals without rejecting jitter")
    args = parser.parse_args(argv)
    if args.max_interval_deviation is not None and (not np.isfinite(args.max_interval_deviation) or args.max_interval_deviation < 0):
        parser.error("--max-interval-deviation must be finite and nonnegative")
    try:
        manifest = convert(args)
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"Conversion failed: {exc}", file=sys.stderr)
        return 1
    summary = {key: manifest[key] for key in ("status", "frequency", "velocity_mode", "state_velocity_mode", "velocity_timebase", "time_policy", "task_override")}
    summary["episodes"] = [{"source_episode": episode["source_episode"], "frames": episode["frames"],
                            "task": episode["task"], "source_task": episode["source_task"],
                            "task_overridden": episode["task_overridden"],
                            "timing": {key: value for key, value in episode["timing"].items() if not key.startswith("row_")}}
                           for episode in manifest["episodes"]]
    if not args.dry_run:
        summary["manifest"] = str(args.output.resolve() / "conversion_manifest.json")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
