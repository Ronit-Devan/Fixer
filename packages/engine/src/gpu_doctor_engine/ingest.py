"""Load PyTorch Profiler Chrome trace files."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from gpu_doctor_engine.types import Event, Trace


def _safe_int(value, default: int = 0) -> int | None:
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            try:
                return int(float(value))
            except ValueError:
                return None
    return None


def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping (start, end) intervals. Returns sorted, non-overlapping list."""
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def _busy_time_us(events: list[Event], categories: set[str]) -> int:
    """Total wall-clock time during which at least one matching event was active."""
    intervals = [(e.ts, e.ts + e.dur) for e in events if e.category in categories]
    merged = _merge_intervals(intervals)
    return sum(end - start for start, end in merged)


# PyTorch Profiler GPU activity categories (real traces use these names)
GPU_KERNEL_CATS = {"kernel", "gpu_op"}
GPU_MEMCPY_CATS = {"gpu_memcpy", "gpu_memset"}
GPU_ALL_CATS = GPU_KERNEL_CATS | GPU_MEMCPY_CATS
CPU_CATS = {"cpu_op", "python_function", "user_annotation"}


def load_trace(path: str | Path) -> Trace:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open

    with opener(path, "rt") as f:
        raw = json.load(f)

    raw_events = raw.get("traceEvents", [])

    events: list[Event] = []
    for e in raw_events:
        if e.get("ph") != "X":
            continue
        pid = _safe_int(e.get("pid", 0))
        tid = _safe_int(e.get("tid", 0))
        ts = _safe_int(e.get("ts", 0))
        dur = _safe_int(e.get("dur", 0))
        if pid is None or tid is None or ts is None or dur is None:
            continue
        events.append(
            Event(
                name=e.get("name", ""),
                category=e.get("cat", ""),
                pid=pid,
                tid=tid,
                ts=ts,
                dur=dur,
                args=e.get("args", {}),
            )
        )

    if not events:
        return Trace(events=[], duration_us=0, gpu_kernel_time_us=0, cpu_time_us=0)

    events.sort(key=lambda x: x.ts)

    start = events[0].ts
    end = max(e.ts + e.dur for e in events)

    # GPU-only profile detection: when CPU events are sparse (<5% of total), the
    # wall-clock span is dominated by CUDA initialisation overhead (cudaMalloc, etc.)
    # that runs before the first kernel. Use the GPU active span instead so that
    # gpu_utilization reflects actual kernel density, not pre-profile gaps.
    cpu_event_count = sum(1 for e in events if e.category in CPU_CATS)
    cpu_event_ratio = cpu_event_count / len(events)

    if cpu_event_ratio < 0.05:
        gpu_events = [e for e in events if e.category in GPU_ALL_CATS]
        if gpu_events:
            duration = max(e.ts + e.dur for e in gpu_events) - min(
                e.ts for e in gpu_events
            )
        else:
            duration = end - start
    else:
        duration = end - start

    # Use merged intervals for accurate "busy time" (multiple streams can run in parallel)
    gpu_time = _busy_time_us(events, GPU_KERNEL_CATS)
    cpu_time = _busy_time_us(events, CPU_CATS)

    return Trace(
        events=events,
        duration_us=duration,
        gpu_kernel_time_us=gpu_time,
        cpu_time_us=cpu_time,
    )
