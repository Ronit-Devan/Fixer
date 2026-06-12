"""Shared idle-window overlap attribution for bottleneck detectors."""

from __future__ import annotations

from dataclasses import dataclass

from gpu_doctor_engine.ingest import _merge_intervals
from gpu_doctor_engine.types import Event


def overlap_idle_time(
    idle_intervals: list[tuple[int, int]],
    events: list[Event],
    patterns: tuple[str, ...],
) -> int:
    """Microseconds of GPU-idle time overlapped by events matching ``patterns``."""
    matching_intervals = [
        (e.ts, e.ts + e.dur)
        for e in events
        if any(p.lower() in e.name.lower() for p in patterns)
    ]
    matching = _merge_intervals(matching_intervals)

    total = 0
    i = j = 0
    while i < len(idle_intervals) and j < len(matching):
        a_start, a_end = idle_intervals[i]
        b_start, b_end = matching[j]
        overlap = max(0, min(a_end, b_end) - max(a_start, b_start))
        total += overlap
        if a_end < b_end:
            i += 1
        else:
            j += 1
    return total


@dataclass(frozen=True)
class IdleShareMeasurement:
    """Idle-window attribution result for one cause family."""

    overlap_us: int
    idle_us: int
    share: float

    def fired_at(self, threshold: float) -> bool:
        return self.overlap_us > 0 and self.share >= threshold
