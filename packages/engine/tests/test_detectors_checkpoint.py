"""Unit tests for CHECKPOINT_BOUND idle-window detector."""

from __future__ import annotations

from gpu_doctor_engine import Verdict, diagnose, diagnose_with_stats
from gpu_doctor_engine.detectors.checkpoint import (
    CHECKPOINT_IDLE_SHARE_THRESHOLD,
    CheckpointBoundDetector,
)
from gpu_doctor_engine.diagnose import _gpu_idle_intervals
from tests.helpers import trace_with_checkpoint_share


def test_dtoh_burst_without_idle_share_does_not_fire() -> None:
    """Many DtoH-Pageable events must not fire CHECKPOINT without idle share."""
    trace = trace_with_checkpoint_share(0.20, 0.05, dtoh_count=200)
    diag, stats = diagnose_with_stats(trace)
    share = stats["checkpoint_us"] / max(stats["idle_us"], 1)
    assert share < CHECKPOINT_IDLE_SHARE_THRESHOLD
    assert diag.verdict != Verdict.CHECKPOINT_BOUND


def test_checkpoint_fires_at_25_percent_idle_share() -> None:
    trace = trace_with_checkpoint_share(0.20, 0.25, dtoh_count=0)
    diag = diagnose(trace)
    assert diag.verdict == Verdict.CHECKPOINT_BOUND
    assert 0.55 <= diag.confidence <= 0.98


def test_measure_uses_idle_denominator() -> None:
    trace = trace_with_checkpoint_share(0.20, 0.65, dtoh_count=0)
    start = trace.events[0].ts
    end = max(e.ts + e.dur for e in trace.events)
    intervals = _gpu_idle_intervals(trace.events, start, end)
    idle_us = sum(e - s for s, e in intervals)
    m = CheckpointBoundDetector.measure(intervals, trace.events, idle_us)
    assert m.share >= 0.60
    assert CheckpointBoundDetector.fired(m)
