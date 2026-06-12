"""Unit tests for the NCCL_BOUND idle-window detector."""

from __future__ import annotations

import pytest

from gpu_doctor_engine import Verdict, diagnose
from gpu_doctor_engine.diagnose import _gpu_idle_intervals
from gpu_doctor_engine.detectors.nccl import (
    NCCL_IDLE_SHARE_THRESHOLD,
    NCCL_PATTERNS,
    NcclBoundDetector,
)
from tests.helpers import (
    make_cpu_event,
    make_kernel_event,
    make_trace,
    trace_with_nccl_share,
)


def test_nccl_patterns_include_collectives() -> None:
    lowered = {p.lower() for p in NCCL_PATTERNS}
    for token in ("nccl", "allreduce", "allgather", "reducescatter", "broadcast"):
        assert any(token in p for p in lowered), token


def _idle_context(trace):
    start = trace.events[0].ts
    end = max(e.ts + e.dur for e in trace.events)
    intervals = _gpu_idle_intervals(trace.events, start, end)
    idle_us = sum(e - s for s, e in intervals)
    return intervals, idle_us


def test_measure_share_at_threshold_fires() -> None:
    trace = trace_with_nccl_share(util=0.20, nccl_share=NCCL_IDLE_SHARE_THRESHOLD)
    intervals, idle_us = _idle_context(trace)
    m = NcclBoundDetector.measure(intervals, trace.events, idle_us=idle_us)
    assert m.share >= NCCL_IDLE_SHARE_THRESHOLD
    assert NcclBoundDetector.fired(m)


def test_measure_below_threshold_does_not_fire() -> None:
    trace = trace_with_nccl_share(util=0.20, nccl_share=0.29)
    intervals, idle_us = _idle_context(trace)
    m = NcclBoundDetector.measure(intervals, trace.events, idle_us=idle_us)
    assert not NcclBoundDetector.fired(m)


@pytest.mark.parametrize(
    "event_name",
    [
        "ncclAllReduce",
        "nccl:all_gather",
        "c10d::broadcast_",
        "custom ReduceScatter kernel",
    ],
)
def test_collective_name_patterns_match(event_name: str) -> None:
    events = [
        make_kernel_event(ts=0, dur=100_000),
        make_cpu_event(event_name, ts=100_000, dur=400_000),
        make_cpu_event("aten::_dummy", ts=500_000, dur=100_000),
    ]
    trace = make_trace(events, duration_us=600_000, gpu_kernel_time_us=100_000)
    diag = diagnose(trace)
    assert diag.verdict == Verdict.NCCL_BOUND, event_name


def test_nccl_bound_end_to_end_via_diagnose() -> None:
    trace = trace_with_nccl_share(util=0.15, nccl_share=0.50)
    diag = diagnose(trace)
    assert diag.verdict == Verdict.NCCL_BOUND
    assert diag.metrics.get("nccl_share", 0) >= NCCL_IDLE_SHARE_THRESHOLD
    assert "NCCL" in diag.summary
