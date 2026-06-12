"""RingBuffer correctness: capacity, windowing, None-tolerant mean."""

from __future__ import annotations

import pytest

from gpu_doctor_agent.buffer import RingBuffer
from gpu_doctor_agent.sampler import Sample


def _sample(ts: float, util: float | None) -> Sample:
    return Sample(
        timestamp_s=ts,
        gpu_index=0,
        util_pct=util,
        mem_used_mb=1000.0,
        mem_total_mb=40000.0,
        sm_clock_mhz=1500,
        power_w=250.0,
    )


def test_capacity_enforced_drops_oldest() -> None:
    buf = RingBuffer(capacity=10)
    for i in range(20):
        buf.append(_sample(ts=float(i), util=0.5))
    assert len(buf) == 10
    # The 10 most recent must remain (ts 10..19).
    recent = buf.recent(10)
    assert [s.timestamp_s for s in recent] == [float(i) for i in range(10, 20)]


def test_recent_fewer_than_buffer() -> None:
    buf = RingBuffer(capacity=10)
    for i in range(3):
        buf.append(_sample(ts=float(i), util=0.5))
    assert [s.timestamp_s for s in buf.recent(10)] == [0.0, 1.0, 2.0]
    assert buf.recent(0) == []


def test_window_filters_by_time() -> None:
    buf = RingBuffer(capacity=100)
    for i in range(10):
        buf.append(_sample(ts=float(i), util=0.5))  # ts 0..9
    # window of 3 seconds from now=9 -> ts in [6, 9].
    win = buf.window(seconds=3.0, now=9.0)
    assert [s.timestamp_s for s in win] == [6.0, 7.0, 8.0, 9.0]


def test_window_excludes_strictly_older() -> None:
    buf = RingBuffer(capacity=100)
    buf.append(_sample(ts=0.0, util=0.5))
    buf.append(_sample(ts=10.0, util=0.5))
    win = buf.window(seconds=5.0, now=10.0)
    assert [s.timestamp_s for s in win] == [10.0]


def test_mean_util_ignores_none() -> None:
    buf = RingBuffer(capacity=100)
    buf.append(_sample(ts=0.0, util=None))
    buf.append(_sample(ts=1.0, util=0.4))
    buf.append(_sample(ts=2.0, util=None))
    buf.append(_sample(ts=3.0, util=0.6))
    mean = buf.mean_util(seconds=10.0, now=3.0)
    assert mean == pytest.approx(0.5)


def test_mean_util_all_none_returns_none() -> None:
    buf = RingBuffer(capacity=100)
    buf.append(_sample(ts=0.0, util=None))
    buf.append(_sample(ts=1.0, util=None))
    assert buf.mean_util(seconds=10.0, now=1.0) is None


def test_mean_util_empty_window_returns_none() -> None:
    buf = RingBuffer(capacity=100)
    buf.append(_sample(ts=0.0, util=0.5))
    # All samples are older than the window cutoff.
    assert buf.mean_util(seconds=1.0, now=100.0) is None


def test_capacity_zero_rejected() -> None:
    with pytest.raises(ValueError):
        RingBuffer(capacity=0)
