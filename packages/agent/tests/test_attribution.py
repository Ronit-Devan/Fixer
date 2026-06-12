"""Tier-2 attribution: agent IdleEvent -> engine Diagnosis bridge.

The point of these tests is to prove that the bridge actually drives the
engine's REAL detectors — not just that the wiring exists. Synthesized events
from build_sync_bound_events / build_dataloader_bound_events should produce
the named verdicts when run through the public `attribute()` path. If the
engine's rule order or thresholds ever shift in a way that breaks this
contract, these tests catch it.
"""

from __future__ import annotations

import pytest

from gpu_doctor_agent import attribution as attribution_mod
from gpu_doctor_agent.attribution import (
    MIN_EVENTS,
    attribute,
    format_attributed_alert,
)
from gpu_doctor_agent.detector import IdleEvent
from gpu_doctor_agent.events import (
    EventSource,
    MockEventSource,
    build_dataloader_bound_events,
    build_sync_bound_events,
)
from gpu_doctor_engine import Event, Verdict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _idle_event(
    gpu_index: int = 0, started_at_s: float = 0.0, mean_util: float = 0.05
) -> IdleEvent:
    return IdleEvent(
        gpu_index=gpu_index, started_at_s=started_at_s, mean_util=mean_util
    )


def _filler_kernels(n: int, base_us: int = 1_000_000) -> list[Event]:
    """Padding kernels well above MIN_EVENTS, far from any rule-triggering shape."""
    return [
        Event(
            name="aten::_dummy_filler",
            category="kernel",
            pid=1,
            tid=1,
            ts=base_us + i * 100,
            dur=10,
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# MockEventSource window filtering
# ---------------------------------------------------------------------------


def test_mock_source_filters_events_inside_window() -> None:
    """Events outside [start_us, end_us] are excluded; inside ones are returned."""
    events = [
        Event(name="k0", category="kernel", pid=1, tid=1, ts=0, dur=10),
        Event(name="k1", category="kernel", pid=1, tid=1, ts=5_000_000, dur=10),
        Event(name="k2", category="kernel", pid=1, tid=1, ts=10_000_000, dur=10),
    ]
    src = MockEventSource(events)

    # Window [3s, 7s] keeps only ts=5_000_000.
    captured = src.capture(gpu_index=0, start_s=3.0, end_s=7.0)
    assert [e.name for e in captured] == ["k1"]


def test_mock_source_inclusive_window_bounds() -> None:
    """Events whose ts equals start_us or end_us are included (closed interval)."""
    events = [
        Event(name="lo", category="kernel", pid=1, tid=1, ts=1_000_000, dur=0),
        Event(name="hi", category="kernel", pid=1, tid=1, ts=2_000_000, dur=0),
    ]
    src = MockEventSource(events)
    captured = src.capture(gpu_index=0, start_s=1.0, end_s=2.0)
    assert {e.name for e in captured} == {"lo", "hi"}


def test_mock_source_empty_window_returns_empty() -> None:
    src = MockEventSource(
        [Event(name="k", category="kernel", pid=1, tid=1, ts=0, dur=10)]
    )
    assert src.capture(gpu_index=0, start_s=10.0, end_s=20.0) == []


def test_mock_source_rejects_both_events_and_scenario() -> None:
    with pytest.raises(ValueError):
        MockEventSource(events=[], scenario="sync_bound")


def test_mock_source_rejects_unknown_scenario() -> None:
    with pytest.raises(ValueError):
        MockEventSource(scenario="nonexistent")


def test_mock_source_scenario_constructs_known_shape() -> None:
    """The "sync_bound" scenario uses build_sync_bound_events under the hood."""
    src = MockEventSource(scenario="sync_bound")
    # capture a window that spans all anchored-at-0 events
    captured = src.capture(gpu_index=0, start_s=0.0, end_s=10.0)
    assert any(e.name == "aten::item" for e in captured)
    assert any(e.category == "kernel" for e in captured)


# ---------------------------------------------------------------------------
# attribute(): the MIN_EVENTS guard
# ---------------------------------------------------------------------------


def test_attribute_returns_none_when_window_empty() -> None:
    """Empty capture -> None (no fabricated verdict)."""
    src = MockEventSource(events=[])
    ie = _idle_event(started_at_s=10.0)
    assert attribute(src, ie, now_s=12.0) is None


def test_attribute_returns_none_below_min_events() -> None:
    """Capturing fewer than MIN_EVENTS events -> None."""
    # MIN_EVENTS - 1 events, all within the window.
    n = MIN_EVENTS - 1
    events = [
        Event(name=f"k{i}", category="kernel", pid=1, tid=1, ts=i * 100, dur=10)
        for i in range(n)
    ]
    src = MockEventSource(events)
    ie = _idle_event(started_at_s=0.001)
    diag = attribute(src, ie, now_s=1.0)
    assert diag is None


def test_attribute_min_events_floor_is_inclusive() -> None:
    """Exactly MIN_EVENTS events is enough to attempt diagnose()."""
    events = _filler_kernels(MIN_EVENTS, base_us=0)
    src = MockEventSource(events)
    ie = _idle_event(started_at_s=0.0)
    diag = attribute(src, ie, now_s=2.0)
    # The padded filler shape isn't tuned to a verdict — but it MUST NOT
    # be None, because we crossed the MIN_EVENTS floor and reached diagnose().
    assert diag is not None


def test_attribute_rejects_sparse_noise_window() -> None:
    """A sparse window of tiny noise events is STILL rejected (warm-up gate).

    The sparse-trace gate now mirrors the engine's warmup_trace_guard rather
    than a flat event count, so diagnosable sparse windows (e.g. a 3-event
    checkpoint stall with a 150ms kernel) reach diagnose(). This pins the other
    half of that contract: a genuinely tiny / noisy window — a few kernels with
    microsecond durations over a sub-millisecond span — has no signal and must
    NOT fabricate a verdict, even though it has the same handful of events a
    diagnosable sparse checkpoint window does. The distinction is substance
    (kernel time / span / util), not raw count.
    """
    noise = [
        Event(name="aten::noise", category="kernel", pid=1, tid=1, ts=i * 50, dur=5)
        for i in range(3)
    ]
    src = MockEventSource(noise)
    ie = _idle_event(started_at_s=0.0)
    assert attribute(src, ie, now_s=1.0) is None


# ---------------------------------------------------------------------------
# attribute(): real bridge into the engine
# ---------------------------------------------------------------------------


def test_attribute_sync_bound_events_yield_sync_bound_verdict() -> None:
    """build_sync_bound_events() must drive the engine to SYNC_BOUND.

    This is THE bridge proof: events created on the agent side, fed through
    the public attribute() path, exit the engine as the expected verdict.
    """
    events = build_sync_bound_events(window_start_us=0)
    src = MockEventSource(events)
    ie = _idle_event(started_at_s=0.0)
    diag = attribute(src, ie, now_s=5.0)
    assert diag is not None
    assert diag.verdict == Verdict.SYNC_BOUND


def test_attribute_dataloader_bound_events_yield_dataloader_bound_verdict() -> None:
    """build_dataloader_bound_events() must drive the engine to DATALOADER_BOUND."""
    events = build_dataloader_bound_events(window_start_us=0)
    src = MockEventSource(events)
    ie = _idle_event(started_at_s=0.0)
    diag = attribute(src, ie, now_s=5.0)
    assert diag is not None
    assert diag.verdict == Verdict.DATALOADER_BOUND


def test_attribute_anchored_events_in_nonzero_window() -> None:
    """When events are anchored to a moving window, the verdict still holds.

    This mirrors the live-loop case where idle_event.started_at_s is a real
    monotonic timestamp and events are synthesized relative to it.
    """
    started = 12345.0
    window_start_us = int((started - 5.0) * 1_000_000)
    events = build_sync_bound_events(window_start_us=window_start_us)
    src = MockEventSource(events)
    ie = _idle_event(started_at_s=started)
    diag = attribute(src, ie, now_s=started + 0.2, lookback_s=5.0)
    assert diag is not None
    assert diag.verdict == Verdict.SYNC_BOUND


# ---------------------------------------------------------------------------
# Resilience: engine and source failures must NEVER crash attribution
# ---------------------------------------------------------------------------


def test_attribute_returns_none_when_diagnose_raises(monkeypatch) -> None:
    """If engine.diagnose() raises, attribute() logs and returns None."""

    def _boom(_trace) -> None:
        raise RuntimeError("engine exploded")

    monkeypatch.setattr(attribution_mod, "diagnose", _boom)

    # Use a window that produces >= MIN_EVENTS so we definitely reach diagnose().
    events = build_sync_bound_events(window_start_us=0)
    src = MockEventSource(events)
    ie = _idle_event(started_at_s=0.0)

    # Must NOT raise. Must return None.
    diag = attribute(src, ie, now_s=5.0)
    assert diag is None


def test_attribute_returns_none_when_source_raises() -> None:
    """An event source that raises during capture() must not propagate."""

    class _BoomSource(EventSource):
        def capture(self, gpu_index, start_s, end_s):  # type: ignore[override]
            raise RuntimeError("source exploded")

    ie = _idle_event(started_at_s=0.0)
    diag = attribute(_BoomSource(), ie, now_s=5.0)
    assert diag is None


def test_attribute_clamps_negative_lookback() -> None:
    """Negative lookback is clamped to 0 (window becomes [started, now])."""
    events = build_sync_bound_events(window_start_us=0)
    src = MockEventSource(events)
    ie = _idle_event(started_at_s=0.0)
    # Negative lookback shouldn't raise; events anchored at ts=0 still match.
    diag = attribute(src, ie, now_s=1.0, lookback_s=-1.0)
    assert diag is not None
    assert diag.verdict == Verdict.SYNC_BOUND


def test_attribute_refuses_inverted_window() -> None:
    """If now_s precedes the idle event's start, refuse to attribute."""
    src = MockEventSource(build_sync_bound_events())
    ie = _idle_event(started_at_s=100.0)
    # now_s before started_at_s, lookback=0 -> window_end < window_start.
    diag = attribute(src, ie, now_s=50.0, lookback_s=0.0)
    assert diag is None


# ---------------------------------------------------------------------------
# format_attributed_alert
# ---------------------------------------------------------------------------


def test_format_attributed_alert_with_diagnosis() -> None:
    """A real diagnosis shows verdict, confidence, and summary."""
    src = MockEventSource(build_sync_bound_events())
    ie = _idle_event(started_at_s=0.0, mean_util=0.05)
    diag = attribute(src, ie, now_s=5.0)
    assert diag is not None

    line = format_attributed_alert(ie, diag, sustain_s=5.0)
    assert "GPU 0 idle for 5s" in line
    assert "5.0%" in line  # mean_util * 100
    assert "SYNC_BOUND" in line
    # Confidence rendered as percent (no decimals).
    assert f"{int(diag.confidence * 100)}%" in line


def test_format_attributed_alert_falls_back_when_none() -> None:
    """No diagnosis -> the Tier-1 'attribution pending' fallback line."""
    ie = _idle_event(started_at_s=0.0, mean_util=0.05)
    line = format_attributed_alert(ie, None, sustain_s=5.0)
    assert "attribution pending (Tier 2)" in line
    assert "GPU 0" in line
    assert "5.0%" in line
