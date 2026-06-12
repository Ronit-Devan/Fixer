"""Tests for the gpu_doctor_engine.diagnose module.

Each test builds a synthetic Trace directly from Event dataclasses so that the
rules logic can be exercised in isolation without needing real trace files.
Timestamps are kept in the low-thousands (microseconds) for readability.
"""

from __future__ import annotations

import pytest

from gpu_doctor_engine.diagnose import (
    _PCIE_MIN_ACTIVE_US,
    diagnose,
    diagnose_with_stats,
)
from gpu_doctor_engine.types import Event, Trace, Verdict

from tests.helpers import (
    make_cpu_event,
    make_kernel_event,
    make_trace,
    trace_with_checkpoint_share,
    trace_with_dataloader_share,
    trace_with_memcpy_ratio,
    trace_with_nccl_share,
)

HOL_ACTION_PREFIX = "Profile your __getitem__ for outlier samples"


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_healthy_verdict(healthy_trace: Trace) -> None:
    """GPU running at 95% utilisation should yield HEALTHY with no confidence score."""
    diag = diagnose(healthy_trace)

    assert diag.verdict == Verdict.HEALTHY
    assert diag.confidence is None


# ---------------------------------------------------------------------------
# Bottleneck verdicts
# ---------------------------------------------------------------------------


def test_dataloader_bound_verdict() -> None:
    """Low GPU util with DataLoader CPU ops covering idle windows → DATALOADER_BOUND.

    A 40ms kernel occupies [0, 40ms], leaving [40ms, 200ms] idle. A DataLoader
    CPU event spans the same idle window, so 100% of idle time is attributed to
    the DataLoader bottleneck — well above the 20% threshold required for a
    confident verdict. Duration is 200ms to bypass the small-trace guard.
    """
    events = [
        make_kernel_event(ts=0, dur=40_000),
        make_cpu_event("DataLoader__next_data", ts=40_000, dur=160_000),
    ]
    trace = make_trace(events, duration_us=200_000, gpu_kernel_time_us=40_000)

    diag = diagnose(trace)

    assert diag.verdict == Verdict.DATALOADER_BOUND


def test_pcie_bound_verdict() -> None:
    """Low GPU util with Memcpy events covering idle windows → PCIE_BOUND.

    'Memcpy' appears in MEMCPY_PATTERNS so the overlap-attribution logic should
    assign virtually all idle time to the PCIe bottleneck bucket. Duration is
    200ms to bypass the small-trace guard.
    """
    events = [
        make_kernel_event(ts=0, dur=40_000),
        make_cpu_event("Memcpy HtoD", ts=40_000, dur=160_000),
    ]
    trace = make_trace(events, duration_us=200_000, gpu_kernel_time_us=40_000)

    diag = diagnose(trace)

    assert diag.verdict == Verdict.PCIE_BOUND


def test_nccl_bound_verdict() -> None:
    """Low GPU util with NCCL AllReduce events dominating idle time → NCCL_BOUND.

    'AllReduce' is listed in NCCL_PATTERNS. The NCCL overlap bucket wins because
    it alone fills 160ms of the 160ms idle window (100% attribution). Duration is
    200ms to bypass the small-trace guard.
    """
    events = [
        make_kernel_event(ts=0, dur=40_000),
        make_cpu_event("ncclAllReduce", ts=40_000, dur=160_000),
    ]
    trace = make_trace(events, duration_us=200_000, gpu_kernel_time_us=40_000)

    diag = diagnose(trace)

    assert diag.verdict == Verdict.NCCL_BOUND


def test_checkpoint_bound_verdict() -> None:
    """Low GPU util with a DtoH-Pageable burst → CHECKPOINT_BOUND via strong signal.

    torch.save() pulls every tensor down to unpinned CPU, producing a burst of
    'Memcpy DtoH (Device -> Pageable)' events. 100 such events is well above the
    50-event threshold for the primary checkpoint signal. Duration is 200ms to
    bypass the small-trace guard.
    """
    events = [
        make_kernel_event(ts=0, dur=40_000),
        make_cpu_event("torch.save", ts=40_000, dur=96_000),
        make_cpu_event("DataLoader__next_data", ts=136_000, dur=64_000),
    ]
    events.extend(
        Event(
            name="Memcpy DtoH (Device -> Pageable)",
            category="gpu_memcpy",
            pid=1,
            tid=1,
            ts=40_000 + i,
            dur=0,
        )
        for i in range(100)
    )
    trace = make_trace(events, duration_us=200_000, gpu_kernel_time_us=40_000)

    diag = diagnose(trace)

    assert diag.verdict == Verdict.CHECKPOINT_BOUND
    assert 0.55 <= diag.confidence <= 0.98


def test_kernel_launch_bound_verdict() -> None:
    """Many sub-50µs kernels at 2% utilisation → KERNEL_LAUNCH_BOUND.

    100 tiny 20µs kernels spaced 1ms apart give a tiny-kernel ratio of 100%
    and an average duration of 20µs. With gpu_utilization = 2000/100000 = 2%
    the kernel-launch-bound heuristic fires. Duration is 100ms and event count
    is 100, so the small-trace guard does not activate.
    """
    events = [make_kernel_event(ts=i * 1000, dur=20) for i in range(100)]
    trace = make_trace(events, duration_us=100_000, gpu_kernel_time_us=2_000)

    diag = diagnose(trace)

    assert diag.verdict == Verdict.KERNEL_LAUNCH_BOUND


def test_sync_bound_verdict() -> None:
    """Low GPU util with sync ops ending just before idle gaps → SYNC_BOUND.

    sync_25 fires after checkpoint rules and before kernel_launch_tiny.
    A 40ms kernel at [0, 40ms] leaves [40ms, 200ms] idle. An aten::item call
    ends exactly when the kernel ends (0µs gap — within the 500µs lookahead),
    so the full 160ms idle is attributed to sync (sync_fraction = 1.0).
    avg_kernel_dur = 40ms >= 50µs so the kernel-launch guard does not block.
    """
    k_dur = 40_000  # 40ms
    events = [
        make_kernel_event(ts=0, dur=k_dur),
        # aten::item ends at k_dur — exactly when the GPU goes idle (0µs gap)
        make_cpu_event("aten::item", ts=k_dur - 100, dur=100),
        make_cpu_event("aten::_dummy_filler", ts=k_dur, dur=160_000),
    ]
    trace = make_trace(events, duration_us=200_000, gpu_kernel_time_us=k_dur)

    diag, stats = diagnose_with_stats(trace)

    assert diag.verdict == Verdict.SYNC_BOUND
    assert stats["sync_fraction"] >= 0.90
    assert 0.85 <= diag.confidence <= 0.98
    assert stats["sync_fraction"] >= 0.25
    fired = [d for d in stats["decisions"] if d["fired"]]
    assert fired[0]["rule"] == "sync_25"
    assert fired[0]["value"] == pytest.approx(stats["sync_fraction"])
    assert fired[0]["threshold"] == 0.25
    assert ".item()" in diag.summary
    assert len(diag.recommended_actions) == 4


# ---------------------------------------------------------------------------
# Ambiguous / degenerate cases
# ---------------------------------------------------------------------------


def test_unknown_verdict() -> None:
    """Low GPU util but no recognisable bottleneck pattern → UNKNOWN.

    'SomeRandomOp' doesn't match DataLoader, NCCL, or Memcpy patterns, so every
    overlap bucket comes back zero.  The engine falls through to the UNKNOWN
    branch because top_us / idle_us < 0.20.
    """
    events = [
        make_kernel_event(ts=0, dur=200),
        make_cpu_event("SomeRandomOp", ts=200, dur=800),
    ]
    trace = make_trace(events, duration_us=1000, gpu_kernel_time_us=200)

    diag = diagnose(trace)

    assert diag.verdict == Verdict.UNKNOWN


def test_hol_blocking_detected() -> None:
    """One 500ms DataLoader event among 99 fast 5ms events triggers HoL blocking.

    _hol_blocking_stats uses method='higher' for p99, so it returns the actual
    observed 99th-percentile sample (500_000 us), not a linear interpolation.
    ratio = 500_000 / 5_000 = 100 >> 10. The outlier recommendation should be
    prepended as the first recommended action.
    """
    # 99 fast DataLoader events at 5ms each
    fast_events = [
        make_cpu_event("DataLoader__next_data", ts=200 + i * 5_000, dur=5_000)
        for i in range(99)
    ]
    # 1 slow DataLoader event at 500ms — the head-of-line blocker
    slow_event = make_cpu_event("DataLoader__next_data", ts=200, dur=500_000)

    events = [make_kernel_event(ts=0, dur=200), slow_event, *fast_events]
    trace = make_trace(events, duration_us=1_000_000, gpu_kernel_time_us=200)

    diag = diagnose(trace)

    assert diag.verdict == Verdict.DATALOADER_BOUND
    assert diag.metrics.get("hol_blocking_likely") is True
    assert diag.recommended_actions[0].startswith(HOL_ACTION_PREFIX)


def test_hol_blocking_not_detected() -> None:
    """Uniform DataLoader events produce a HoL ratio of 1.0, well below threshold.

    All 20 events are identical 10ms fetches, so p99 == median and the ratio is 1.
    hol_blocking_likely must be False and the outlier action must not be prepended.
    """
    events = [
        make_kernel_event(ts=0, dur=200),
        *[
            make_cpu_event("DataLoader__next_data", ts=200 + i * 10_000, dur=10_000)
            for i in range(20)
        ],
    ]
    trace = make_trace(events, duration_us=300_000, gpu_kernel_time_us=200)

    diag = diagnose(trace)

    assert diag.verdict == Verdict.DATALOADER_BOUND
    assert diag.metrics.get("hol_blocking_likely") is False
    assert not diag.recommended_actions[0].startswith(HOL_ACTION_PREFIX)


def test_empty_trace() -> None:
    """A Trace with no events should return UNKNOWN without raising.

    The diagnose() function has an early-exit guard for empty event lists.
    This test verifies the engine degrades gracefully instead of crashing with
    an IndexError or ZeroDivisionError.
    """
    trace = Trace(events=[], duration_us=0, gpu_kernel_time_us=0, cpu_time_us=0)

    diag = diagnose(trace)

    assert diag.verdict == Verdict.UNKNOWN


# ---------------------------------------------------------------------------
# Decision log
# ---------------------------------------------------------------------------


def test_decision_log_present() -> None:
    """diagnose_with_stats returns a non-empty decisions list."""
    trace = make_trace(
        events=[make_kernel_event(ts=0, dur=950)],
        duration_us=1000,
        gpu_kernel_time_us=950,
    )
    diag, stats = diagnose_with_stats(trace)

    assert "decisions" in stats
    assert len(stats["decisions"]) > 0


def test_checkpoint_not_fired_on_tiny_trace() -> None:
    """Tiny traces (<100ms, <100 events) must not produce CHECKPOINT_BOUND.

    Regression test for the false positive seen on edge_tiny_trace.json, where
    profiler overhead events ('Runtime Triggered Module Loading', 'Lazy Function
    Loading') were mistakenly included in CHECKPOINT_PATTERNS and triggered a
    confident CHECKPOINT_BOUND verdict on a 3-event warm-up trace.
    The small-trace guard (< 100 events or < 100 ms) must intercept first and
    return UNKNOWN regardless of which patterns are present.
    """
    events = [
        make_kernel_event(ts=0, dur=100),
        make_cpu_event("aten::linear", ts=100, dur=200),
        make_cpu_event("aten::relu", ts=300, dur=100),
    ]
    # 3 events, 400 µs — far below both small-trace thresholds.
    trace = make_trace(events, duration_us=400, gpu_kernel_time_us=100)

    diag = diagnose(trace)

    assert diag.verdict == Verdict.UNKNOWN, (
        f"Expected UNKNOWN for tiny trace, got {diag.verdict} "
        f"(confidence={diag.confidence:.2f})"
    )
    assert diag.verdict != Verdict.CHECKPOINT_BOUND


def test_checkpoint_no_false_positive_on_module_loading_noise() -> None:
    """Traces with 'Runtime Triggered Module Loading' noise but no real torch.save
    must NOT fire CHECKPOINT_BOUND.

    Regression test for the bug where PyTorch's lazy-import overhead was
    misclassified as checkpoint activity in cuda_sync_stalls.json. Checkpoint
    now requires >=25% idle-window overlap on checkpoint name patterns.
    """
    # Util ~50%: kernel = 100ms, idle = 100ms. Many module-loading events
    # cover most of idle, but zero DtoH-Pageable memcpys and zero aten::copy_
    # or torch.save events.
    events: list[Event] = [make_kernel_event(ts=0, dur=100_000)]
    for i in range(40):
        events.append(
            make_cpu_event(
                "Runtime Triggered Module Loading",
                ts=100_000 + i * 2_000,
                dur=2_000,
            )
        )
    for i in range(10):
        events.append(
            make_cpu_event(
                "Lazy Function Loading",
                ts=180_000 + i * 2_000,
                dur=2_000,
            )
        )
    trace = make_trace(events, duration_us=200_000, gpu_kernel_time_us=100_000)

    diag = diagnose(trace)

    assert diag.verdict != Verdict.CHECKPOINT_BOUND, (
        f"Module-loading noise misclassified as checkpoint: {diag.verdict} "
        f"(confidence={diag.confidence:.2f})"
    )


def test_gpu_only_profile_returns_healthy_when_busy() -> None:
    """CUDA-only profile with a busy GPU should return HEALTHY, not UNKNOWN.

    Regression test for edge_gpu_only.json, which had only 'kernel' and
    'cuda_runtime' events and a large pre-profile gap that inflated wall-clock
    duration to 120 ms while kernels only ran for 2 ms. After the ingest fix,
    duration_us is computed against the GPU active span, giving ~90% util.

    This test constructs a Trace whose duration_us is already set to the GPU
    active span (simulating what the fixed load_trace produces) and verifies
    that diagnose() returns HEALTHY.
    """
    # 100 consecutive kernel events each running 900 µs out of every 1000 µs slot.
    # GPU active span = 100 × 1000 = 100 000 µs; gpu_kernel_time = 100 × 900 = 90 000 µs.
    # util = 90 000 / 100 000 = 90 % → healthy_85 fires.
    # Using exactly 100 events and 100 000 µs so neither small-trace threshold triggers.
    events = [
        Event(
            name="volta_sgemm",
            category="kernel",
            pid=1,
            tid=1,
            ts=i * 1000,
            dur=900,
        )
        for i in range(100)
    ]
    trace = make_trace(events, duration_us=100_000, gpu_kernel_time_us=90_000)

    diag = diagnose(trace)

    assert diag.verdict == Verdict.HEALTHY, (
        f"Expected HEALTHY for busy GPU-only profile, got {diag.verdict} "
        f"(util={diag.metrics.get('gpu_util', 0):.0%}, conf={diag.confidence:.2f})"
    )


def test_decision_log_one_rule_fired() -> None:
    """Exactly one rule fires per diagnosis (the one that determined the verdict)."""
    trace = make_trace(
        events=[make_kernel_event(ts=0, dur=950)],
        duration_us=1000,
        gpu_kernel_time_us=950,
    )
    diag, stats = diagnose_with_stats(trace)

    fired = [d for d in stats["decisions"] if d["fired"]]
    assert len(fired) == 1, f"Expected exactly 1 fired rule, got {len(fired)}: {fired}"


# ---------------------------------------------------------------------------
# Ground-truth regression tests — the two failures the harness confirmed,
# plus a property test that locks dominant-cause selection. All built from
# in-memory factories so they run on the no-GPU box.
# ---------------------------------------------------------------------------


def test_dataloader_dominant_beats_tiny_kernels() -> None:
    """Ground-truth failure #1: a dataloader-dominated trace with tiny kernels
    must be DATALOADER_BOUND, not KERNEL_LAUNCH_BOUND.

    Tiny kernels are a SYMPTOM of GPU starvation, not the root cause. When
    DataLoader stalls dominate idle time (dl_share ~0.86), the verdict must
    name the cause (DataLoader), not the symptom. The old first-match cascade
    evaluated kernel_launch_tiny before dataloader_fallback and mislabelled it.
    """
    # 20 back-to-back 30µs kernels -> tiny_kernel_ratio=1.0, avg=30µs, low util.
    # GPU busy [0, 600µs]; idle [600µs, 6600µs] = 6000µs.
    kernels = [make_kernel_event(ts=i * 30, dur=30) for i in range(20)]
    # DataLoader covers 86% of idle; a non-suspect filler covers the rest.
    dataloader = make_cpu_event("DataLoader__next_data", ts=600, dur=5_160)
    filler = make_cpu_event("aten::_dummy_filler", ts=5_760, dur=840)
    trace = make_trace(
        [*kernels, dataloader, filler], duration_us=6_600, gpu_kernel_time_us=600
    )

    diag, stats = diagnose_with_stats(trace)

    # The symptom is present (kernel_launch_tiny would fire in isolation)...
    assert stats["tiny_kernel_ratio"] > 0.50
    assert stats["util"] < 0.60
    # ...but DataLoader dominates idle, so the cause must win over the symptom.
    assert stats["dataloader_us"] / max(stats["idle_us"], 1) > 0.80
    assert diag.verdict == Verdict.DATALOADER_BOUND
    assert diag.verdict != Verdict.KERNEL_LAUNCH_BOUND


def test_sync_dominant_not_called_healthy() -> None:
    """Ground-truth failure #2: a sync-dominated trace at borderline util must
    be SYNC_BOUND, not HEALTHY.

    sync_fraction ~0.88 at util ~0.69. The old borderline_healthy_45 gate's
    no_dominant check ignored sync_fraction and fired HEALTHY before sync_25
    could run. The strengthened no_dominant refuses to call a sync-dominated
    trace healthy, and the dominant-cause competition then selects SYNC.
    """
    k = 345_000  # avg_kernel_dur = 345ms >= 50µs sync guard
    a = 617_800  # 2nd kernel start; the idle gap [345k, 617.8k] is sync-attributed
    events = [
        make_kernel_event(ts=0, dur=k),
        make_kernel_event(ts=a, dur=k),
        # aten::item ends exactly at the 1st kernel's end (0µs gap, within the
        # 500µs lookahead) so the following idle interval is attributed to sync.
        make_cpu_event("aten::item", ts=k - 100, dur=100),
        # Trailing filler extends the trace; its idle gap has NO preceding sync,
        # so sync_fraction lands at ~0.88 rather than 1.0.
        make_cpu_event("aten::_dummy_filler", ts=a + k, dur=1_000_000 - (a + k)),
    ]
    trace = make_trace(events, duration_us=1_000_000, gpu_kernel_time_us=2 * k)

    diag, stats = diagnose_with_stats(trace)

    assert stats["util"] == pytest.approx(0.69, abs=0.01)
    assert stats["sync_fraction"] > 0.80  # sync dominates idle
    assert diag.verdict == Verdict.SYNC_BOUND
    assert diag.verdict != Verdict.HEALTHY


def _sync_dominant_trace() -> Trace:
    """Low-util trace where sync stalls are the only cause above threshold."""
    k = 100_000  # avg_kernel_dur >= 50µs guard
    events = [
        make_kernel_event(ts=0, dur=k),
        make_cpu_event("aten::item", ts=k - 100, dur=100),  # ends at kernel end
        make_cpu_event("aten::_dummy_filler", ts=k, dur=400_000),  # idle, synced
    ]
    return make_trace(events, duration_us=500_000, gpu_kernel_time_us=k)


@pytest.mark.parametrize(
    "builder,expected_verdict",
    [
        (lambda: trace_with_dataloader_share(0.10, 0.50), Verdict.DATALOADER_BOUND),
        (lambda: trace_with_nccl_share(0.10, 0.50), Verdict.NCCL_BOUND),
        (lambda: trace_with_memcpy_ratio(0.70), Verdict.PCIE_BOUND),
        (
            lambda: trace_with_checkpoint_share(0.20, 0.40, dtoh_count=100),
            Verdict.CHECKPOINT_BOUND,
        ),
        (_sync_dominant_trace, Verdict.SYNC_BOUND),
    ],
)
def test_single_dominant_cause_wins(builder, expected_verdict) -> None:
    """Property: when exactly ONE idle cause is above its firing threshold and
    all others are ~0, the dominant-cause competition returns that cause.

    Sweeps all five idle causes (dataloader, nccl, pcie, checkpoint, sync) so a
    future change to the selection logic cannot silently mis-route a clean,
    single-cause trace.
    """
    diag, stats = diagnose_with_stats(builder())
    assert (
        diag.verdict == expected_verdict
    ), f"Expected {expected_verdict}, got {diag.verdict} (rule={stats.get('rule')})"
    # Exactly the one winning rule fired.
    fired = [d for d in stats["decisions"] if d["fired"]]
    assert len(fired) == 1, f"Expected exactly 1 fired rule, got {fired}"


# ---------------------------------------------------------------------------
# PCIE_BOUND active-time floor — regression for a false-positive the expanded
# ground-truth harness surfaced. pcie_ratio = gpu_memcpy_time_us / gpu_active_us
# is active-normalized, so a near-idle GPU (tiny gpu_active_us) makes the ratio a
# meaningless artifact of two small numbers and used to fire PCIE_BOUND even when
# sync/dataloader own the idle. The _PCIE_MIN_ACTIVE_US floor guards that.
# ---------------------------------------------------------------------------


def test_pcie_not_fired_when_gpu_barely_active() -> None:
    """gpu_active below the floor + a sync-dominated idle MUST be SYNC_BOUND.

    The GPU does almost no work (gpu_active_us ~920, under _PCIE_MIN_ACTIVE_US)
    yet gpu_memcpy is most of it, so pcie_ratio ~0.87 — high purely because the
    denominator is tiny. Meanwhile sync stalls own half the idle (a SPECIFIC
    cause that fires) and a DataLoader spans ~43% of it. Pre-fix, pcie_ratio_50
    fired and — because pcie_ratio (0.87) > sync_fraction (0.50) — WON the
    dominant-cause competition: a false PCIE_BOUND. The active-time floor removes
    pcie from contention, so the real idle cause (sync) wins, beating both the
    blocked pcie rule and the generic DataLoader fallback.

    NOTE on the numbers: the harness's raw finding was gpu_active_us=40, but the
    sync rule's avg_kernel_dur >= 50us guard cannot be satisfied with ~20us of
    kernel time, so SYNC_BOUND would be unreachable there. We therefore use the
    smallest self-consistent active time that (a) stays under the 1000us floor
    and (b) lets sync legitimately fire (two 60us kernels -> avg 60us) — same
    bug, a trace the engine can actually resolve to SYNC.
    """
    # gpu_kernel_time_us field = 120 (= two 60us kernels, consistent with events).
    # gpu_memcpy = 800us -> gpu_active = 920us (< 1000 floor); pcie_ratio = 0.87.
    events = [
        make_kernel_event(ts=0, dur=60),  # busy [0, 60]
        Event(
            name="Memcpy HtoD",
            category="gpu_memcpy",
            pid=1,
            tid=1,
            ts=60,
            dur=800,  # busy [60, 860]; the only "active" work besides 2 tiny kernels
        ),
        # aten::item ends exactly at idle-1 start (0us gap, within the 500us
        # lookahead) so the first idle interval is attributed to sync.
        make_cpu_event("aten::item", ts=810, dur=50),  # ends at 860
        # A DataLoader spanning ~43% of idle — the generic fallback that sync,
        # a specific cause, must still beat.
        make_cpu_event("DataLoader__next_data", ts=860, dur=42_604),
        # A second kernel splits the idle so a trailing, NON-sync-preceded idle
        # interval exists -> sync_fraction lands at ~0.50 (not 1.0).
        make_kernel_event(ts=50_400, dur=60),  # busy [50400, 50460]
        make_cpu_event("aten::_dummy_filler", ts=50_460, dur=49_540),  # ends 100000
    ]
    trace = make_trace(events, duration_us=100_000, gpu_kernel_time_us=120)

    diag, stats = diagnose_with_stats(trace)

    # The bug's preconditions hold: a high pcie_ratio from a sub-floor active time.
    assert stats["gpu_active_us"] < _PCIE_MIN_ACTIVE_US
    pcie_ratio = stats["gpu_memcpy_time_us"] / max(stats["gpu_active_us"], 1)
    assert pcie_ratio >= 0.50, f"expected pcie_ratio >= 0.50, got {pcie_ratio:.2f}"
    # ...but sync genuinely owns the idle (and a DataLoader is present too).
    assert stats["sync_fraction"] >= 0.25
    assert stats["dataloader_us"] > 0

    # The floor blocks pcie, so the real idle cause wins.
    assert (
        diag.verdict == Verdict.SYNC_BOUND
    ), f"got {diag.verdict} (rule={stats['rule']})"
    assert diag.verdict != Verdict.PCIE_BOUND
    # pcie_ratio_50 neither won nor even passed its (now floored) firing condition.
    pcie_decision = next(d for d in stats["decisions"] if d["rule"] == "pcie_ratio_50")
    assert pcie_decision["fired"] is False
    assert pcie_decision["passed"] is False
    fired = [d for d in stats["decisions"] if d["fired"]]
    assert len(fired) == 1 and fired[0]["rule"] == "sync_25"


def test_pcie_still_fires_with_real_active_time() -> None:
    """gpu_active comfortably above the floor + pcie_ratio >= 0.50 -> PCIE_BOUND.

    The no-regression counterpart: when the GPU actually did meaningful work the
    ratio is trustworthy and the floor must not interfere. Here gpu_active_us is
    2000us (>= the 1000us floor) with memcpy = kernel, so pcie_ratio = 0.50 and
    PCIE_BOUND fires exactly as before.
    """
    events = [
        make_kernel_event(ts=0, dur=1_000),  # busy [0, 1000]
        Event(
            name="Memcpy HtoD",
            category="gpu_memcpy",
            pid=1,
            tid=1,
            ts=1_000,
            dur=1_000,  # busy [1000, 2000]
        ),
        make_cpu_event("aten::_dummy_filler", ts=2_000, dur=98_000),  # idle, no cause
    ]
    trace = make_trace(events, duration_us=100_000, gpu_kernel_time_us=1_000)

    diag, stats = diagnose_with_stats(trace)

    assert stats["gpu_active_us"] >= _PCIE_MIN_ACTIVE_US
    pcie_ratio = stats["gpu_memcpy_time_us"] / max(stats["gpu_active_us"], 1)
    assert pcie_ratio >= 0.50
    assert (
        diag.verdict == Verdict.PCIE_BOUND
    ), f"got {diag.verdict} (rule={stats['rule']})"
    fired = [d for d in stats["decisions"] if d["fired"]]
    assert len(fired) == 1 and fired[0]["rule"] == "pcie_ratio_50"
