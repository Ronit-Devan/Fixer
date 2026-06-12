"""Synthetic boundary tests for the gpu_doctor_engine diagnostic rules.

Six categories encode confirmed engine behaviour from probe data so future
detector additions cannot silently regress thresholds.

Category 1 – Empty / pathological inputs (6 tests)
Category 2 – HEALTHY threshold boundaries at 85 / 70 / 45 % (8 tests)
Category 3 – Bottleneck threshold boundaries: PCIE 0.50, checkpoint 0.25,
             specific_30 0.30, dataloader 0.20, kernel_launch (8 tests)
Category 4 – Multi-bottleneck priority: specific beats generic (5 tests)
Category 5 – Warmup guard four-condition AND boundary (6 tests)
Category 6 – Decision log invariants (4 tests)
"""

from __future__ import annotations

import pytest

from gpu_doctor_engine.diagnose import diagnose, diagnose_with_stats
from gpu_doctor_engine.types import Event, Trace, Verdict

from tests.helpers import (
    make_cpu_event,
    make_kernel_event,
    make_trace,
    trace_at_util,
    trace_warmup,
    trace_with_checkpoint_share,
    trace_with_dataloader_share,
    trace_with_memcpy_ratio,
    trace_with_nccl_share,
    trace_with_tiny_kernels,
)


# ---------------------------------------------------------------------------
# Category 1: Empty / pathological inputs
# ---------------------------------------------------------------------------


def test_empty_trace_returns_unknown() -> None:
    """Empty Trace returns UNKNOWN gracefully, no crash."""
    trace = Trace(events=[], duration_us=0, gpu_kernel_time_us=0, cpu_time_us=0)
    diag = diagnose(trace)
    assert diag.verdict == Verdict.UNKNOWN


def test_zero_duration_returns_unknown() -> None:
    """Trace with duration_us=0 returns UNKNOWN, no division by zero."""
    # gpu_utilization property guards against /0; warmup fires (all four hold).
    events = [make_kernel_event(ts=0, dur=1_000)]
    trace = Trace(events=events, duration_us=0, gpu_kernel_time_us=0, cpu_time_us=0)
    diag = diagnose(trace)
    assert diag.verdict == Verdict.UNKNOWN


def test_single_kernel_high_util_returns_healthy() -> None:
    """A single high-util kernel still produces HEALTHY (matches engine behavior)."""
    # util=0.95 >= 0.85 → healthy_85 fires; warmup 4th condition (util<0.85) fails.
    events = [make_kernel_event(ts=0, dur=950)]
    trace = make_trace(events, duration_us=1_000, gpu_kernel_time_us=950)
    diag = diagnose(trace)
    assert diag.verdict == Verdict.HEALTHY
    assert diag.confidence is None


def test_only_cpu_events_no_kernels() -> None:
    """Trace with only CPU events (no kernels) returns UNKNOWN."""
    # duration=200ms bypasses warmup (condition 3: duration<50ms fails).
    # util=0, no patterns → falls through to UNKNOWN.
    events = [make_cpu_event("aten::linear", ts=0, dur=200_000)]
    trace = make_trace(events, duration_us=200_000, gpu_kernel_time_us=0)
    diag = diagnose(trace)
    assert diag.verdict == Verdict.UNKNOWN


def test_zero_kernel_time_with_events() -> None:
    """Events present but gpu_kernel_time_us = 0 returns UNKNOWN."""
    # Field set to 0 → util=0; warmup skipped (duration=200ms ≥ 50ms).
    events = [
        make_kernel_event(ts=0, dur=1_000),
        make_cpu_event("aten::linear", ts=1_000, dur=199_000),
    ]
    trace = make_trace(events, duration_us=200_000, gpu_kernel_time_us=0)
    diag = diagnose(trace)
    assert diag.verdict == Verdict.UNKNOWN


def test_very_long_unicode_event_name() -> None:
    """Event with 10,000-char unicode name doesn't crash pattern matching."""
    long_name = "\u03b1" * 10_000  # 'α' × 10 000
    events = [
        make_kernel_event(ts=0, dur=200_000),
        Event(name=long_name, category="cpu_op", pid=0, tid=0, ts=200_000, dur=800_000),
    ]
    trace = make_trace(events, duration_us=1_000_000, gpu_kernel_time_us=200_000)
    diag = diagnose(trace)
    assert diag.verdict is not None  # any verdict is fine; must not raise


# ---------------------------------------------------------------------------
# Category 2: HEALTHY threshold boundaries
# ---------------------------------------------------------------------------


def test_healthy_at_exactly_85_percent() -> None:
    """util=0.85 fires healthy_85 with no confidence score."""
    diag = diagnose(trace_at_util(0.85))
    assert diag.verdict == Verdict.HEALTHY
    assert diag.confidence is None


def test_healthy_just_below_85_uses_70_rule() -> None:
    """util=0.8499 falls to healthy_70_no_dominant tier."""
    diag = diagnose(trace_at_util(0.8499))
    assert diag.verdict == Verdict.HEALTHY
    assert diag.confidence is None


def test_healthy_at_exactly_70_percent() -> None:
    """util=0.70 fires healthy_70_no_dominant."""
    diag = diagnose(trace_at_util(0.70))
    assert diag.verdict == Verdict.HEALTHY
    assert diag.confidence is None


def test_healthy_just_below_70_uses_45_rule() -> None:
    """util=0.6999 falls to borderline_healthy_45 tier."""
    diag = diagnose(trace_at_util(0.6999))
    assert diag.verdict == Verdict.HEALTHY
    assert diag.confidence is None


def test_healthy_at_exactly_45_percent() -> None:
    """util=0.45 fires borderline_healthy_45."""
    diag = diagnose(trace_at_util(0.45))
    assert diag.verdict == Verdict.HEALTHY
    assert diag.confidence is None


def test_just_below_45_falls_through() -> None:
    """util=0.4499 with no suspect patterns falls through to UNKNOWN."""
    diag = diagnose(trace_at_util(0.4499))
    assert diag.verdict == Verdict.UNKNOWN


def test_high_util_with_minor_dataloader_still_healthy() -> None:
    """util=0.90 with 30% dataloader share still HEALTHY (high util wins)."""
    # healthy_85 fires (step 1) before dataloader_fallback (step 7).
    diag = diagnose(trace_with_dataloader_share(0.90, 0.30))
    assert diag.verdict == Verdict.HEALTHY
    assert diag.confidence is None


def test_mid_util_with_dominant_dataloader_is_bottleneck() -> None:
    """util=0.50, dl_share=0.70: healthy rules need no dominant suspect, DATALOADER_BOUND fires."""
    # borderline_healthy_45 requires dl_share < 0.40; 0.70 ≥ 0.40 blocks it.
    diag = diagnose(trace_with_dataloader_share(0.50, 0.70))
    assert diag.verdict == Verdict.DATALOADER_BOUND


# ---------------------------------------------------------------------------
# Category 3: Bottleneck threshold boundaries
# ---------------------------------------------------------------------------


def test_pcie_at_exactly_50_percent_ratio() -> None:
    """memcpy_ratio=0.50 fires PCIE_BOUND."""
    diag = diagnose(trace_with_memcpy_ratio(0.50))
    assert diag.verdict == Verdict.PCIE_BOUND


def test_pcie_just_below_50_falls_through() -> None:
    """memcpy_ratio=0.4999 does NOT fire PCIE_BOUND (low util causes UNKNOWN or other)."""
    # The 4× idle gap in trace_with_memcpy_ratio keeps cat_memcpy_us/idle_us < 0.30
    # so specific_30 also cannot fire via the memcpy path.
    diag = diagnose(trace_with_memcpy_ratio(0.4999))
    assert diag.verdict != Verdict.PCIE_BOUND


def test_checkpoint_at_exactly_25_percent_share() -> None:
    """checkpoint_share=0.25 of idle fires CHECKPOINT_BOUND."""
    # util=0.20 < 0.45 keeps borderline_healthy_45 silent; ckpt fires at step 4.
    diag = diagnose(trace_with_checkpoint_share(0.20, 0.25))
    assert diag.verdict == Verdict.CHECKPOINT_BOUND


def test_checkpoint_just_below_25_falls_through() -> None:
    """checkpoint_share=0.2499 does NOT fire CHECKPOINT_BOUND."""
    diag = diagnose(trace_with_checkpoint_share(0.20, 0.2499, dtoh_count=0))
    assert diag.verdict != Verdict.CHECKPOINT_BOUND


def test_checkpoint_dtoh_burst_below_share_does_not_fire() -> None:
    """DtoH-Pageable count alone must not fire CHECKPOINT_BOUND."""
    diag = diagnose(trace_with_checkpoint_share(0.20, 0.05, dtoh_count=150))
    assert diag.verdict != Verdict.CHECKPOINT_BOUND


def test_dataloader_at_exactly_20_percent_share() -> None:
    """dl_share=0.20 of idle fires DATALOADER_BOUND."""
    diag = diagnose(trace_with_dataloader_share(0.10, 0.20))
    assert diag.verdict == Verdict.DATALOADER_BOUND


def test_dataloader_just_below_20_returns_unknown() -> None:
    """dl_share=0.19 with low util and no other patterns returns UNKNOWN."""
    diag = diagnose(trace_with_dataloader_share(0.10, 0.19))
    assert diag.verdict == Verdict.UNKNOWN


def test_nccl_bound_30_threshold() -> None:
    """nccl_share=0.30 of idle fires nccl_bound_30 -> NCCL_BOUND."""
    # util=0.20 < 0.45 keeps all healthy rules silent; nccl_share=0.30 hits nccl_bound_30.
    diag = diagnose(trace_with_nccl_share(0.20, 0.30))
    assert diag.verdict == Verdict.NCCL_BOUND


def test_kernel_launch_thresholds() -> None:
    """tiny_ratio>0.5 AND avg_dur<100us AND util<0.6 -> KERNEL_LAUNCH_BOUND."""
    # 20 kernels of 30 µs with 200 µs gaps: tiny_ratio=1.0, avg=30 µs, util≈13%.
    diag = diagnose(trace_with_tiny_kernels(n_kernels=20, kernel_dur_us=30, gap_us=200))
    assert diag.verdict == Verdict.KERNEL_LAUNCH_BOUND


# ---------------------------------------------------------------------------
# Category 4: Multi-bottleneck priority
# ---------------------------------------------------------------------------


def test_pcie_beats_dataloader_when_both_present() -> None:
    """High memcpy_ratio AND high dl_share: PCIE_BOUND wins (specific > generic)."""
    # pcie_ratio_50 fires at step 3; dataloader_fallback is step 7.
    # kernel=200ms, gpu_memcpy=200ms → pcie_ratio=0.50.  DataLoader covers all idle.
    events = [
        make_kernel_event(ts=0, dur=200_000),
        Event(
            name="gpu_memcpy_htod",
            category="gpu_memcpy",
            pid=1,
            tid=1,
            ts=200_000,
            dur=200_000,
        ),
        make_cpu_event("DataLoader__next_data", ts=400_000, dur=600_000),
    ]
    trace = make_trace(events, duration_us=1_000_000, gpu_kernel_time_us=200_000)
    diag = diagnose(trace)
    assert diag.verdict == Verdict.PCIE_BOUND


def test_checkpoint_beats_dataloader_when_both_present() -> None:
    """High checkpoint share AND high dl_share: CHECKPOINT_BOUND wins."""
    # checkpoint_25 fires at step 4 (before dataloader_fallback at step 7).
    # kernel=100ms; torch.save covers 25% of 900ms idle; DataLoader covers 60%.
    events = [
        make_kernel_event(ts=0, dur=100_000),
        make_cpu_event("torch.save", ts=100_000, dur=225_000),
        make_cpu_event("DataLoader__next_data", ts=325_000, dur=540_000),
        make_cpu_event("aten::_dummy_filler", ts=865_000, dur=135_000),
    ]
    trace = make_trace(events, duration_us=1_000_000, gpu_kernel_time_us=100_000)
    diag = diagnose(trace)
    assert diag.verdict == Verdict.CHECKPOINT_BOUND


def test_sync_25_fires_before_kernel_launch() -> None:
    """sync_25 (step 4c) wins over kernel_launch_tiny when both conditions could apply.

    20 kernels × 60µs each (avg=60 >= 50 → not guarded out), with aten::item ending
    at each kernel boundary (0µs gap → within 500µs lookahead). sync_fraction = 100%.
    kernel_launch_tiny would need tiny_ratio > 0.5 AND avg_dur < 100 AND util < 0.60;
    but with avg_dur=60 and sync_25 firing first at step 4c, it stays SYNC_BOUND.
    """
    n, k_dur, gap = 20, 60, 400  # avg_kernel_dur = 60µs >= 50 guard
    slot = k_dur + gap
    events = [make_kernel_event(ts=i * slot, dur=k_dur) for i in range(n)]
    # aten::item ending at each kernel's end — 0µs gap to idle start (within 500µs)
    for i in range(n):
        kernel_end = i * slot + k_dur
        events.append(make_cpu_event("aten::item", ts=kernel_end - 50, dur=50))
    trace = make_trace(events, duration_us=n * slot, gpu_kernel_time_us=n * k_dur)
    diag = diagnose(trace)
    assert diag.verdict == Verdict.SYNC_BOUND


def test_dataloader_beats_kernel_launch_when_both_present() -> None:
    """Trace with both tiny kernels AND a dominant DataLoader: DATALOADER_BOUND wins.

    Tiny kernels are a SYMPTOM of GPU starvation, not the root cause, so when
    DataLoader stalls dominate the idle time the dominant-cause competition
    must name the cause (DataLoader), not the symptom (kernel-launch overhead).

    NOTE: this test previously asserted the inverted precedence
    (KERNEL_LAUNCH_BOUND, with the rationale "kernel_launch_tiny fires at step
    5; dataloader_fallback is step 7") — i.e. it encoded the first-match-wins
    ordering that WAS Bug 1. It was updated when the dominant-cause competition
    replaced first-match-wins; its failure under the old assertion was the
    intended fix, not a regression.
    """
    # 20 × 30 µs kernels with 200 µs gaps; one DataLoader event spanning all idle.
    n, k_dur, gap = 20, 30, 200
    slot = k_dur + gap  # 230 µs per slot
    events = [make_kernel_event(ts=i * slot, dur=k_dur) for i in range(n)]
    # DataLoader starts right after the first kernel and spans the whole trace.
    events.append(make_cpu_event("DataLoader__next_data", ts=k_dur, dur=n * slot))
    trace_end = k_dur + n * slot  # last event (DataLoader) determines trace_end
    trace = make_trace(events, duration_us=trace_end, gpu_kernel_time_us=n * k_dur)
    diag = diagnose(trace)
    assert diag.verdict == Verdict.DATALOADER_BOUND


def test_nccl_bound_30_beats_dataloader() -> None:
    """nccl_share=0.40 AND dl_share=0.60: NCCL_BOUND wins via nccl_bound_30 rule."""
    # nccl_bound_30 competes in dominant-cause stage; dataloader_fallback is fallback.
    # kernel=100ms; NCCL=360ms (40% of 900ms idle); DataLoader=540ms (60%).
    events = [
        make_kernel_event(ts=0, dur=100_000),
        make_cpu_event("ncclAllReduce", ts=100_000, dur=360_000),
        make_cpu_event("DataLoader__next_data", ts=460_000, dur=540_000),
    ]
    trace = make_trace(events, duration_us=1_000_000, gpu_kernel_time_us=100_000)
    diag = diagnose(trace)
    assert diag.verdict == Verdict.NCCL_BOUND


def test_pcie_and_nccl_present_nccl_bound_30_picks_winner() -> None:
    """Both nccl_share and memcpy share above 0.30: the higher one wins (test deterministic)."""
    # GPU setup: kernel=500ms, gpu_memcpy=130ms → pcie_ratio=0.21 (no pcie_ratio_50).
    # idle=370ms.  cat_memcpy_us=130ms (35% of idle).  NCCL=200ms (54% of idle).
    # nccl_bound_30 / pcie_idle_30 compete; NCCL wins (higher share); running twice must match.
    events = [
        make_kernel_event(ts=0, dur=500_000),
        Event(
            name="gpu_memcpy_htod",
            category="gpu_memcpy",
            pid=1,
            tid=1,
            ts=500_000,
            dur=130_000,
        ),
        make_cpu_event("ncclAllReduce", ts=630_000, dur=200_000),
        make_cpu_event("aten::_dummy_filler", ts=830_000, dur=170_000),
    ]
    trace = make_trace(events, duration_us=1_000_000, gpu_kernel_time_us=500_000)
    diag1 = diagnose(trace)
    diag2 = diagnose(trace)
    assert diag1.verdict == Verdict.NCCL_BOUND
    assert diag1.verdict == diag2.verdict


# ---------------------------------------------------------------------------
# Category 5: Warmup guard boundaries
# ---------------------------------------------------------------------------


def test_warmup_fires_when_all_four_conditions_hold() -> None:
    """3 kernels, gpu_active=300us, duration=20ms, util<0.85 -> warmup guard fires, UNKNOWN."""
    # All four conditions: n<5, gpu_active<5ms, duration<50ms, util<0.85.
    trace = trace_warmup(n_kernels=3, kernel_dur_us=100, total_us=20_000)
    diag = diagnose(trace)
    assert diag.verdict == Verdict.UNKNOWN


def test_warmup_skipped_when_kernel_count_high() -> None:
    """6 kernels but tiny otherwise: NOT warmup, proceeds to diagnose."""
    # n_kernels=6 ≥ 5 → condition 1 fails; warmup_trace_guard fires=False.
    trace = trace_warmup(n_kernels=6, kernel_dur_us=100, total_us=5_000)
    _, stats = diagnose_with_stats(trace)
    wd = next(d for d in stats["decisions"] if d["rule"] == "warmup_trace_guard")
    assert wd["fired"] is False


def test_warmup_skipped_when_gpu_active_long() -> None:
    """2 kernels but gpu_active=10ms: NOT warmup."""
    # gpu_kernel_time_us=10_000 ≥ 5_000 → condition 2 fails.
    events = [
        make_kernel_event(ts=0, dur=5_000),
        make_kernel_event(ts=5_000, dur=5_000),
    ]
    trace = make_trace(events, duration_us=20_000, gpu_kernel_time_us=10_000)
    _, stats = diagnose_with_stats(trace)
    wd = next(d for d in stats["decisions"] if d["rule"] == "warmup_trace_guard")
    assert wd["fired"] is False


def test_warmup_skipped_when_duration_long() -> None:
    """2 kernels, gpu_active<5ms, but duration=100ms: NOT warmup."""
    # duration_us=100_000 ≥ 50_000 → condition 3 fails.
    events = [make_kernel_event(ts=0, dur=500), make_kernel_event(ts=1_000, dur=500)]
    trace = make_trace(events, duration_us=100_000, gpu_kernel_time_us=1_000)
    _, stats = diagnose_with_stats(trace)
    wd = next(d for d in stats["decisions"] if d["rule"] == "warmup_trace_guard")
    assert wd["fired"] is False


def test_warmup_skipped_when_util_high() -> None:
    """1 kernel of 950us in 1ms (util=0.95): NOT warmup, returns HEALTHY."""
    # util=0.95 ≥ 0.85 → condition 4 fails; healthy_85 fires.
    events = [make_kernel_event(ts=0, dur=950)]
    trace = make_trace(events, duration_us=1_000, gpu_kernel_time_us=950)
    diag, stats = diagnose_with_stats(trace)
    wd = next(d for d in stats["decisions"] if d["rule"] == "warmup_trace_guard")
    assert wd["fired"] is False
    assert diag.verdict == Verdict.HEALTHY


def test_warmup_guard_logged_in_decisions() -> None:
    """warmup_trace_guard appears in decisions list for both fired=True and fired=False cases."""
    trace_w = trace_warmup(n_kernels=2, kernel_dur_us=100, total_us=5_000)
    _, stats_w = diagnose_with_stats(trace_w)
    fired_entry = next(
        (d for d in stats_w["decisions"] if d["rule"] == "warmup_trace_guard"), None
    )
    assert fired_entry is not None
    assert fired_entry["fired"] is True

    trace_n = trace_at_util(0.95)
    _, stats_n = diagnose_with_stats(trace_n)
    not_fired_entry = next(
        (d for d in stats_n["decisions"] if d["rule"] == "warmup_trace_guard"), None
    )
    assert not_fired_entry is not None
    assert not_fired_entry["fired"] is False


# ---------------------------------------------------------------------------
# Category 6: Decision log invariants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "builder,expected_verdict",
    [
        (lambda: trace_at_util(0.95), Verdict.HEALTHY),
        (lambda: trace_with_dataloader_share(0.10, 0.50), Verdict.DATALOADER_BOUND),
        (lambda: trace_with_memcpy_ratio(0.70), Verdict.PCIE_BOUND),
        (lambda: trace_with_checkpoint_share(0.20, 0.40), Verdict.CHECKPOINT_BOUND),
        (lambda: trace_with_nccl_share(0.20, 0.50), Verdict.NCCL_BOUND),
        (lambda: trace_with_tiny_kernels(20, 30, 200), Verdict.KERNEL_LAUNCH_BOUND),
    ],
)
def test_exactly_one_rule_fires_per_diagnosis(builder, expected_verdict) -> None:
    """For every non-UNKNOWN verdict, exactly one decision has fired=True."""
    trace = builder()
    diag, stats = diagnose_with_stats(trace)
    assert (
        diag.verdict == expected_verdict
    ), f"Expected {expected_verdict}, got {diag.verdict}"
    fired = [d for d in stats["decisions"] if d["fired"]]
    assert len(fired) == 1, f"Expected exactly 1 fired rule, got {len(fired)}: {fired}"


def test_decisions_have_all_required_fields() -> None:
    """Every decision dict has keys: rule, fired, value, threshold, note."""
    _, stats = diagnose_with_stats(trace_at_util(0.90))
    required = {"rule", "fired", "value", "threshold", "note"}
    for d in stats["decisions"]:
        assert required.issubset(d.keys()), f"Decision missing fields: {d}"


def test_decision_ordering_is_stable() -> None:
    """Same trace analyzed twice produces identical decision list (order and content)."""
    trace = trace_at_util(0.90)
    _, stats1 = diagnose_with_stats(trace)
    _, stats2 = diagnose_with_stats(trace)
    assert stats1["decisions"] == stats2["decisions"]


def test_unknown_verdict_has_no_fired_decisions() -> None:
    """When verdict is UNKNOWN (fallthrough), no decision has fired=True."""
    # util=0.30 < 0.45, no suspect patterns, duration=1s → normal fallthrough UNKNOWN.
    trace = trace_at_util(0.30)
    diag, stats = diagnose_with_stats(trace)
    assert diag.verdict == Verdict.UNKNOWN
    fired = [d for d in stats["decisions"] if d["fired"]]
    assert len(fired) == 0, f"Expected 0 fired decisions, got: {fired}"
