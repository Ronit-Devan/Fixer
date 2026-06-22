"""Wave A correctness fixes (from the product audit), each proven here.

1. FREE_STALE_CACHE issues a real `kill` and must honor the never-kill guard.
2. Recovery is judged only on telemetry sampled AFTER the fix (no contamination).
3. A minimum number of post-apply samples is required before confirm/rollback.
4. Circuit-breaker half-open trial applies are not counted toward flap/rate.
"""

from __future__ import annotations

from dataclasses import dataclass

from conftest import diag

from et_remediation import (
    ActionKind,
    BreakerState,
    CapsConfig,
    CircuitBreaker,
    FakeActuator,
    FakeTelemetryModel,
    OutcomeKind,
    RemediationConfig,
    RemediationManager,
    RemediationMode,
    default_registry,
)
from et_remediation.actions import PROCESS_TOUCHING_KINDS

K = "node-0"


@dataclass(frozen=True)
class TSample:
    """A telemetry sample carrying a timestamp (like the monitor's Snapshot)."""

    timestamp_s: float
    clock_ratio: float | None = None
    util_pct: float | None = None
    mem_used_ratio: float | None = None


def _mgr(model=None, *, recover=False, protected=None, vw=100.0, min_samples=2):
    cfg = RemediationConfig(
        mode=RemediationMode.AUTO, verify_window_s=vw, min_verify_samples=min_samples
    )
    if protected is not None:
        cfg.protected_pids = protected
    act = FakeActuator(model or FakeTelemetryModel(), recover_on_apply=recover)
    mgr = RemediationManager(default_registry(), cfg, [act], now_fn=lambda: 0.0)
    return mgr, act


# 1) FREE_STALE_CACHE never-kill hole -----------------------------------------


def test_free_stale_cache_is_process_touching():
    assert ActionKind.FREE_STALE_CACHE in PROCESS_TOUCHING_KINDS


def test_free_stale_cache_refused_when_target_is_workload():
    mgr, act = _mgr(
        FakeTelemetryModel(util_pct=30, mem_used_ratio=0.8), protected=[4242]
    )
    mgr.config.knobs["stale_pid"] = 4242  # the "stale" pid IS the protected workload
    out = mgr.observe(diag("unknown", metrics={"fragmentation_ratio": 0.5}), [], now=0.0)
    # The selection guard refuses it -> advised, and the kill never reaches the actuator.
    assert out.kind is OutcomeKind.ADVISED
    assert 4242 not in act.targeted_pids
    assert act.protection_violations == 0


# 2 + 3) verify only on post-apply samples, with a minimum count ---------------


def _thermal():
    return diag("thermal_throttle", metrics={"clock_ratio": 0.55})


def test_verify_ignores_pre_apply_samples_and_needs_minimum():
    mgr, act = _mgr(min_samples=2)
    # Apply at t=10; pre window is throttled.
    o0 = mgr.observe(_thermal(), [TSample(9, clock_ratio=0.55)], now=10.0)
    assert o0.kind is OutcomeKind.APPLIED

    # t=11: window mixes stale pre-apply lows (ts<=10) with ONE fresh high.
    # The pre-apply lows are filtered out; one post sample is below the minimum.
    o1 = mgr.observe(
        _thermal(),
        [TSample(9, clock_ratio=0.55), TSample(5, clock_ratio=0.55), TSample(11, clock_ratio=0.97)],
        now=11.0,
    )
    assert o1.kind is OutcomeKind.VERIFYING  # 1 fresh sample < min_verify_samples

    # t=12: now two post-apply highs -> recovery confirmed.
    o2 = mgr.observe(
        _thermal(),
        [TSample(11, clock_ratio=0.97), TSample(12, clock_ratio=0.97)],
        now=12.0,
    )
    assert o2.kind is OutcomeKind.CONFIRMED


def test_verify_does_not_confirm_on_pre_apply_data():
    # If only pre-apply samples are present, recovery is never (falsely) confirmed.
    mgr, act = _mgr(vw=5.0, min_samples=2)
    assert mgr.observe(_thermal(), [TSample(0, clock_ratio=0.55)], now=10.0).kind is OutcomeKind.APPLIED
    # Window has only stale samples (ts <= applied_at=10) that LOOK recovered.
    o = mgr.observe(_thermal(), [TSample(9, clock_ratio=0.99), TSample(8, clock_ratio=0.99)], now=11.0)
    assert o.kind is OutcomeKind.VERIFYING  # filtered out -> nothing to judge yet
    # Past the deadline with still no post-apply samples -> rollback, not confirm.
    o2 = mgr.observe(_thermal(), [TSample(9, clock_ratio=0.99)], now=16.0)
    assert o2.kind is OutcomeKind.ROLLED_BACK


# 4) circuit breaker: half-open trial not counted -----------------------------


def test_half_open_trial_applies_not_counted_as_flap():
    cb = CircuitBreaker(CapsConfig(failure_threshold=1, flap_threshold=2, breaker_cooldown_s=50))
    cb.record_failure(K, 0)  # trip OPEN
    cb.allow(K, 60)  # cooldown elapsed -> HALF_OPEN trial granted
    # Two trial applies of the same kind would hit flap_threshold under the old
    # accounting; as a half-open probe they must NOT be counted.
    cb.record_apply(K, "set_power_limit", 60)
    cb.record_apply(K, "set_power_limit", 60)
    assert cb.state(K) is BreakerState.HALF_OPEN  # not re-tripped by the probe


def test_success_clears_flap_history():
    cb = CircuitBreaker(CapsConfig(flap_threshold=3, max_actions_per_window=100))
    cb.record_apply(K, "k", 0)
    cb.record_apply(K, "k", 1)  # 2 of 3 toward flap
    cb.record_success(K, 2)  # a confirmed fix is not a flap -> clear kind history
    cb.record_apply(K, "k", 3)  # count restarts at 1
    assert cb.state(K) is BreakerState.CLOSED
