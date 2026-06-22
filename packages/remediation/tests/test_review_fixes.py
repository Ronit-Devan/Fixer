"""Fixes from the adversarial review of the improvement waves, each proven."""

from __future__ import annotations

from dataclasses import dataclass

from conftest import diag
from sim import Sim

from et_remediation import (
    CapsConfig,
    FakeActuator,
    FakeTelemetryModel,
    FleetCoordinator,
    OutcomeKind,
    RemediationConfig,
    RemediationManager,
    RemediationMode,
    default_registry,
)


def _thermal():
    return FakeTelemetryModel(util_pct=85, clock_ratio=0.55, power_limit_w=300)


# -- predicted thermal is advise-only (raising power would worsen heat) -------


@dataclass
class PredDiag:
    verdict_value: str
    metrics: dict
    severity: str = "warn"
    summary: str = ""
    predicted: bool = True

    @property
    def verdict(self):
        class V:
            pass

        v = V()
        v.value = self.verdict_value
        return v


def test_predicted_thermal_is_advise_only():
    sim = Sim(model=_thermal())
    out = sim.tick(PredDiag("thermal_throttle", {"clock_ratio": 0.55}))
    assert out.kind is OutcomeKind.ADVISED  # never auto-raises power on a prediction
    assert not sim.actuator.executed_kinds()


def test_reactive_thermal_still_auto_applies():
    sim = Sim(model=_thermal())
    # predicted defaults to False on the plain FakeDiagnosis -> reactive path acts.
    assert sim.tick(diag("thermal_throttle")).kind is OutcomeKind.APPLIED


# -- OFF mid-verify resolves and releases the fleet slot ----------------------


def test_off_midverify_resolves_and_frees_fleet_slot():
    fleet = FleetCoordinator(max_concurrent=1)
    cfg = RemediationConfig(mode=RemediationMode.AUTO, verify_window_s=2.0)
    act = FakeActuator(_thermal(), recover_on_apply=False)
    mgr = RemediationManager(default_registry(), cfg, [act], fleet=fleet, now_fn=lambda: 0.0)

    assert mgr.observe(diag("thermal_throttle"), [], now=0.0).kind is OutcomeKind.APPLIED
    assert fleet.active_count() == 1
    mgr.set_mode(RemediationMode.OFF)  # flip OFF while verifying
    # The verify still resolves at the deadline (rollback) and frees the slot,
    # rather than freezing the run forever.
    out = mgr.observe(diag("thermal_throttle"), [], now=5.0)
    assert out.kind is OutcomeKind.ROLLED_BACK
    assert fleet.active_count() == 0


# -- breaker not stranded HALF_OPEN when a later gate blocks the apply --------


def test_breaker_not_stuck_half_open_when_fleet_blocks_trial():
    # Breaker tripped, cooldown elapsed -> a HALF_OPEN trial would be granted;
    # but the fleet slot is taken by another GPU, so we must NOT consume the trial.
    fleet = FleetCoordinator(max_concurrent=1)
    fleet.try_acquire("other-gpu")  # the only slot is held elsewhere
    caps = CapsConfig(failure_threshold=1, breaker_cooldown_s=1.0)
    cfg = RemediationConfig(mode=RemediationMode.AUTO, caps=caps)
    act = FakeActuator(_thermal(), recover_on_apply=False)
    mgr = RemediationManager(default_registry(), cfg, [act], fleet=fleet, now_fn=lambda: 0.0)

    mgr.breaker.record_failure(mgr.node_id, 0.0)  # trip OPEN
    # Past cooldown: the apply is blocked by the fleet, so the breaker trial is
    # NOT consumed and the breaker is not stranded HALF_OPEN.
    out = mgr.observe(diag("thermal_throttle"), [], now=10.0)
    assert out.kind is OutcomeKind.BLOCKED
    from et_remediation import BreakerState
    assert mgr.breaker.state(mgr.node_id) is BreakerState.OPEN  # still open, trial intact


# -- debounce does not immediately re-fire after a rollback -------------------


def test_debounce_resets_after_action_so_no_immediate_refire():
    cfg = RemediationConfig(mode=RemediationMode.AUTO, verify_window_s=1.0, trigger_debounce=2)
    sim = Sim(model=_thermal(), recover_on_apply=False, config=cfg)
    assert sim.tick(diag("thermal_throttle")).kind is OutcomeKind.DEBOUNCING  # 1/2
    assert sim.tick(diag("thermal_throttle")).kind is OutcomeKind.APPLIED      # 2/2 -> act
    # verify rolls back (no recovery), and the next sighting must NOT act
    # immediately — it has to re-accumulate the debounce.
    sim.tick(diag("thermal_throttle"))  # resolves -> rollback
    assert sim.tick(diag("thermal_throttle")).kind is OutcomeKind.DEBOUNCING   # re-confirming


# -- deadline never confirms on too-few samples ------------------------------


def test_deadline_with_too_few_samples_rolls_back_not_confirm():
    # min_verify_samples=2; provide only a single (recovered-looking) post sample
    # right at the deadline -> must roll back, not confirm on one reading.
    @dataclass(frozen=True)
    class TS:
        timestamp_s: float
        clock_ratio: float

    cfg = RemediationConfig(mode=RemediationMode.AUTO, verify_window_s=3.0, min_verify_samples=2)
    act = FakeActuator(_thermal(), recover_on_apply=False)
    mgr = RemediationManager(default_registry(), cfg, [act], now_fn=lambda: 0.0)
    assert mgr.observe(diag("thermal_throttle"), [TS(0, 0.55)], now=0.0).kind is OutcomeKind.APPLIED
    # one fresh, healthy-looking post sample at the deadline -> still not enough.
    out = mgr.observe(diag("thermal_throttle"), [TS(4, 0.99)], now=4.0)
    assert out.kind is OutcomeKind.ROLLED_BACK
