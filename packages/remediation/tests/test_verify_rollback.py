"""Verify predicates + the manager's verify->confirm / verify->rollback loop."""

from __future__ import annotations

from et_remediation.telemetry import summarize
from et_remediation.verify import (
    clock_recovered,
    memory_freed,
    util_recovered,
)
from conftest import diag
from sim import Sim

from et_remediation import OutcomeKind, RemediationMode
from et_remediation.actuators.fake import FakeSample


def _w(**kw):
    return summarize([FakeSample(**kw)])


# -- predicate unit tests ----------------------------------------------------


def test_clock_recovered_requires_floor_and_real_gain():
    # Recovered: cleared the floor AND moved meaningfully off the throttle.
    assert clock_recovered(_w(clock_ratio=0.55), _w(clock_ratio=0.97))
    # Ticked up to 0.62 but still below the throttle floor -> NOT recovered.
    assert not clock_recovered(_w(clock_ratio=0.55), _w(clock_ratio=0.62))
    # Barely moved -> NOT recovered.
    assert not clock_recovered(_w(clock_ratio=0.55), _w(clock_ratio=0.56))
    # False-confirm guard: window already looks healthy but the fix moved
    # nothing (gain < min_gain) -> NOT recovered (must not credit a no-op fix).
    assert not clock_recovered(_w(clock_ratio=0.90), _w(clock_ratio=0.91))


def test_missing_signal_is_not_recovered():
    # No clock reading post-fix => not proven recovered (so it will roll back).
    assert not clock_recovered(_w(clock_ratio=0.55), _w())


def test_util_and_memory_predicates():
    assert util_recovered(_w(util_pct=10), _w(util_pct=45))
    assert not util_recovered(_w(util_pct=10), _w(util_pct=12))
    assert memory_freed(_w(mem_used_ratio=0.8), _w(mem_used_ratio=0.3))
    assert not memory_freed(_w(mem_used_ratio=0.8), _w(mem_used_ratio=0.78))


# -- manager loop ------------------------------------------------------------


def test_auto_apply_confirms_on_recovery():
    sim = Sim(model=_thermal_model(), recover_on_apply=True, verify_window_s=5.0)
    o0 = sim.tick(diag("thermal_throttle"))
    assert o0.kind is OutcomeKind.APPLIED
    o1 = sim.tick(diag("thermal_throttle"))
    assert o1.kind is OutcomeKind.CONFIRMED
    assert sim.actuator.executed_kinds()  # something actually ran
    assert not sim.actuator.rolled_back_kinds()


def test_auto_rollback_on_non_recovery():
    sim = Sim(model=_thermal_model(), recover_on_apply=False, verify_window_s=2.0)
    assert sim.tick(diag("thermal_throttle")).kind is OutcomeKind.APPLIED  # t=0, deadline=2
    assert sim.tick(diag("thermal_throttle")).kind is OutcomeKind.VERIFYING  # t=1
    out = sim.tick(diag("thermal_throttle"))  # t=2 >= deadline -> rollback
    assert out.kind is OutcomeKind.ROLLED_BACK
    assert sim.actuator.rolled_back_kinds()


def test_advise_mode_never_actuates():
    sim = Sim(model=_thermal_model(), mode=RemediationMode.ADVISE)
    outs = sim.run(diag("thermal_throttle"), 3)
    assert outs == ["advised", "advised", "advised"]
    assert not sim.actuator.executed_kinds()


def _thermal_model():
    from et_remediation import FakeTelemetryModel

    return FakeTelemetryModel(util_pct=85, clock_ratio=0.55, temp_c=84, power_limit_w=300)
