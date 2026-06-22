"""Fleet blast-radius: cap how many GPUs auto-remediate at once across a box."""

from __future__ import annotations

from conftest import diag

from et_remediation import (
    FakeActuator,
    FakeTelemetryModel,
    FleetCoordinator,
    OutcomeKind,
    RemediationConfig,
    RemediationManager,
    RemediationMode,
    default_registry,
)


def _thermal_model():
    return FakeTelemetryModel(util_pct=85, clock_ratio=0.55, power_limit_w=300)


def _mgr(node_id, fleet, *, recover=False):
    cfg = RemediationConfig(mode=RemediationMode.AUTO, verify_window_s=5.0)
    act = FakeActuator(_thermal_model(), recover_on_apply=recover)
    mgr = RemediationManager(
        default_registry(), cfg, [act], fleet=fleet, now_fn=lambda: 0.0, node_id=node_id
    )
    return mgr, act


def test_coordinator_basic_cap():
    fc = FleetCoordinator(max_concurrent=1)
    assert fc.try_acquire("a") is True
    assert fc.try_acquire("b") is False  # cap reached
    assert fc.try_acquire("a") is True  # re-entrant for the holder
    fc.release("a")
    assert fc.try_acquire("b") is True


def test_second_gpu_blocked_while_first_remediating():
    fleet = FleetCoordinator(max_concurrent=1)
    a, aact = _mgr("host:gpu0", fleet, recover=False)
    b, bact = _mgr("host:gpu1", fleet, recover=False)

    # GPU 0 applies and enters verify (holds the only fleet slot).
    assert a.observe(diag("thermal_throttle"), [], now=0.0).kind is OutcomeKind.APPLIED
    # GPU 1 wants to act at the same time -> blocked by the fleet cap (advise-only).
    out_b = b.observe(diag("thermal_throttle"), [], now=0.0)
    assert out_b.kind is OutcomeKind.BLOCKED
    assert "fleet" in out_b.detail
    assert not bact.executed_kinds()  # GPU 1 never actuated

    # GPU 0's verify resolves (rolls back, no recovery) -> releases the slot.
    a.observe(diag("thermal_throttle"), [], now=10.0)  # deadline passed -> rollback
    assert fleet.active_count() == 0
    # Now GPU 1 may act.
    assert b.observe(diag("thermal_throttle"), [], now=11.0).kind is OutcomeKind.APPLIED


def test_higher_cap_allows_more_concurrency():
    fleet = FleetCoordinator(max_concurrent=2)
    mgrs = [_mgr(f"host:gpu{i}", fleet, recover=False)[0] for i in range(3)]
    assert mgrs[0].observe(diag("thermal_throttle"), [], now=0.0).kind is OutcomeKind.APPLIED
    assert mgrs[1].observe(diag("thermal_throttle"), [], now=0.0).kind is OutcomeKind.APPLIED
    # Third is over the cap.
    assert mgrs[2].observe(diag("thermal_throttle"), [], now=0.0).kind is OutcomeKind.BLOCKED
