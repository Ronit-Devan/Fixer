"""RemediationManager state machine: outcomes, dry-run, no-op, approval gate."""

from __future__ import annotations

from conftest import diag
from sim import Sim

from et_remediation import FakeTelemetryModel, OutcomeKind, RemediationMode


def _thermal():
    return FakeTelemetryModel(util_pct=85, clock_ratio=0.55, temp_c=84, power_limit_w=300)


def test_healthy_diagnosis_is_noop():
    sim = Sim(model=_thermal())
    assert sim.tick(diag("healthy", severity="ok")).kind is OutcomeKind.HEALTHY


def test_off_mode_does_nothing():
    sim = Sim(model=_thermal(), mode=RemediationMode.OFF)
    assert sim.tick(diag("thermal_throttle")).kind is OutcomeKind.OFF
    assert not sim.actuator.calls


def test_dry_run_builds_but_does_not_execute():
    sim = Sim(model=_thermal(), mode=RemediationMode.DRY_RUN)
    out = sim.tick(diag("thermal_throttle"))
    assert out.kind is OutcomeKind.PLANNED
    # apply was recorded, but nothing executed and no verify loop started.
    assert sim.actuator.calls and not sim.actuator.executed_kinds()
    assert sim.mgr.status()["state"] == "normal"


def test_idempotent_no_op_when_already_at_target():
    # power_limit already equals the target the strategy would set (300*1.15=345).
    model = FakeTelemetryModel(util_pct=85, clock_ratio=0.55, power_limit_w=345)
    sim = Sim(model=model)
    # knob fixes the target so the requested value matches the model exactly.
    sim.cfg.knobs["target_power_w"] = 345
    out = sim.tick(diag("thermal_throttle"))
    assert out.kind is OutcomeKind.NO_OP


def test_disruptive_opens_single_approval_and_never_executes():
    sim = Sim(model=FakeTelemetryModel(util_pct=40, clock_ratio=0.95))
    for _ in range(3):
        out = sim.tick(diag("decode_bandwidth_bound"))
        assert out.kind is OutcomeKind.APPROVAL_REQUIRED
    assert len(sim.mgr.pending_approvals()) == 1  # deduped per (node, cause)
    assert not sim.actuator.executed_kinds()  # NEVER auto-fires


def test_approval_executes_only_on_human_approve():
    sim = Sim(model=FakeTelemetryModel(util_pct=40, clock_ratio=0.95))
    out = sim.tick(diag("decode_bandwidth_bound"))
    ar = out.approval
    assert ar is not None and ar.status == "pending"
    assert not sim.actuator.executed_kinds()
    res = sim.mgr.approve(ar.id, now=99.0)
    assert res.kind is OutcomeKind.CONFIRMED
    assert ar.status == "applied"
    # The restart actually ran now (human-approved), via the disruptive path.
    from et_remediation import ActionKind

    assert ActionKind.RESTART_LLAMA_SERVER in sim.actuator.executed_kinds()


def test_approval_preview_uses_real_actuator_command():
    # Wire the manager with the real llama actuator: the preview is the exact
    # tuned restart command an operator would approve.
    from et_remediation import (
        DataCenterActuator,
        LlamaCppActuator,
        RemediationConfig,
        RemediationManager,
        default_registry,
    )

    cfg = RemediationConfig(mode=RemediationMode.AUTO, knobs={"model": "m.gguf"})
    mgr = RemediationManager(
        default_registry(), cfg,
        [DataCenterActuator(), LlamaCppActuator()],
        now_fn=lambda: 0.0,
    )
    out = mgr.observe(diag("decode_bandwidth_bound"), [], now=0.0)
    assert out.kind is OutcomeKind.APPROVAL_REQUIRED
    assert "llama-server" in out.approval.command_preview
    assert "-ngl" in out.approval.command_preview


def test_reject_approval():
    sim = Sim(model=FakeTelemetryModel(util_pct=40, clock_ratio=0.95))
    ar = sim.tick(diag("decode_bandwidth_bound")).approval
    assert sim.mgr.reject(ar.id) is True
    assert sim.mgr.approve(ar.id).kind is OutcomeKind.BLOCKED  # no longer pending


def test_blast_radius_one_in_flight():
    # While verifying, a second tick does not start a new action.
    sim = Sim(model=_thermal(), recover_on_apply=False, verify_window_s=10.0)
    assert sim.tick(diag("thermal_throttle")).kind is OutcomeKind.APPLIED
    # still verifying; no new apply
    assert sim.tick(diag("thermal_throttle")).kind is OutcomeKind.VERIFYING
    assert len([c for c in sim.actuator.calls if c.op == "apply"]) == 1


def test_trigger_debounce_requires_consecutive_observations():
    from et_remediation import RemediationConfig

    cfg = RemediationConfig(mode=RemediationMode.AUTO, verify_window_s=5.0, trigger_debounce=3)
    sim = Sim(model=_thermal(), config=cfg)
    # First two sightings just confirm the cause; no action yet.
    assert sim.tick(diag("thermal_throttle")).kind is OutcomeKind.DEBOUNCING
    assert sim.tick(diag("thermal_throttle")).kind is OutcomeKind.DEBOUNCING
    assert not sim.actuator.executed_kinds()
    # Third consecutive sighting -> act.
    assert sim.tick(diag("thermal_throttle")).kind is OutcomeKind.APPLIED


def test_debounce_resets_when_cause_changes():
    from et_remediation import RemediationConfig

    cfg = RemediationConfig(mode=RemediationMode.AUTO, verify_window_s=5.0, trigger_debounce=2)
    sim = Sim(model=_thermal(), config=cfg)
    assert sim.tick(diag("thermal_throttle")).kind is OutcomeKind.DEBOUNCING
    # A different (healthy) reading resets the debounce counter.
    assert sim.tick(diag("healthy", severity="ok")).kind is OutcomeKind.HEALTHY
    # So the next thermal sighting starts counting again from 1.
    assert sim.tick(diag("thermal_throttle")).kind is OutcomeKind.DEBOUNCING


def test_audit_records_full_lifecycle():
    sim = Sim(model=_thermal(), recover_on_apply=True, verify_window_s=5.0)
    sim.tick(diag("thermal_throttle"))
    sim.tick(diag("thermal_throttle"))
    phases = [r.phase.value for r in sim.mgr.audit.recent()]
    assert "trigger" in phases and "apply" in phases and "verify" in phases
