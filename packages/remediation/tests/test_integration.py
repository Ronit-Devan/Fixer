"""End-to-end: the full detect->diagnose->remediate->verify loop on the sim.

Drives the manager exactly as the live monitor would (one observe() per tick),
proving the headline behaviors together: every strategy is reachable, the
circuit breaker trips to advise-only, and the kill-switch forces advise-only.
"""

from __future__ import annotations

from conftest import diag
from sim import Sim

from et_remediation import (
    ActionKind,
    BreakerState,
    CapsConfig,
    FakeTelemetryModel,
    OutcomeKind,
    RemediationConfig,
    RemediationMode,
)

PROTECTED = [4242]


def _sim_with_workload(model, **kw):
    return Sim(model=model, protected_pids=PROTECTED, verify_window_s=5.0, **kw)


# -- every strategy is reachable end to end ----------------------------------


def test_thermal_throttle_auto_applies():
    sim = Sim(model=FakeTelemetryModel(util_pct=85, clock_ratio=0.55, power_limit_w=300))
    out = sim.tick(diag("thermal_throttle"))
    assert out.kind is OutcomeKind.APPLIED
    assert out.action_kind is ActionKind.SET_POWER_LIMIT


def test_idle_zombie_kills_orphan():
    sim = _sim_with_workload(FakeTelemetryModel(util_pct=2, clock_ratio=0.95, mem_used_ratio=0.7))
    sim.cfg.knobs["orphan_pid"] = 9999
    out = sim.tick(diag("idle_no_requests", metrics={"mem_used_ratio": 0.7}))
    assert out.kind is OutcomeKind.APPLIED and out.action_kind is ActionKind.KILL_ORPHAN_PROCESS


def test_cpu_bound_renices_worker():
    sim = _sim_with_workload(FakeTelemetryModel(util_pct=10, clock_ratio=0.95))
    sim.cfg.knobs["worker_pid"] = 7001
    out = sim.tick(diag("dataloader_bound", metrics={"hol_blocking_likely": True}), source="engine")
    assert out.kind is OutcomeKind.APPLIED and out.action_kind is ActionKind.RENICE_PROCESS


def test_memory_fragmentation_frees_stale():
    sim = _sim_with_workload(FakeTelemetryModel(util_pct=30, clock_ratio=0.95, mem_used_ratio=0.8))
    sim.cfg.knobs["stale_pid"] = 8001
    out = sim.tick(diag("unknown", metrics={"fragmentation_ratio": 0.5}))
    assert out.kind is OutcomeKind.APPLIED and out.action_kind is ActionKind.FREE_STALE_CACHE


def test_data_pipeline_starvation_requires_approval():
    sim = Sim(model=FakeTelemetryModel(util_pct=20, clock_ratio=0.95))
    out = sim.tick(diag("dataloader_bound"), source="engine")
    assert out.kind is OutcomeKind.APPROVAL_REQUIRED
    assert out.action_kind is ActionKind.DRAIN_AND_RESTART_WORKLOAD


def test_nccl_comm_stall_requires_approval():
    sim = Sim(model=FakeTelemetryModel(util_pct=20, clock_ratio=0.95))
    out = sim.tick(diag("nccl_bound"), source="engine")
    assert out.kind is OutcomeKind.APPROVAL_REQUIRED
    assert out.action_kind is ActionKind.DRAIN_AND_RESTART_WORKLOAD


def test_suboptimal_flags_requires_approval():
    sim = Sim(model=FakeTelemetryModel(util_pct=40, clock_ratio=0.95))
    out = sim.tick(diag("decode_bandwidth_bound"))
    assert out.kind is OutcomeKind.APPROVAL_REQUIRED
    assert out.action_kind is ActionKind.RESTART_LLAMA_SERVER
    assert out.approval.command_preview  # operator sees what they'd approve
    assert out.approval.requires_drain  # disruptive llama restart drains first


# -- circuit breaker trips to advise-only ------------------------------------


def test_breaker_trips_after_repeated_non_recovery_then_advise_only():
    cfg = RemediationConfig(
        mode=RemediationMode.AUTO,
        verify_window_s=1.0,
        caps=CapsConfig(failure_threshold=3, max_actions_per_window=50, breaker_cooldown_s=10_000),
    )
    sim = Sim(
        model=FakeTelemetryModel(util_pct=85, clock_ratio=0.55, power_limit_w=300),
        recover_on_apply=False,
        config=cfg,
    )
    outs = [sim.tick(diag("thermal_throttle")).kind.value for _ in range(9)]
    # Three apply->rollback cycles, then the breaker is OPEN and we fall back to
    # advise-only (BLOCKED) instead of actuating.
    assert outs.count("rolled_back") == 3
    assert "blocked" in outs
    assert sim.breaker.state("node-0") is BreakerState.OPEN
    # No more than three real applies ever executed.
    assert sim.actuator.executed_kinds().count(ActionKind.SET_POWER_LIMIT) == 3


# -- kill-switch -------------------------------------------------------------


def test_kill_switch_off_forces_no_action():
    sim = Sim(model=FakeTelemetryModel(util_pct=85, clock_ratio=0.55, power_limit_w=300))
    assert sim.tick(diag("thermal_throttle")).kind is OutcomeKind.APPLIED
    sim.mgr.set_mode(RemediationMode.OFF)  # operator flips the kill-switch mid-verify
    # The already-applied fix still completes its verify (it must NOT freeze or
    # leak its fleet slot just because the switch flipped)...
    sim.tick(diag("thermal_throttle"))
    # ...but no NEW action ever fires while OFF.
    for _ in range(3):
        assert sim.tick(diag("thermal_throttle")).kind is OutcomeKind.OFF
    # Only the single pre-flip apply ever ran.
    assert sim.actuator.executed_kinds().count(ActionKind.SET_POWER_LIMIT) == 1


def test_kill_switch_advise_forces_advise_only():
    sim = Sim(model=FakeTelemetryModel(util_pct=85, clock_ratio=0.55, power_limit_w=300))
    sim.mgr.set_mode(RemediationMode.ADVISE)
    outs = [sim.tick(diag("thermal_throttle")).kind.value for _ in range(3)]
    assert outs == ["advised", "advised", "advised"]
    assert not sim.actuator.executed_kinds()
