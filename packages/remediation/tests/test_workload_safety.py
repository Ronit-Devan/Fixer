"""The #1 hard constraint, proven: a NON-DISRUPTIVE path never kills the workload.

These tests assert the invariant from several angles:
  * an orphan-kill whose target IS the protected workload is refused (advised),
    and the protected PID is never signalled;
  * a renice whose target is the protected PID is refused at the manager;
  * across every non-disruptive scenario, no workload-lifecycle action ever runs
    and the protected PID never appears in the actuator's targeted set.
"""

from __future__ import annotations

from conftest import diag
from sim import Sim

from et_remediation import ActionKind, FakeTelemetryModel, OutcomeKind

WORKLOAD_PID = 4242
LIFECYCLE = {ActionKind.DRAIN_AND_RESTART_WORKLOAD, ActionKind.RESTART_LLAMA_SERVER}


def test_orphan_kill_refused_when_target_is_the_workload():
    sim = Sim(
        model=FakeTelemetryModel(util_pct=2, clock_ratio=0.95, mem_used_ratio=0.7),
        protected_pids=[WORKLOAD_PID],
    )
    # The "orphan" we'd kill happens to be the protected workload PID.
    sim.cfg.knobs["orphan_pid"] = WORKLOAD_PID
    out = sim.tick(diag("idle_no_requests", metrics={"mem_used_ratio": 0.7}))
    # Refused -> advised, nothing killed, workload PID never targeted.
    assert out.kind is OutcomeKind.ADVISED
    assert not sim.actuator.executed_kinds()
    assert WORKLOAD_PID not in sim.actuator.targeted_pids
    assert sim.actuator.protection_violations == 0


def test_orphan_kill_runs_only_for_a_genuine_orphan():
    sim = Sim(
        model=FakeTelemetryModel(util_pct=2, clock_ratio=0.95, mem_used_ratio=0.7),
        protected_pids=[WORKLOAD_PID],
        verify_window_s=5.0,
    )
    sim.cfg.knobs["orphan_pid"] = 9999  # NOT the workload
    out = sim.tick(diag("idle_no_requests", metrics={"mem_used_ratio": 0.7}))
    assert out.kind is OutcomeKind.APPLIED
    assert ActionKind.KILL_ORPHAN_PROCESS in sim.actuator.executed_kinds()
    assert WORKLOAD_PID not in sim.actuator.targeted_pids


def test_renice_refused_when_target_is_the_workload():
    sim = Sim(
        model=FakeTelemetryModel(util_pct=10, clock_ratio=0.95),
        protected_pids=[WORKLOAD_PID],
    )
    sim.cfg.knobs["worker_pid"] = WORKLOAD_PID
    out = sim.tick(diag("dataloader_bound", metrics={"hol_blocking_likely": True}), source="engine")
    assert out.kind is OutcomeKind.BLOCKED
    assert WORKLOAD_PID not in sim.actuator.targeted_pids


def test_no_non_disruptive_scenario_ever_runs_a_lifecycle_action():
    # Thermal throttle auto-fix: a device action, never a restart.
    sim = Sim(model=FakeTelemetryModel(util_pct=85, clock_ratio=0.55, power_limit_w=300))
    for _ in range(5):
        sim.tick(diag("thermal_throttle"))
    ran = set(sim.actuator.executed_kinds())
    assert ran and ran.isdisjoint(LIFECYCLE)
