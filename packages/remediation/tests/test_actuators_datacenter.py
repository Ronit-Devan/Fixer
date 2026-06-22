"""DataCenter actuator: real command construction, dry-run gate, never-kill guard."""

from __future__ import annotations

import pytest

from et_remediation.actions import (
    ActionClass,
    ActionKind,
    ActionRequest,
    ProtectedWorkload,
)
from et_remediation.actuators.base import CommandRunner, WorkloadProtectionError
from et_remediation.actuators.datacenter import DataCenterActuator


def _req(kind, params, *, protected=None, dry_run=True, target="0"):
    return ActionRequest(
        kind=kind,
        action_class=ActionClass.NON_DISRUPTIVE,
        node_id="n0",
        target=target,
        params=params,
        protected=protected or ProtectedWorkload(),
        dry_run=dry_run,
    )


def test_build_power_limit_command():
    act = DataCenterActuator()
    cmd = act.build_command(_req(ActionKind.SET_POWER_LIMIT, {"power_limit_w": 345}))
    assert cmd == ["nvidia-smi", "-i", "0", "-pl", "345"]


def test_build_renice_and_affinity():
    act = DataCenterActuator()
    assert act.build_command(_req(ActionKind.RENICE_PROCESS, {"pid": 1234, "nice": -5})) == [
        "renice", "-n", "-5", "-p", "1234",
    ]
    assert act.build_command(_req(ActionKind.SET_CPU_AFFINITY, {"pid": 1234, "cpu_list": "0-7"})) == [
        "taskset", "-pc", "0-7", "1234",
    ]


def test_build_mps_mig_and_clocks():
    act = DataCenterActuator()
    assert "-mig" in act.build_command(_req(ActionKind.CONFIGURE_MIG, {"mig_enable": "1"}))
    assert act.build_command(_req(ActionKind.RESET_CLOCKS, {})) == ["nvidia-smi", "-i", "0", "-rgc"]
    assert act.build_command(_req(ActionKind.LOCK_CLOCKS, {"min_mhz": 1000, "max_mhz": 2000})) == [
        "nvidia-smi", "-i", "0", "-lgc", "1000,2000",
    ]


def test_dry_run_builds_but_never_executes():
    # Even with an executing runner, a dry-run request must not run.
    act = DataCenterActuator(CommandRunner(execute=True))
    res = act.apply(_req(ActionKind.SET_POWER_LIMIT, {"power_limit_w": 300}, dry_run=True))
    assert res.ok and not res.executed
    assert res.command == "nvidia-smi -i 0 -pl 300"


def test_never_kill_protected_pid():
    act = DataCenterActuator(CommandRunner(execute=True))
    protected = ProtectedWorkload(pids=frozenset({4242}), label="live-job")
    req = _req(ActionKind.KILL_ORPHAN_PROCESS, {"pid": 4242}, protected=protected, dry_run=False)
    with pytest.raises(WorkloadProtectionError):
        act.apply(req)


def test_orphan_kill_allowed_for_unprotected_pid():
    # No execution (non-executing runner) so the test never signals a real PID.
    act = DataCenterActuator(CommandRunner(execute=False))
    protected = ProtectedWorkload(pids=frozenset({4242}))
    res = act.apply(_req(ActionKind.KILL_ORPHAN_PROCESS, {"pid": 9999}, protected=protected, dry_run=False))
    assert res.ok and res.command == "kill -9 9999"


def test_irreversible_kinds_have_no_rollback():
    act = DataCenterActuator()
    state = act.snapshot_state(_req(ActionKind.KILL_ORPHAN_PROCESS, {"pid": 9999}))
    res = act.rollback(_req(ActionKind.KILL_ORPHAN_PROCESS, {"pid": 9999}), state)
    assert res.ok and "irreversible" in res.message.lower()


def test_capabilities_cover_non_disruptive_and_drain():
    caps = DataCenterActuator().capabilities()
    assert ActionKind.SET_POWER_LIMIT in caps
    assert ActionKind.DRAIN_AND_RESTART_WORKLOAD in caps
