"""Static policy gates: mode, class enable, protected workload, blast radius."""

from __future__ import annotations

from et_remediation.actions import ProtectedWorkload
from et_remediation.config import RemediationConfig, RemediationMode
from et_remediation.policy import PolicyEngine
from et_remediation.strategies import (
    CPU_BOUND_PREPROCESSING,
    SUBOPTIMAL_RUNTIME_FLAGS,
    THERMAL_POWER_THROTTLE,
)

NO_WORKLOAD = ProtectedWorkload()
WITH_WORKLOAD = ProtectedWorkload(pids=frozenset({4242}), label="job-x")


def _policy(mode):
    return PolicyEngine(RemediationConfig(mode=mode))


def test_off_skips_everything():
    d = _policy(RemediationMode.OFF).decide(THERMAL_POWER_THROTTLE, NO_WORKLOAD, in_flight_on_node=0)
    assert d.outcome == "skip"


def test_advise_mode_only_advises_non_disruptive():
    d = _policy(RemediationMode.ADVISE).decide(THERMAL_POWER_THROTTLE, NO_WORKLOAD, in_flight_on_node=0)
    assert d.outcome == "advise"


def test_auto_actuates_device_action_without_workload():
    # Power-limit doesn't touch a process, so no protected workload is required.
    d = _policy(RemediationMode.AUTO).decide(THERMAL_POWER_THROTTLE, NO_WORKLOAD, in_flight_on_node=0)
    assert d.outcome == "actuate"


def test_process_touching_requires_protected_workload():
    pol = _policy(RemediationMode.AUTO)
    # No protected workload defined -> refuse to actuate a renice; advise instead.
    assert pol.decide(CPU_BOUND_PREPROCESSING, NO_WORKLOAD, in_flight_on_node=0).outcome == "advise"
    # With a workload defined we know what to protect -> actuate.
    assert pol.decide(CPU_BOUND_PREPROCESSING, WITH_WORKLOAD, in_flight_on_node=0).outcome == "actuate"


def test_disruptive_always_routes_to_approval_never_actuate():
    for mode in (RemediationMode.ADVISE, RemediationMode.DRY_RUN, RemediationMode.AUTO):
        d = _policy(mode).decide(SUBOPTIMAL_RUNTIME_FLAGS, WITH_WORKLOAD, in_flight_on_node=0)
        assert d.outcome == "approval"


def test_blast_radius_blocks_when_in_flight():
    d = _policy(RemediationMode.AUTO).decide(THERMAL_POWER_THROTTLE, NO_WORKLOAD, in_flight_on_node=1)
    assert d.outcome == "blocked"


def test_dry_run_mode_builds_only():
    d = _policy(RemediationMode.DRY_RUN).decide(THERMAL_POWER_THROTTLE, NO_WORKLOAD, in_flight_on_node=0)
    assert d.outcome == "dry_run"


def test_disabling_non_disruptive_falls_to_advise():
    cfg = RemediationConfig(mode=RemediationMode.AUTO, enable_non_disruptive=False)
    d = PolicyEngine(cfg).decide(THERMAL_POWER_THROTTLE, NO_WORKLOAD, in_flight_on_node=0)
    assert d.outcome == "advise"
