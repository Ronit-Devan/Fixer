"""Registry: pluggable root-cause -> strategy mapping, non-disruptive first."""

from __future__ import annotations

from et_remediation.actions import ActionClass, ActionKind, ActionSpec
from et_remediation.registry import ActionRegistry
from et_remediation.rootcause import RootCause
from et_remediation.strategies import ALL_STRATEGIES, default_registry


def _spec(cls, kind):
    return ActionSpec(
        root_cause=RootCause.MEMORY_FRAGMENTATION,
        kind=kind,
        action_class=cls,
        reversible=True,
        summary="x",
        build_params=lambda ctx: {},
        recovered=lambda pre, post: True,
    )


def test_default_registry_has_all_strategies():
    reg = default_registry()
    assert len(reg.all_specs()) == len(ALL_STRATEGIES)
    # Every canonical actionable cause is represented.
    for rc in (
        RootCause.THERMAL_POWER_THROTTLE,
        RootCause.IDLE_ZOMBIE_PROCESS,
        RootCause.CPU_BOUND_PREPROCESSING,
        RootCause.MEMORY_FRAGMENTATION,
        RootCause.DATA_PIPELINE_STARVATION,
        RootCause.DISTRIBUTED_COMM_STALL,
        RootCause.SUBOPTIMAL_RUNTIME_FLAGS,
    ):
        assert reg.resolve(rc), f"no strategy for {rc}"


def test_resolve_orders_non_disruptive_first():
    reg = ActionRegistry()
    reg.register(_spec(ActionClass.DISRUPTIVE, ActionKind.DRAIN_AND_RESTART_WORKLOAD))
    reg.register(_spec(ActionClass.NON_DISRUPTIVE, ActionKind.FREE_STALE_CACHE))
    specs = reg.resolve(RootCause.MEMORY_FRAGMENTATION)
    assert specs[0].action_class is ActionClass.NON_DISRUPTIVE


def test_resolve_unknown_cause_is_empty():
    assert default_registry().resolve(RootCause.NONE) == []
