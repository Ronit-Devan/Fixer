"""Classifier: enforces the NON-DISRUPTIVE vs DISRUPTIVE contract."""

from __future__ import annotations

import pytest

from et_remediation.actions import (
    ActionClass,
    ActionKind,
    ActionSpec,
)
from et_remediation.classifier import classify, is_auto_appliable, requires_approval
from et_remediation.rootcause import RootCause
from et_remediation.strategies import ALL_STRATEGIES


def _spec(kind, cls):
    return ActionSpec(
        root_cause=RootCause.THERMAL_POWER_THROTTLE,
        kind=kind,
        action_class=cls,
        reversible=True,
        summary="x",
        build_params=lambda ctx: {},
        recovered=lambda pre, post: True,
    )


def test_non_disruptive_cannot_carry_lifecycle_kind_at_construction():
    with pytest.raises(ValueError):
        _spec(ActionKind.RESTART_LLAMA_SERVER, ActionClass.NON_DISRUPTIVE)


def test_auto_appliable_only_for_non_disruptive():
    nd = _spec(ActionKind.SET_POWER_LIMIT, ActionClass.NON_DISRUPTIVE)
    d = _spec(ActionKind.RESTART_LLAMA_SERVER, ActionClass.DISRUPTIVE)
    assert is_auto_appliable(nd) and not requires_approval(nd)
    assert requires_approval(d) and not is_auto_appliable(d)


def test_every_builtin_strategy_classifies_cleanly():
    # No built-in strategy is mis-tagged.
    for spec in ALL_STRATEGIES:
        assert classify(spec) is spec.action_class
