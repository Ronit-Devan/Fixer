"""Pluggable root-cause -> action registry.

The mapping from a diagnosed ``RootCause`` to candidate ``ActionSpec``s lives
here so new strategies are added without touching the manager. A root cause may
have several candidate specs (e.g. a non-disruptive attempt plus a disruptive
fallback); the manager picks the first whose preconditions hold.
"""

from __future__ import annotations

from collections import defaultdict

from et_remediation.actions import ActionClass, ActionSpec
from et_remediation.rootcause import RootCause


class ActionRegistry:
    def __init__(self) -> None:
        self._by_cause: dict[RootCause, list[ActionSpec]] = defaultdict(list)

    def register(self, spec: ActionSpec) -> "ActionRegistry":
        self._by_cause[spec.root_cause].append(spec)
        return self

    def register_all(self, specs: list[ActionSpec]) -> "ActionRegistry":
        for s in specs:
            self.register(s)
        return self

    def resolve(self, root_cause: RootCause) -> list[ActionSpec]:
        """Candidate specs for a root cause, non-disruptive first.

        Non-disruptive candidates are ordered ahead of disruptive ones so the
        manager always prefers the auto-appliable, workload-preserving fix and
        only falls back to the approval-gated path when none applies.
        """
        specs = list(self._by_cause.get(root_cause, []))
        specs.sort(key=lambda s: 0 if s.action_class is ActionClass.NON_DISRUPTIVE else 1)
        return specs

    def causes(self) -> list[RootCause]:
        return list(self._by_cause.keys())

    def all_specs(self) -> list[ActionSpec]:
        return [s for specs in self._by_cause.values() for s in specs]
