"""The NON-DISRUPTIVE vs DISRUPTIVE classifier.

Each ``ActionSpec`` declares its class, but the classifier is the single place
that *enforces* what those classes are allowed to be — so a mis-tagged spec can
never widen the auto-apply blast radius. The rules:

  * A NON_DISRUPTIVE action must not use a workload-lifecycle kind (restart /
    drain). Verified here and at ``ActionSpec`` construction.
  * Only NON_DISRUPTIVE actions are eligible for unattended auto-apply.
  * A DISRUPTIVE action is *only ever* surfaced as an approval request.
"""

from __future__ import annotations

from et_remediation.actions import (
    WORKLOAD_LIFECYCLE_KINDS,
    ActionClass,
    ActionSpec,
)


def classify(spec: ActionSpec) -> ActionClass:
    """Return the enforced class for a spec, raising on an unsafe combination."""
    if (
        spec.action_class is ActionClass.NON_DISRUPTIVE
        and spec.kind in WORKLOAD_LIFECYCLE_KINDS
    ):
        raise ValueError(
            f"spec for {spec.root_cause.value} tagged NON_DISRUPTIVE but uses "
            f"workload-lifecycle kind {spec.kind.value}"
        )
    return spec.action_class


def is_auto_appliable(spec: ActionSpec) -> bool:
    """Only non-disruptive actions may ever auto-apply unattended."""
    return classify(spec) is ActionClass.NON_DISRUPTIVE


def requires_approval(spec: ActionSpec) -> bool:
    """Disruptive actions always require human approval; never auto-fire."""
    return classify(spec) is ActionClass.DISRUPTIVE
