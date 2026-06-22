"""Static policy gates (the config-level half of the guardrail engine).

The circuit breaker handles the *dynamic* gates (failure / flap / rate). This
module handles the *static* ones derived from ``RemediationConfig``:

  * the global kill-switch / operating mode,
  * per-class master switches,
  * presence of a defined protected workload (we refuse process-touching fixes
    when we don't know what to protect),
  * blast radius (one in-flight action per node).

The manager composes both halves.
"""

from __future__ import annotations

from dataclasses import dataclass

from et_remediation.actions import (
    PROCESS_TOUCHING_KINDS,
    ActionClass,
    ActionSpec,
    ProtectedWorkload,
)
from et_remediation.config import RemediationConfig, RemediationMode


@dataclass(frozen=True)
class PolicyDecision:
    # One of: "actuate" (auto-apply), "dry_run", "advise", "approval",
    # "blocked", "skip".
    outcome: str
    reason: str


class PolicyEngine:
    def __init__(self, config: RemediationConfig) -> None:
        self.config = config

    @property
    def mode(self) -> RemediationMode:
        return self.config.mode

    def decide(
        self,
        spec: ActionSpec,
        protected: ProtectedWorkload,
        *,
        in_flight_on_node: int,
    ) -> PolicyDecision:
        """Static gate for a candidate spec, before the breaker is consulted."""
        mode = self.config.mode

        if mode is RemediationMode.OFF:
            return PolicyDecision("skip", "mode_off")

        if spec.action_class is ActionClass.DISRUPTIVE:
            # Disruptive is NEVER auto-executed. The only thing any mode does is
            # decide whether to open an approval request at all.
            if not self.config.enable_disruptive_requests:
                return PolicyDecision("advise", "disruptive_requests_disabled")
            return PolicyDecision("approval", "disruptive_requires_approval")

        # --- non-disruptive ---
        if not self.config.enable_non_disruptive:
            return PolicyDecision("advise", "non_disruptive_disabled")

        # A process-touching fix with no defined workload to protect is unsafe:
        # we cannot prove we won't hit the live task. Fall back to advise.
        if spec.kind in PROCESS_TOUCHING_KINDS and not protected.pids:
            return PolicyDecision("advise", "no_protected_workload_defined")

        # Blast radius: at most one in-flight (verifying) action per node.
        if in_flight_on_node >= self.config.caps.max_concurrent_per_node:
            return PolicyDecision("blocked", "blast_radius_in_flight")

        if mode is RemediationMode.ADVISE:
            return PolicyDecision("advise", "mode_advise")
        if mode is RemediationMode.DRY_RUN:
            return PolicyDecision("dry_run", "mode_dry_run")
        if mode is RemediationMode.AUTO:
            return PolicyDecision("actuate", "mode_auto")

        return PolicyDecision("advise", "fallthrough")  # pragma: no cover
