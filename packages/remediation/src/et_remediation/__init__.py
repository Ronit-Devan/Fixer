"""ET remediation/actuation layer.

Consumes a diagnosed root cause (from the training engine or the inference
monitor) and *resolves* it: NON-DISRUPTIVE fixes auto-apply unattended through a
guarded path (classify -> apply -> verify recovery -> confirm/auto-rollback);
DISRUPTIVE fixes are surfaced only as human-gated approval requests. A circuit
breaker, a global kill-switch (operating mode), per-node/job caps, and a full
audit log govern the whole thing. Detection/diagnosis are untouched — the layer
consumes them through duck-typed Protocols.
"""

from __future__ import annotations

__version__ = "0.2.0"

from et_remediation.actions import (
    ActionClass,
    ActionKind,
    ActionRequest,
    ActionResult,
    ActionSpec,
    ProtectedWorkload,
)
from et_remediation.actuators import (
    Actuator,
    CommandRunner,
    DataCenterActuator,
    FakeActuator,
    FakeTelemetryModel,
    LlamaCppActuator,
)
from et_remediation.audit import AuditLog, AuditRecord, Phase
from et_remediation.breaker import BreakerState, CircuitBreaker
from et_remediation.config import (
    CapsConfig,
    RemediationConfig,
    RemediationMode,
    default_config_path,
)
from et_remediation.engine import (
    ApprovalRequest,
    Outcome,
    OutcomeKind,
    RemediationManager,
    RunState,
)
from et_remediation.fleet import FleetCoordinator
from et_remediation.registry import ActionRegistry
from et_remediation.rootcause import (
    RootCause,
    map_engine_verdict,
    map_from_metrics,
    map_monitor_verdict,
)
from et_remediation.strategies import ALL_STRATEGIES, default_registry

__all__ = [
    "__version__",
    # config / mode (kill-switch)
    "RemediationConfig",
    "RemediationMode",
    "CapsConfig",
    "default_config_path",
    # actions
    "ActionClass",
    "ActionKind",
    "ActionRequest",
    "ActionResult",
    "ActionSpec",
    "ProtectedWorkload",
    # registry / strategies
    "ActionRegistry",
    "default_registry",
    "ALL_STRATEGIES",
    # root cause
    "RootCause",
    "map_monitor_verdict",
    "map_engine_verdict",
    "map_from_metrics",
    # guardrails
    "CircuitBreaker",
    "BreakerState",
    "AuditLog",
    "AuditRecord",
    "Phase",
    # actuators
    "Actuator",
    "CommandRunner",
    "DataCenterActuator",
    "LlamaCppActuator",
    "FakeActuator",
    "FakeTelemetryModel",
    # manager
    "RemediationManager",
    "Outcome",
    "OutcomeKind",
    "RunState",
    "ApprovalRequest",
    "FleetCoordinator",
]
