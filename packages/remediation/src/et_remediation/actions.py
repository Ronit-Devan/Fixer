"""Action vocabulary: what a remediation *is*, before any backend runs it.

These types are deliberately backend-free. An ``ActionSpec`` says "for this root
cause, the fix is to set a power limit; it is non-disruptive; here is how to
build its parameters from the live context, and here is how to tell whether it
worked." A concrete ``Actuator`` (DataCenter / llama.cpp / Fake) turns an
``ActionRequest`` into a real command.

The ``ProtectedWorkload`` rides on every request. It is the structural guard for
the #1 hard constraint — *never kill/evict/restart the live compute task as a
side effect*: any process-touching actuator must refuse a target inside it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from et_remediation.rootcause import RootCause
from et_remediation.telemetry import WindowSummary


class ActionClass(str, Enum):
    """The safety class that decides the operating path."""

    # Power/clock limits, MPS/MIG tuning, freeing stale cache, killing only
    # orphaned procs: never touches the running workload. Auto-appliable.
    NON_DISRUPTIVE = "non_disruptive"
    # Needs a process/job restart. Checkpoint + drain + human approval. Never
    # fires automatically.
    DISRUPTIVE = "disruptive"


class ActionKind(str, Enum):
    """The concrete operation an actuator performs."""

    # --- non-disruptive (DataCenter) ---
    SET_POWER_LIMIT = "set_power_limit"
    LOCK_CLOCKS = "lock_clocks"
    RESET_CLOCKS = "reset_clocks"
    RENICE_PROCESS = "renice_process"
    SET_CPU_AFFINITY = "set_cpu_affinity"
    CONFIGURE_MPS = "configure_mps"
    CONFIGURE_MIG = "configure_mig"
    KILL_ORPHAN_PROCESS = "kill_orphan_process"
    FREE_STALE_CACHE = "free_stale_cache"

    # --- disruptive (DataCenter / llama.cpp) ---
    DRAIN_AND_RESTART_WORKLOAD = "drain_and_restart_workload"
    RESTART_LLAMA_SERVER = "restart_llama_server"


# Kinds that touch the running workload's lifecycle. A NON_DISRUPTIVE spec may
# never use one of these — asserted in the classifier and at apply time.
WORKLOAD_LIFECYCLE_KINDS: frozenset[ActionKind] = frozenset(
    {ActionKind.DRAIN_AND_RESTART_WORKLOAD, ActionKind.RESTART_LLAMA_SERVER}
)

# Kinds that send signals to / re-prioritize processes. These must honor the
# ProtectedWorkload guard (never target a protected PID). FREE_STALE_CACHE is in
# this set because it, too, issues a real ``kill`` against a PID (freeing the
# VRAM a dead context still holds) — leaving it out was a hole in the never-kill
# invariant: the manager pre-apply guard and the actuator backstop both key off
# this set, so any signalling kind MUST be listed here.
PROCESS_TOUCHING_KINDS: frozenset[ActionKind] = frozenset(
    {
        ActionKind.RENICE_PROCESS,
        ActionKind.SET_CPU_AFFINITY,
        ActionKind.KILL_ORPHAN_PROCESS,
        ActionKind.FREE_STALE_CACHE,
    }
)


@dataclass(frozen=True)
class ProtectedWorkload:
    """The live compute task that must survive every remediation.

    ``pids`` are the workload's process ids (e.g. the llama-server PID, or the
    training job's ranks). ``label`` is a human/scheduler identifier
    (a k8s pod, a Slurm job). A remediation that would target any of these is
    refused — this is the structural enforcement of the never-kill invariant.
    """

    pids: frozenset[int] = frozenset()
    label: str = ""

    def protects(self, pid: int | None) -> bool:
        return pid is not None and pid in self.pids


@dataclass(frozen=True)
class ActionRequest:
    """A fully-specified, ready-to-run remediation instance."""

    kind: ActionKind
    action_class: ActionClass
    node_id: str
    target: str  # gpu index, pid, mig device, server url — kind-specific
    params: dict = field(default_factory=dict)
    job_id: str | None = None
    protected: ProtectedWorkload = field(default_factory=ProtectedWorkload)
    # When True the actuator builds the real command but does NOT execute it
    # (dry-run / mode gate). When False it actually runs.
    dry_run: bool = True
    reversible: bool = True


@dataclass(frozen=True)
class ActionResult:
    """Outcome of an actuator ``apply`` / ``rollback``."""

    ok: bool
    kind: ActionKind
    # The exact command line (or NVML call descriptor) — recorded for audit even
    # when not executed, so dry-run shows operators precisely what *would* run.
    command: str
    message: str = ""
    executed: bool = False  # did a real subprocess/NVML write happen?
    no_op: bool = False  # idempotent skip: already at target
    error: str | None = None


@dataclass(frozen=True)
class ActionContext:
    """Everything a spec needs to build params and judge recovery."""

    node_id: str
    gpu_index: int
    verdict_value: str
    metrics: dict
    pre: WindowSummary  # window summary at trigger time
    protected: ProtectedWorkload
    job_id: str | None = None
    # Free-form knobs from RemediationConfig (e.g. power headroom %, target nice).
    knobs: dict = field(default_factory=dict)


# A spec builds the request params from the live context, and later judges
# whether the post-apply window shows recovery relative to the pre window.
ParamBuilder = Callable[[ActionContext], dict]
RecoveryCheck = Callable[[WindowSummary, WindowSummary], bool]


@dataclass(frozen=True)
class ActionSpec:
    """The pluggable unit in the registry: root cause -> how to fix it."""

    root_cause: RootCause
    kind: ActionKind
    action_class: ActionClass
    reversible: bool
    summary: str
    build_params: ParamBuilder
    recovered: RecoveryCheck
    requires_checkpoint: bool = False
    requires_drain: bool = False
    # An irreversible non-disruptive action (e.g. killing an orphan) cannot be
    # rolled back, so it is held to a stricter pre-condition before it may fire.
    irreversible_guard: Callable[[ActionContext], bool] | None = None

    def __post_init__(self) -> None:
        # Structural invariant: a non-disruptive spec can never carry a
        # workload-lifecycle kind. Catch mis-tagging at construction, not in prod.
        if (
            self.action_class is ActionClass.NON_DISRUPTIVE
            and self.kind in WORKLOAD_LIFECYCLE_KINDS
        ):
            raise ValueError(
                f"NON_DISRUPTIVE spec may not use workload-lifecycle kind {self.kind}"
            )
