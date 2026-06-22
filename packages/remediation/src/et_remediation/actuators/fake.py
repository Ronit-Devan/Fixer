"""In-memory fake actuator + telemetry model for the simulation harness.

This is what makes the whole guarded loop testable without a GPU. The
``FakeTelemetryModel`` is a mutable bag of the same fields a real ``Snapshot``
carries; ``sample()`` freezes it into a read-only sample the telemetry summarizer
understands. The ``FakeActuator``:

  * records every ``apply`` / ``rollback`` call (so tests assert exactly what ran),
  * enforces the protected-workload guard for process-touching kinds (so we can
    prove the live task is never targeted),
  * optionally mutates the telemetry model toward *recovery* on apply, so the
    verify loop can be driven to either confirm or roll back deterministically,
  * supports idempotency: applying a value that already holds is a no-op.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from et_remediation.actions import (
    PROCESS_TOUCHING_KINDS,
    ActionKind,
    ActionRequest,
    ActionResult,
)
from et_remediation.actuators.base import Actuator, ActuationState


@dataclass(frozen=True)
class FakeSample:
    """A frozen telemetry reading the ``telemetry.summarize`` reducer can read."""

    util_pct: float | None = None
    mem_used_ratio: float | None = None
    clock_ratio: float | None = None
    temp_c: float | None = None
    power_w: float | None = None
    power_limit_w: float | None = None
    requests_processing: float | None = None
    kv_cache_usage_ratio: float | None = None


@dataclass
class FakeTelemetryModel:
    """Mutable GPU/inference state a sim drives and the actuator nudges."""

    util_pct: float = 5.0
    mem_used_ratio: float = 0.45
    clock_ratio: float = 0.95
    temp_c: float = 45.0
    power_w: float = 120.0
    power_limit_w: float = 300.0
    requests_processing: float = 0.0
    kv_cache_usage_ratio: float = 0.1

    def sample(self) -> FakeSample:
        return FakeSample(
            util_pct=self.util_pct,
            mem_used_ratio=self.mem_used_ratio,
            clock_ratio=self.clock_ratio,
            temp_c=self.temp_c,
            power_w=self.power_w,
            power_limit_w=self.power_limit_w,
            requests_processing=self.requests_processing,
            kv_cache_usage_ratio=self.kv_cache_usage_ratio,
        )

    def window(self, n: int = 3) -> list[FakeSample]:
        """A flat window of identical current samples (for synchronous tests)."""
        s = self.sample()
        return [replace(s) for _ in range(n)]

    def apply_recovery(self, kind: ActionKind) -> None:
        """Nudge the relevant signal toward healthy, as a real fix would."""
        if kind in (ActionKind.SET_POWER_LIMIT, ActionKind.LOCK_CLOCKS, ActionKind.RESET_CLOCKS):
            self.clock_ratio = 0.97
            self.temp_c = max(35.0, self.temp_c - 8.0)
        elif kind in (ActionKind.KILL_ORPHAN_PROCESS, ActionKind.FREE_STALE_CACHE):
            self.mem_used_ratio = max(0.05, self.mem_used_ratio - 0.40)
        elif kind in (ActionKind.RENICE_PROCESS, ActionKind.SET_CPU_AFFINITY):
            self.util_pct = min(100.0, self.util_pct + 35.0)
        elif kind in (ActionKind.CONFIGURE_MPS, ActionKind.CONFIGURE_MIG):
            self.util_pct = min(100.0, self.util_pct + 25.0)
        elif kind in (
            ActionKind.DRAIN_AND_RESTART_WORKLOAD,
            ActionKind.RESTART_LLAMA_SERVER,
        ):
            self.util_pct = min(100.0, self.util_pct + 40.0)
            self.kv_cache_usage_ratio = 0.4


@dataclass
class RecordedCall:
    op: str  # "apply" | "rollback"
    request: ActionRequest
    executed: bool
    no_op: bool


class FakeActuator(Actuator):
    backend = "fake"

    def __init__(
        self,
        model: FakeTelemetryModel | None = None,
        *,
        recover_on_apply: bool = True,
        fail_kinds: set[ActionKind] | None = None,
    ) -> None:
        self.model = model or FakeTelemetryModel()
        self.recover_on_apply = recover_on_apply
        self.fail_kinds = fail_kinds or set()
        self.calls: list[RecordedCall] = []
        # PIDs we were ever asked to signal — tests assert the protected set
        # never appears here.
        self.targeted_pids: list[int] = []
        self.protection_violations: int = 0

    def capabilities(self) -> set[ActionKind]:
        return set(ActionKind)  # the fake can stand in for any backend

    def snapshot_state(self, req: ActionRequest) -> ActuationState:
        # Capture the current value of whatever this kind would change.
        values = {
            "clock_ratio": self.model.clock_ratio,
            "power_limit_w": self.model.power_limit_w,
            "mem_used_ratio": self.model.mem_used_ratio,
            "util_pct": self.model.util_pct,
        }
        return ActuationState(kind=req.kind, target=req.target, values=values)

    def _is_no_op(self, req: ActionRequest) -> bool:
        # Idempotency: setting the power limit to its current value is a no-op.
        if req.kind is ActionKind.SET_POWER_LIMIT:
            want = req.params.get("power_limit_w")
            return want is not None and float(want) == float(self.model.power_limit_w)
        return False

    def apply(self, req: ActionRequest) -> ActionResult:
        # Enforce the never-kill guard for any process-touching kind.
        if req.kind in PROCESS_TOUCHING_KINDS:
            pid = req.params.get("pid")
            if pid is not None:
                self.targeted_pids.append(int(pid))
            try:
                self.guard_protected(req, int(pid) if pid is not None else None)
            except Exception:
                self.protection_violations += 1
                raise

        no_op = self._is_no_op(req)
        executed = (not req.dry_run) and (not no_op)

        if executed and req.kind in self.fail_kinds:
            self.calls.append(RecordedCall("apply", req, executed=True, no_op=False))
            return ActionResult(
                ok=False,
                kind=req.kind,
                command=f"fake:{req.kind.value}",
                message="injected failure",
                executed=True,
                error="injected",
            )

        if executed:
            # Reflect the change in the model.
            if req.kind is ActionKind.SET_POWER_LIMIT and "power_limit_w" in req.params:
                self.model.power_limit_w = float(req.params["power_limit_w"])
            if self.recover_on_apply:
                self.model.apply_recovery(req.kind)

        self.calls.append(RecordedCall("apply", req, executed=executed, no_op=no_op))
        return ActionResult(
            ok=True,
            kind=req.kind,
            command=f"fake:{req.kind.value} {req.params}",
            message="no-op (already at target)" if no_op else "applied",
            executed=executed,
            no_op=no_op,
        )

    def rollback(self, req: ActionRequest, prior: ActuationState) -> ActionResult:
        executed = not req.dry_run
        if executed:
            # Restore the captured prior values.
            self.model.clock_ratio = prior.values.get("clock_ratio", self.model.clock_ratio)
            self.model.power_limit_w = prior.values.get(
                "power_limit_w", self.model.power_limit_w
            )
            self.model.util_pct = prior.values.get("util_pct", self.model.util_pct)
            # Memory cannot be un-freed; nothing to restore for kill/free.
        self.calls.append(RecordedCall("rollback", req, executed=executed, no_op=False))
        return ActionResult(
            ok=True,
            kind=req.kind,
            command=f"fake:rollback {req.kind.value}",
            message="rolled back",
            executed=executed,
        )

    # -- test helpers --------------------------------------------------------

    def applied_kinds(self) -> list[ActionKind]:
        return [c.request.kind for c in self.calls if c.op == "apply"]

    def executed_kinds(self) -> list[ActionKind]:
        return [c.request.kind for c in self.calls if c.op == "apply" and c.executed]

    def rolled_back_kinds(self) -> list[ActionKind]:
        return [c.request.kind for c in self.calls if c.op == "rollback"]
