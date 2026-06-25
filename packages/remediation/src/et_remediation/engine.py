"""RemediationManager: the guarded auto-apply state machine.

Plugs in exactly parallel to ``et_monitor.AlertManager`` — the monitor calls
``observe(diagnosis, window, now)`` once per tick. The manager:

  1. maps the verdict to a canonical RootCause (no coupling to either product);
  2. resolves candidate strategies and picks the first applicable;
  3. classifies NON-DISRUPTIVE vs DISRUPTIVE;
  4. for DISRUPTIVE: opens a human-gated ApprovalRequest — never executes;
  5. for NON-DISRUPTIVE: runs the full guarded path through policy + circuit
     breaker, applies, then watches a bounded telemetry window for recovery and
     either CONFIRMS or AUTO-ROLLS-BACK;
  6. records every step to the audit log.

It is a per-node state machine: while an action is VERIFYING on a node, no new
action starts there (blast radius = 1 in flight). Time is injected via
``now``/``now_fn`` so the whole loop is testable synchronously.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Sequence

from et_remediation.actions import (
    PROCESS_TOUCHING_KINDS,
    ActionClass,
    ActionContext,
    ActionKind,
    ActionRequest,
    ActionSpec,
    ProtectedWorkload,
)
from et_remediation.actuators.base import Actuator, ActuationState
from et_remediation.audit import AuditLog, AuditRecord, Phase
from et_remediation.breaker import CircuitBreaker
from et_remediation.classifier import classify
from et_remediation.config import RemediationConfig, RemediationMode
from et_remediation.diagnosis import DiagnosisLike, normalize
from et_remediation.policy import PolicyEngine
from et_remediation.registry import ActionRegistry
from et_remediation.rootcause import (
    RootCause,
    map_engine_verdict,
    map_from_metrics,
    map_monitor_verdict,
)
from et_remediation.telemetry import WindowSummary, summarize


class RunState(str, Enum):
    NORMAL = "normal"
    VERIFYING = "verifying"


class OutcomeKind(str, Enum):
    OFF = "off"
    HEALTHY = "healthy"  # nothing to do
    DEBOUNCING = "debouncing"  # actionable cause seen, awaiting confirmation
    ADVISED = "advised"  # advise-only plan emitted
    PLANNED = "planned"  # dry-run: real command built, not executed
    APPLIED = "applied"  # non-disruptive action applied; verifying
    VERIFYING = "verifying"  # still watching for recovery
    CONFIRMED = "confirmed"  # recovery confirmed
    ROLLED_BACK = "rolled_back"  # no recovery -> reverted
    NO_OP = "no_op"  # already at target / idempotent skip
    BLOCKED = "blocked"  # breaker / caps / kill-switch blocked actuation
    APPROVAL_REQUIRED = "approval_required"  # disruptive: awaiting human
    APPLY_FAILED = "apply_failed"


@dataclass
class ApprovalRequest:
    id: str
    node_id: str
    job_id: str | None
    root_cause: str
    verdict: str
    kind: ActionKind
    summary: str
    command_preview: str
    requires_checkpoint: bool
    requires_drain: bool
    created_at: float
    status: str = "pending"  # pending | approved | applied | rejected | failed
    # Stored so approve() can execute exactly what was previewed.
    _spec: ActionSpec | None = field(default=None, repr=False)
    _request: ActionRequest | None = field(default=None, repr=False)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "node_id": self.node_id,
            "job_id": self.job_id,
            "root_cause": self.root_cause,
            "verdict": self.verdict,
            "kind": self.kind.value,
            "summary": self.summary,
            "command_preview": self.command_preview,
            "requires_checkpoint": self.requires_checkpoint,
            "requires_drain": self.requires_drain,
            "created_at": self.created_at,
            "status": self.status,
        }


@dataclass
class Outcome:
    kind: OutcomeKind
    root_cause: RootCause = RootCause.NONE
    detail: str = ""
    action_kind: ActionKind | None = None
    approval: ApprovalRequest | None = None


@dataclass
class _NodeRun:
    state: RunState = RunState.NORMAL
    spec: ActionSpec | None = None
    request: ActionRequest | None = None
    prior: ActuationState | None = None
    pre: WindowSummary | None = None
    deadline: float = 0.0
    applied_at: float = 0.0  # when the fix was applied; only later samples judge recovery


def _is_post_apply(sample: object, applied_at: float) -> bool:
    """True if a telemetry sample was taken after the fix was applied.

    Samples that expose a ``timestamp_s`` (the live monitor's Snapshot) are
    kept only when strictly newer than ``applied_at`` — this excludes the
    apply-tick reading itself, which still reflects the pre-fix state. Samples
    with no timestamp (the simulation's FakeSample) are treated as fresh so the
    sim harness and any timestamp-less telemetry source still work.
    """
    ts = getattr(sample, "timestamp_s", None)
    return ts is None or ts > applied_at


class RemediationManager:
    def __init__(
        self,
        registry: ActionRegistry,
        config: RemediationConfig,
        actuators: Sequence[Actuator],
        *,
        audit: AuditLog | None = None,
        breaker: CircuitBreaker | None = None,
        protected: ProtectedWorkload | None = None,
        now_fn: Callable[[], float] = time.monotonic,
        node_id: str = "node-0",
        gpu_index: int = 0,
        job_id: str | None = None,
        fleet=None,
    ) -> None:
        self.registry = registry
        self.config = config
        self.actuators = list(actuators)
        self.audit = audit or AuditLog(jsonl_path=config.audit_path)
        self.breaker = breaker or CircuitBreaker(config.caps)
        self.policy = PolicyEngine(config)
        self.protected = protected or ProtectedWorkload(
            pids=frozenset(config.protected_pids), label=config.protected_label
        )
        self._now_fn = now_fn
        self.node_id = node_id
        self.gpu_index = gpu_index
        self.job_id = job_id
        # Optional fleet-wide blast-radius limiter shared across per-GPU managers.
        self.fleet = fleet
        self._runs: dict[str, _NodeRun] = {}
        self.approvals: dict[str, ApprovalRequest] = {}
        self._approval_seq = 0
        # Cap on retained RESOLVED approvals (pending are always kept). Stops the
        # dict — and the O(n) pending scan in status() — from growing without
        # bound on a box that hits a disruptive cause repeatedly over weeks.
        self._approval_history_cap = 50
        # Trigger debounce: how many consecutive observes have seen this cause.
        self._debounce_rc: RootCause = RootCause.NONE
        self._debounce_count: int = 0
        # The manager is touched from two threads in production: the monitor
        # sampling loop (observe) and the HTTP layer (approve/set_mode/status).
        # A reentrant lock serializes them so shared state stays consistent.
        self._lock = threading.RLock()

    # -- public knobs --------------------------------------------------------

    @property
    def mode(self) -> RemediationMode:
        return self.config.mode

    def set_mode(self, mode: RemediationMode) -> None:
        """The kill-switch. Flipping to OFF/ADVISE forces advise-only at once."""
        with self._lock:
            self.config.mode = mode

    def _run(self, node: str) -> _NodeRun:
        return self._runs.setdefault(node, _NodeRun())

    def _actuator_for(self, kind: ActionKind) -> Actuator | None:
        for a in self.actuators:
            if kind in a.capabilities():
                return a
        return None

    # -- the tick entrypoint -------------------------------------------------

    def observe(
        self,
        diagnosis: DiagnosisLike,
        window: Sequence[object],
        now: float | None = None,
        *,
        source: str = "monitor",
    ) -> Outcome:
        with self._lock:
            return self._observe(diagnosis, window, now, source=source)

    def _observe(
        self,
        diagnosis: DiagnosisLike,
        window: Sequence[object],
        now: float | None = None,
        *,
        source: str = "monitor",
    ) -> Outcome:
        now = self._now_fn() if now is None else now
        nd = normalize(diagnosis, source)
        run = self._run(self.node_id)
        self._prune_approvals()

        # 1) Resolve an in-flight verify FIRST — even if the kill-switch just
        #    flipped to OFF. A fix we already applied must always reach its
        #    confirm-or-rollback (and release its fleet slot); abandoning it
        #    mid-verify would freeze the run and leak the slot.
        if run.state is RunState.VERIFYING:
            return self._resolve_verify(run, window, now, nd)

        if self.config.mode is RemediationMode.OFF:
            return Outcome(OutcomeKind.OFF)

        # 2) Map the verdict to a canonical root cause.
        rc = self._root_cause(nd, source)
        if rc is RootCause.NONE:
            self._debounce_rc = RootCause.NONE
            self._debounce_count = 0
            return Outcome(OutcomeKind.HEALTHY, RootCause.NONE)

        # 2a) A PREDICTED (heat-imminent) throttle must not auto-raise the power
        #     limit — that would add heat and worsen the very thing we predicted.
        #     Surface it as advice (the early warning is the value); the reactive
        #     path still auto-acts if it's genuinely power-capped once it lands.
        if nd.predicted and rc is RootCause.THERMAL_POWER_THROTTLE:
            self._audit(Phase.ADVISE, nd, rc, None, None, "predicted_thermal_advise_only")
            return Outcome(
                OutcomeKind.ADVISED, rc,
                "predicted throttle; raising power could worsen heat — advise/cool instead",
            )

        # 2b) Trigger debounce: require the same cause for N consecutive observes
        #     before acting, so a single noisy verdict never drives actuation.
        if rc == self._debounce_rc:
            self._debounce_count += 1
        else:
            self._debounce_rc = rc
            self._debounce_count = 1
        if self._debounce_count < self.config.trigger_debounce:
            return Outcome(
                OutcomeKind.DEBOUNCING, rc,
                f"confirming cause ({self._debounce_count}/{self.config.trigger_debounce})",
            )

        ctx = ActionContext(
            node_id=self.node_id,
            gpu_index=self.gpu_index,
            verdict_value=nd.verdict_value,
            metrics=nd.metrics,
            pre=summarize(window),
            protected=self.protected,
            job_id=self.job_id,
            knobs=dict(self.config.knobs),
        )

        # 3) Pick the first applicable strategy.
        chosen = self._select(rc, ctx)
        if chosen is None:
            self._audit(Phase.ADVISE, nd, rc, None, None, "no_applicable_strategy")
            return Outcome(OutcomeKind.ADVISED, rc, "no applicable strategy")
        spec, params = chosen
        classify(spec)  # enforce invariants (raises on a mis-tagged spec)

        self._audit(Phase.TRIGGER, nd, rc, spec.kind, spec.action_class, "triggered")

        # 4/5) Branch on class via the static policy.
        decision = self.policy.decide(spec, self.protected, in_flight_on_node=0)

        if decision.outcome == "approval":
            return self._open_approval(spec, params, ctx, nd, rc, now)
        if decision.outcome in ("advise", "blocked", "skip"):
            phase = Phase.BLOCKED if decision.outcome == "blocked" else Phase.ADVISE
            self._audit(phase, nd, rc, spec.kind, spec.action_class, decision.reason)
            kind = OutcomeKind.BLOCKED if decision.outcome == "blocked" else OutcomeKind.ADVISED
            return Outcome(kind, rc, decision.reason, spec.kind)

        # decision.outcome in ("actuate", "dry_run") -> non-disruptive path.
        return self._apply_non_disruptive(spec, params, ctx, nd, rc, now, decision.outcome)

    # -- non-disruptive guarded apply ----------------------------------------

    def _apply_non_disruptive(
        self, spec, params, ctx, nd, rc, now, outcome
    ) -> Outcome:
        # Last manager-level enforcement of the never-kill invariant: refuse a
        # process-touching action whose target PID is the protected workload,
        # before it ever reaches the actuator. (The actuator's guard_protected is
        # the final backstop; this keeps the refusal a clean advise, not a raise.)
        if spec.kind in PROCESS_TOUCHING_KINDS and self.protected.protects(params.get("pid")):
            self._audit(
                Phase.BLOCKED, nd, rc, spec.kind, spec.action_class,
                "would_target_protected_workload", {"pid": params.get("pid")},
            )
            return Outcome(OutcomeKind.BLOCKED, rc, "would target protected workload", spec.kind)

        # Resolve the actuator BEFORE consuming any breaker/fleet budget, so a
        # missing actuator is a clean advise that costs nothing.
        actuator = self._actuator_for(spec.kind)
        if actuator is None:
            self._audit(Phase.ADVISE, nd, rc, spec.kind, spec.action_class, "no_actuator")
            return Outcome(OutcomeKind.ADVISED, rc, "no actuator for kind", spec.kind)

        execute = outcome == "actuate"
        # Fleet blast-radius: claim a shared slot before actuating for real, so
        # we never auto-apply on more than N GPUs/nodes across the fleet at once.
        fleet_held = False
        if execute and self.fleet is not None:
            if not self.fleet.try_acquire(self.node_id):
                self._audit(
                    Phase.BLOCKED, nd, rc, spec.kind, spec.action_class,
                    "fleet_blast_radius", {"active": self.fleet.active_count()},
                )
                return Outcome(OutcomeKind.BLOCKED, rc, "fleet blast-radius cap", spec.kind)
            fleet_held = True

        # Circuit breaker is the LAST gate before applying, so a granted HALF_OPEN
        # trial is always followed by a real apply (never stranded mid-state). If
        # it blocks, release the fleet slot we just took.
        bd = self.breaker.allow(self.node_id, now)
        if not bd.allowed:
            self._fleet_release(fleet_held)
            self._audit(
                Phase.BLOCKED, nd, rc, spec.kind, spec.action_class,
                f"breaker:{bd.reason}", {"breaker_state": bd.state.value},
            )
            return Outcome(OutcomeKind.BLOCKED, rc, f"breaker_{bd.reason}", spec.kind)

        req = ActionRequest(
            kind=spec.kind,
            action_class=ActionClass.NON_DISRUPTIVE,
            node_id=self.node_id,
            target=str(self.gpu_index),
            params=params,
            job_id=self.job_id,
            protected=self.protected,
            dry_run=not execute,
            reversible=spec.reversible,
        )

        prior = actuator.snapshot_state(req)
        result = actuator.apply(req)
        # Count only really-executed applies toward rate + flap (an idempotent
        # no-op or a dry-run build is not an "action" for the breaker).
        if result.executed:
            self.breaker.record_apply(self.node_id, spec.kind.value, now)

        self._audit(
            Phase.APPLY, nd, rc, spec.kind, spec.action_class,
            "applied" if result.ok else "apply_failed",
            {"command": result.command, "executed": result.executed, "no_op": result.no_op},
        )

        if not result.ok:
            self.breaker.record_failure(self.node_id, now)
            self._fleet_release(fleet_held)
            return Outcome(OutcomeKind.APPLY_FAILED, rc, result.error or "", spec.kind)
        if result.no_op:
            self._fleet_release(fleet_held)
            return Outcome(OutcomeKind.NO_OP, rc, "already at target", spec.kind)
        if not execute:
            # dry-run: real command built + logged, nothing executed, no verify.
            return Outcome(OutcomeKind.PLANNED, rc, result.command, spec.kind)

        # Enter the verify window. Reset the debounce so that if this fix later
        # rolls back, the cause must be re-confirmed for trigger_debounce ticks
        # before we act again (no immediate re-fire after a rollback).
        self._debounce_rc = RootCause.NONE
        self._debounce_count = 0
        run = self._run(self.node_id)
        run.state = RunState.VERIFYING
        run.spec = spec
        run.request = req
        run.prior = prior
        run.pre = ctx.pre
        run.applied_at = now
        run.deadline = now + self.config.verify_window_s
        return Outcome(OutcomeKind.APPLIED, rc, "verifying recovery", spec.kind)

    def _fleet_release(self, held: bool) -> None:
        if held and self.fleet is not None:
            self.fleet.release(self.node_id)

    def _resolve_verify(self, run, window, now, nd) -> Outcome:
        assert run.spec is not None and run.request is not None and run.prior is not None
        spec = run.spec
        kind = spec.kind
        rc = spec.root_cause

        # Judge recovery ONLY on samples taken strictly after the fix was applied.
        # The live monitor hands us a rolling window that still contains pre-fix
        # readings; comparing run.pre (captured at apply) against a window that
        # overlaps the pre-fix period caused both false CONFIRM and false ROLLBACK.
        # Samples without a timestamp (the sim's FakeSample) are treated as fresh.
        post_window = [s for s in window if _is_post_apply(s, run.applied_at)]
        post = summarize(post_window)
        enough = post.n >= self.config.min_verify_samples
        deadline_passed = now >= run.deadline

        # Confirming ALWAYS requires a minimum of post-apply samples, even at the
        # deadline — a single reading must never confirm a fix. If the deadline
        # arrives under-sampled, we roll back (the conservative choice).
        if enough and post.n > 0 and bool(spec.recovered(run.pre, post)):
            run.state = RunState.NORMAL
            self.breaker.record_success(self.node_id, now)
            self._fleet_release(True)  # free the fleet slot this GPU held
            self._audit(
                Phase.VERIFY, nd, rc, kind, spec.action_class,
                "recovered", {"post_samples": post.n},
            )
            self._clear(run)
            return Outcome(OutcomeKind.CONFIRMED, rc, "recovery confirmed", kind)

        if deadline_passed:
            actuator = self._actuator_for(kind)
            if actuator is not None:
                actuator.rollback(run.request, run.prior)
            self.breaker.record_failure(self.node_id, now)
            self._fleet_release(True)
            self._audit(
                Phase.ROLLBACK, nd, rc, kind, spec.action_class,
                "rolled_back", {"reversible": spec.reversible, "post_samples": post.n},
            )
            run.state = RunState.NORMAL
            self._clear(run)
            return Outcome(OutcomeKind.ROLLED_BACK, rc, "no recovery; reverted", kind)

        return Outcome(OutcomeKind.VERIFYING, rc, "watching for recovery", kind)

    @staticmethod
    def _clear(run: _NodeRun) -> None:
        run.spec = run.request = run.prior = run.pre = None

    # -- disruptive approval gate --------------------------------------------

    def _open_approval(self, spec, params, ctx, nd, rc, now) -> Outcome:
        # One pending request per (node, root_cause); don't spam every tick.
        for ar in self.approvals.values():
            if ar.node_id == self.node_id and ar.root_cause == rc.value and ar.status == "pending":
                return Outcome(OutcomeKind.APPROVAL_REQUIRED, rc, "already pending", spec.kind, ar)

        actuator = self._actuator_for(spec.kind)
        req = ActionRequest(
            kind=spec.kind,
            action_class=ActionClass.DISRUPTIVE,
            node_id=self.node_id,
            target=str(ctx.knobs.get("llama_url", self.gpu_index)),
            params=params,
            job_id=self.job_id,
            protected=self.protected,
            dry_run=True,  # NEVER executed at request time
            reversible=spec.reversible,
        )
        preview = ""
        if actuator is not None and hasattr(actuator, "build_command"):
            try:
                preview = " ".join(actuator.build_command(req))  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                preview = ""
        if not preview and actuator is not None and hasattr(actuator, "build_argv"):
            try:
                preview = " ".join(actuator.build_argv(req))  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                preview = ""
        if not preview:
            # Generic fallback so the audit/dashboard always shows *something*
            # the operator is approving, even behind a backend without a builder.
            preview = f"{spec.kind.value} {params}"

        self._approval_seq += 1
        ar = ApprovalRequest(
            id=f"appr-{self.node_id}-{self._approval_seq}",
            node_id=self.node_id,
            job_id=self.job_id,
            root_cause=rc.value,
            verdict=nd.verdict_value,
            kind=spec.kind,
            summary=spec.summary,
            command_preview=preview,
            requires_checkpoint=spec.requires_checkpoint,
            requires_drain=spec.requires_drain,
            created_at=now,
            _spec=spec,
            _request=req,
        )
        self.approvals[ar.id] = ar
        self._audit(
            Phase.APPROVAL_REQUESTED, nd, rc, spec.kind, spec.action_class,
            "approval_requested", {"approval_id": ar.id, "command_preview": preview},
        )
        return Outcome(OutcomeKind.APPROVAL_REQUIRED, rc, "human approval required", spec.kind, ar)

    def approve(self, approval_id: str, now: float | None = None) -> Outcome:
        """Human-gated execution of a disruptive action. Never called by the loop.

        Idempotent against double-submit: the request is *claimed* (status flips
        to "applying") under the lock, so a second concurrent approve sees a
        non-pending request and is refused. The slow drain+restart then runs
        OUTSIDE the lock so it never stalls the sampling loop's observe().
        """
        now = self._now_fn() if now is None else now
        with self._lock:
            ar = self.approvals.get(approval_id)
            if ar is None or ar.status != "pending":
                return Outcome(OutcomeKind.BLOCKED, RootCause.NONE, "no such pending approval")
            if self.config.mode is RemediationMode.OFF:
                return Outcome(OutcomeKind.BLOCKED, RootCause.NONE, "mode_off")
            assert ar._spec is not None and ar._request is not None
            actuator = self._actuator_for(ar.kind)
            if actuator is None:
                ar.status = "failed"
                return Outcome(OutcomeKind.APPLY_FAILED, RootCause(ar.root_cause), "no actuator")
            ar.status = "applying"  # claim it before releasing the lock
            exec_req = ActionRequest(
                kind=ar._request.kind,
                action_class=ActionClass.DISRUPTIVE,
                node_id=ar.node_id,
                target=ar._request.target,
                params=ar._request.params,
                job_id=ar.job_id,
                protected=self.protected,
                dry_run=False,  # approved -> may execute (gated by actuator runner)
                reversible=ar._spec.reversible,
            )

        # Drain + restart runs without holding the lock (it can take seconds).
        result = actuator.apply(exec_req)

        with self._lock:
            ar.status = "applied" if result.ok else "failed"
            self._audit_raw(
                Phase.APPROVAL_APPLIED, ar.root_cause, ar.verdict, ar.kind,
                ActionClass.DISRUPTIVE, "applied" if result.ok else "failed",
                now, {"approval_id": ar.id, "command": result.command, "executed": result.executed},
            )
        kind = OutcomeKind.CONFIRMED if result.ok else OutcomeKind.APPLY_FAILED
        return Outcome(kind, RootCause(ar.root_cause), result.message, ar.kind, ar)

    def reject(self, approval_id: str) -> bool:
        with self._lock:
            ar = self.approvals.get(approval_id)
            if ar is None or ar.status != "pending":
                return False
            ar.status = "rejected"
            return True

    def pending_approvals(self) -> list[ApprovalRequest]:
        with self._lock:
            return [a for a in self.approvals.values() if a.status == "pending"]

    def _prune_approvals(self) -> None:
        """Keep all pending approvals + the most recent ``_approval_history_cap``
        resolved ones; drop older resolved entries so the dict stays bounded over
        a long run. Cheap: only sorts when the cap is actually exceeded."""
        resolved = [(aid, ar) for aid, ar in self.approvals.items() if ar.status != "pending"]
        excess = len(resolved) - self._approval_history_cap
        if excess <= 0:
            return
        resolved.sort(key=lambda kv: kv[1].created_at)  # oldest first
        for aid, _ in resolved[:excess]:
            del self.approvals[aid]

    # -- helpers -------------------------------------------------------------

    def _root_cause(self, nd, source: str) -> RootCause:
        if source == "engine":
            rc = map_engine_verdict(nd.verdict_value, nd.metrics)
        else:
            rc = map_monitor_verdict(nd.verdict_value, nd.metrics)
        if rc is RootCause.NONE:
            rc = map_from_metrics(nd.metrics)  # detect causes w/o a dedicated verdict
        return rc

    def _select(self, rc: RootCause, ctx: ActionContext):
        for spec in self.registry.resolve(rc):
            try:
                params = spec.build_params(ctx)
            except (KeyError, ValueError):
                continue  # preconditions not met -> try the next candidate
            if spec.irreversible_guard is not None and not spec.irreversible_guard(ctx):
                continue  # irreversible action whose strict guard failed
            return spec, params
        return None

    def _audit(self, phase, nd, rc, kind, action_class, decision, detail=None) -> AuditRecord:
        return self._audit_raw(
            phase, rc.value, nd.verdict_value, kind, action_class, decision,
            self._now_fn(), detail,
        )

    def _audit_raw(
        self, phase, root_cause, verdict, kind, action_class, decision, ts, detail=None
    ) -> AuditRecord:
        return self.audit.record(
            AuditRecord(
                ts=ts,
                phase=phase,
                node_id=self.node_id,
                job_id=self.job_id,
                root_cause=root_cause,
                verdict=verdict,
                action_kind=kind.value if kind else None,
                action_class=action_class.value if action_class else None,
                mode=self.config.mode.value,
                decision=decision,
                detail=detail or {},
            )
        )

    # -- read API for the dashboard ------------------------------------------

    def status(self) -> dict:
        with self._lock:
            run = self._run(self.node_id)
            return self._status_locked(run)

    def _status_locked(self, run: _NodeRun) -> dict:
        return {
            "mode": self.config.mode.value,
            "node_id": self.node_id,
            "state": run.state.value,
            "in_flight_kind": run.spec.kind.value if run.spec else None,
            "breaker_state": self.breaker.state(self.node_id).value,
            "pending_approvals": [a.to_dict() for a in self.pending_approvals()],
            "enabled_non_disruptive": self.config.enable_non_disruptive,
            "verify_window_s": self.config.verify_window_s,
        }
