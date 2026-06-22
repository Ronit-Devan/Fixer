"""Data-center backend: nvidia-smi / NVML / DCGM + OS + K8s/Slurm.

Builds *real* commands for every action and runs them through the mode-gated
``CommandRunner`` (so advise/dry-run build-and-log while AUTO executes). Every
non-disruptive action here preserves the running workload:

  * power/clock tuning touches the device, not any process;
  * renice/affinity re-prioritize the *data-loader worker* procs, never the
    protected training/inference PIDs (guarded);
  * MPS/MIG reconfigure scheduling on a card with free capacity;
  * orphan-kill / free-stale only ever signals a PID proven NOT to be in the
    protected set.

The one disruptive kind (DRAIN_AND_RESTART_WORKLOAD) builds a K8s/Slurm
drain+restart command but is only ever reached through the approval gate.
"""

from __future__ import annotations

import logging

from et_remediation.actions import (
    PROCESS_TOUCHING_KINDS,
    ActionKind,
    ActionRequest,
    ActionResult,
)
from et_remediation.actuators.base import Actuator, ActuationState, CommandRunner

log = logging.getLogger(__name__)


class DataCenterActuator(Actuator):
    backend = "datacenter"

    def __init__(self, runner: CommandRunner | None = None) -> None:
        # Default runner does NOT execute — a DataCenterActuator only ever runs
        # real commands when the manager hands it an executing runner (AUTO).
        self.runner = runner or CommandRunner(execute=False)

    def _runner_for(self, req: ActionRequest) -> CommandRunner:
        """Per-request execution gate: dry_run forces build-only regardless of
        how the actuator's runner was configured. Effective execute =
        runner.execute AND not req.dry_run."""
        return self.runner if not req.dry_run else CommandRunner(execute=False)

    def capabilities(self) -> set[ActionKind]:
        return {
            ActionKind.SET_POWER_LIMIT,
            ActionKind.LOCK_CLOCKS,
            ActionKind.RESET_CLOCKS,
            ActionKind.RENICE_PROCESS,
            ActionKind.SET_CPU_AFFINITY,
            ActionKind.CONFIGURE_MPS,
            ActionKind.CONFIGURE_MIG,
            ActionKind.KILL_ORPHAN_PROCESS,
            ActionKind.FREE_STALE_CACHE,
            ActionKind.DRAIN_AND_RESTART_WORKLOAD,
        }

    # -- command construction ------------------------------------------------

    def build_command(self, req: ActionRequest) -> list[str]:
        """Render the real command for a request. Pure; no execution."""
        gpu = str(req.target)
        p = req.params
        k = req.kind
        if k is ActionKind.SET_POWER_LIMIT:
            return ["nvidia-smi", "-i", gpu, "-pl", str(int(p["power_limit_w"]))]
        if k is ActionKind.LOCK_CLOCKS:
            return ["nvidia-smi", "-i", gpu, "-lgc", f"{int(p['min_mhz'])},{int(p['max_mhz'])}"]
        if k is ActionKind.RESET_CLOCKS:
            return ["nvidia-smi", "-i", gpu, "-rgc"]
        if k is ActionKind.RENICE_PROCESS:
            return ["renice", "-n", str(int(p["nice"])), "-p", str(int(p["pid"]))]
        if k is ActionKind.SET_CPU_AFFINITY:
            return ["taskset", "-pc", str(p["cpu_list"]), str(int(p["pid"]))]
        if k is ActionKind.CONFIGURE_MPS:
            return ["nvidia-smi", "-i", gpu, "-c", str(p.get("compute_mode", "EXCLUSIVE_PROCESS"))]
        if k is ActionKind.CONFIGURE_MIG:
            return ["nvidia-smi", "-i", gpu, "-mig", str(p.get("mig_enable", "1"))]
        if k in (ActionKind.KILL_ORPHAN_PROCESS, ActionKind.FREE_STALE_CACHE):
            return ["kill", "-9", str(int(p["pid"]))]
        if k is ActionKind.DRAIN_AND_RESTART_WORKLOAD:
            # Operator-supplied graceful drain+restart (e.g. scontrol requeue
            # after a checkpoint, or kubectl rollout restart of the controller).
            cmd = p.get("restart_command")
            if isinstance(cmd, list):
                return [str(c) for c in cmd]
            return ["scontrol", "requeue", str(p.get("job_id", req.job_id or ""))]
        raise ValueError(f"DataCenterActuator cannot build kind {k}")

    # -- Actuator contract ---------------------------------------------------

    def snapshot_state(self, req: ActionRequest) -> ActuationState:
        """Best-effort read of the value this action will change, for rollback.

        Without execution (advise/dry-run) we cannot read the device, so values
        are empty and rollback falls back to a safe reset. With a real runner we
        could query nvidia-smi here; we keep it conservative and store the
        requested-from baseline the caller passed in ``params['prior']`` if any.
        """
        prior = dict(req.params.get("prior", {}))
        return ActuationState(kind=req.kind, target=str(req.target), values=prior)

    def apply(self, req: ActionRequest) -> ActionResult:
        # Never-kill guard: any process-touching kind must clear the protected set.
        if req.kind in PROCESS_TOUCHING_KINDS:
            pid = req.params.get("pid")
            self.guard_protected(req, int(pid) if pid is not None else None)

        cmd = self.build_command(req)
        res = self._runner_for(req).run(cmd)
        rendered = CommandRunner.render(cmd)
        return ActionResult(
            ok=res.ok,
            kind=req.kind,
            command=rendered,
            message=res.output,
            executed=res.executed,
            no_op=False,
            error=None if res.ok else res.output,
        )

    def rollback(self, req: ActionRequest, prior: ActuationState) -> ActionResult:
        # Irreversible kinds: killing/freeing cannot be undone. We say so plainly
        # rather than pretend; the manager flags these as irreversible up front.
        if req.kind in (ActionKind.KILL_ORPHAN_PROCESS, ActionKind.FREE_STALE_CACHE):
            return ActionResult(
                ok=True,
                kind=req.kind,
                command="(no rollback: irreversible action)",
                message="irreversible; nothing to revert",
                executed=False,
            )
        if req.kind in (ActionKind.LOCK_CLOCKS,):
            cmd = ["nvidia-smi", "-i", str(req.target), "-rgc"]
        elif req.kind is ActionKind.SET_POWER_LIMIT and "power_limit_w" in prior.values:
            cmd = ["nvidia-smi", "-i", str(req.target), "-pl", str(int(prior.values["power_limit_w"]))]
        elif req.kind is ActionKind.RENICE_PROCESS and "nice" in prior.values:
            cmd = ["renice", "-n", str(int(prior.values["nice"])), "-p", str(int(req.params["pid"]))]
        else:
            # Generic safe revert: reset clocks / leave as-is.
            cmd = ["nvidia-smi", "-i", str(req.target), "-rgc"]
        res = self._runner_for(req).run(cmd)
        return ActionResult(
            ok=res.ok,
            kind=req.kind,
            command=CommandRunner.render(cmd),
            message=res.output,
            executed=res.executed,
            error=None if res.ok else res.output,
        )
