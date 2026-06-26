"""Common actuator interface + the mode-gated command runner.

Every backend implements the same four-method contract so the manager is
backend-agnostic:

  capabilities()       -> which ActionKinds this backend can perform
  snapshot_state(req)  -> capture prior state (for rollback AND idempotency)
  apply(req)           -> perform the action (idempotent; honors dry_run)
  rollback(req, prior) -> revert to the captured prior state

The ``CommandRunner`` is the single chokepoint where "build a real command" and
"actually execute it" are separated. In every mode except AUTO the request is
``dry_run=True`` and the runner returns the command *without executing it* — the
real safety gate that makes advise/dry-run incapable of touching the box.
"""

from __future__ import annotations

import logging
import shlex
import shutil
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from et_remediation.actions import ActionKind, ActionRequest, ActionResult

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ActuationState:
    """Prior device/process state captured before an action.

    Used for two things: rolling back (restore these values) and idempotency
    (if the live state already equals the requested target, ``apply`` no-ops).
    """

    kind: ActionKind
    target: str
    values: dict = field(default_factory=dict)


@dataclass(frozen=True)
class RunResult:
    ok: bool
    executed: bool
    output: str


class CommandRunner:
    """Executes shell commands only when ``execute`` is True.

    When ``execute`` is False (advise / dry-run / no real binary), it returns a
    successful, *non-executed* result carrying the rendered command. This is the
    mechanism that lets the same actuator code path be safe in dry-run and live
    in AUTO — the difference is one boolean, set from the operating mode.
    """

    def __init__(self, *, execute: bool, timeout_s: float = 10.0) -> None:
        self.execute = execute
        self.timeout_s = timeout_s

    @staticmethod
    def render(cmd: list[str]) -> str:
        """Render argv as a copy-pasteable shell string.

        Execution always passes the argv *list* to subprocess (shell=False), so a
        path with spaces runs correctly regardless; this is purely the human-facing
        preview/audit string, which an operator may paste into a shell — so quote
        each token (e.g. a draft model path with spaces) to keep it faithful.
        """
        return " ".join(shlex.quote(part) for part in cmd)

    def run(self, cmd: list[str]) -> RunResult:
        rendered = self.render(cmd)
        if not self.execute:
            return RunResult(ok=True, executed=False, output="PLANNED (not executed)")
        binary = cmd[0]
        if shutil.which(binary) is None:
            return RunResult(
                ok=False, executed=False, output=f"{binary} not found on PATH"
            )
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
            )
        except Exception as e:  # noqa: BLE001 - any exec failure is a failed action
            log.warning("command failed: %s (%s)", rendered, e)
            return RunResult(ok=False, executed=True, output=str(e))
        ok = proc.returncode == 0
        return RunResult(ok=ok, executed=True, output=(proc.stdout + proc.stderr).strip())


class Actuator(ABC):
    """Backend contract. All four methods are pure w.r.t. the manager."""

    backend: str = "abstract"

    @abstractmethod
    def capabilities(self) -> set[ActionKind]:
        """Which kinds this backend knows how to perform."""

    @abstractmethod
    def snapshot_state(self, req: ActionRequest) -> ActuationState:
        """Capture prior state for rollback + idempotency. Never raises."""

    @abstractmethod
    def apply(self, req: ActionRequest) -> ActionResult:
        """Perform the action. Idempotent. Honors ``req.dry_run``."""

    @abstractmethod
    def rollback(self, req: ActionRequest, prior: ActuationState) -> ActionResult:
        """Restore the captured prior state."""

    # -- shared guard --------------------------------------------------------

    @staticmethod
    def guard_protected(req: ActionRequest, pid: int | None) -> None:
        """Raise if an action would target the protected live workload.

        This is the last line of the never-kill invariant: even a buggy spec or
        registry entry cannot make an actuator signal the running task, because
        every process-touching apply funnels through here first.
        """
        if req.protected.protects(pid):
            raise WorkloadProtectionError(
                f"refusing {req.kind.value} on protected workload pid={pid} "
                f"(label={req.protected.label!r})"
            )


class WorkloadProtectionError(RuntimeError):
    """Raised when an action would touch the protected live workload."""
