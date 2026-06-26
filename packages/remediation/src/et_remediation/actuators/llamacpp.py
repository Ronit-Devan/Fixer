"""llama.cpp backend: drain-then-restart with tuned flags.

llama-server exposes no live reconfiguration for the knobs that matter
(``-ngl``/``-t``/``-b``/KV-cache type/``--mlock``/``--no-mmap``/``--ctx-size``/
``--parallel``): changing them means a restart. A restart is *disruptive*, so it
only ever runs through the approval gate, and even then only AFTER a
request-drain so in-flight generations are not severed:

    drain (wait until requests_processing == 0, bounded)  ->  restart with flags

The drain reads the same llama-server metric the monitor already scrapes, via an
injected ``requests_inflight`` callable, so it is testable without a server.
"""

from __future__ import annotations

import logging
import time
from typing import Callable

from et_remediation.actions import ActionKind, ActionRequest, ActionResult
from et_remediation.actuators.base import Actuator, ActuationState, CommandRunner

log = logging.getLogger(__name__)

# The runtime-tunable llama-server flags a restart strategy may set (value flags).
_FLAG_MAP: dict[str, str] = {
    "n_gpu_layers": "-ngl",
    "threads": "-t",
    "batch_size": "-b",
    "ubatch_size": "--ubatch-size",
    "ctx_size": "--ctx-size",
    "parallel": "--parallel",
    "cache_type_k": "--cache-type-k",
    "cache_type_v": "--cache-type-v",
    "model_draft": "--model-draft",  # speculative decoding draft model (-md)
    # How many DRAFT-model layers go on the GPU. WITHOUT this the draft model
    # runs on the CPU, which serialises every draft step and negates the entire
    # speculative-decoding speedup on a GPU box. Short form -ngld is stable across
    # the llama.cpp builds that support speculative decoding.
    "n_gpu_layers_draft": "-ngld",
    "draft": "--draft",  # draft tokens per step (alias of --draft-max on most builds)
    # Acceptance-tuning knobs for speculative decoding. Only rendered when the
    # operator sets them (they default to unset), so behaviour is unchanged by
    # default and the operator opts in with names matching their llama.cpp build.
    "draft_min": "--draft-min",
    "draft_p_min": "--draft-p-min",
    "threads_batch": "--threads-batch",  # prefill thread count (host-bound case)
}

# Boolean (presence-only) flags: included when the param is truthy.
_BOOL_FLAG_MAP: dict[str, str] = {
    "mlock": "--mlock",
    "no_mmap": "--no-mmap",
    "cont_batching": "--cont-batching",
}

# Flags that take an explicit on/off-style argument in current llama.cpp. A bare
# presence flag is WRONG for these on recent builds (e.g. ``-fa, --flash-attn
# [on|off|auto]`` consumes an argument), so we always render the value.
_VALUE_BOOL_FLAG_MAP: dict[str, str] = {
    "flash_attn": "--flash-attn",
}


def _onoff(value: object) -> str:
    """Normalise a flag value to llama.cpp's on/off/auto vocabulary."""
    if value is True:
        return "on"
    if value is False:
        return "off"
    return str(value)


class LlamaCppActuator(Actuator):
    backend = "llamacpp"

    def __init__(
        self,
        runner: CommandRunner | None = None,
        *,
        requests_inflight: Callable[[], float | None] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.runner = runner or CommandRunner(execute=False)
        # Returns current llama-server requests_processing, or None if unknown.
        self._inflight = requests_inflight or (lambda: None)
        self._sleep = sleep

    def _runner_for(self, req: ActionRequest) -> CommandRunner:
        """dry_run forces build-only; otherwise defer to the configured runner."""
        return self.runner if not req.dry_run else CommandRunner(execute=False)

    def capabilities(self) -> set[ActionKind]:
        return {ActionKind.RESTART_LLAMA_SERVER}

    # -- command construction ------------------------------------------------

    def build_argv(self, req: ActionRequest) -> list[str]:
        """Render the tuned llama-server restart command."""
        p = req.params
        # An operator-provided restart wrapper (systemd unit, supervisor) takes
        # the tuned flags; default to a direct llama-server invocation.
        base = p.get("restart_command")
        model = p.get("model")
        argv: list[str] = list(base) if isinstance(base, list) else ["llama-server"]
        if model:
            argv += ["-m", str(model)]
        for key, flag in _FLAG_MAP.items():
            if key in p and p[key] is not None:
                argv += [flag, str(p[key])]
        for key, flag in _BOOL_FLAG_MAP.items():
            if p.get(key):
                argv += [flag]
        # Value-bearing on/off flags (current llama.cpp requires the argument).
        for key, flag in _VALUE_BOOL_FLAG_MAP.items():
            if key in p and p[key] is not None:
                argv += [flag, _onoff(p[key])]
        return argv

    # -- drain ---------------------------------------------------------------

    def drain(self, *, timeout_s: float = 30.0, poll_s: float = 0.5) -> bool:
        """Block until no requests are in flight (bounded). True if drained.

        Never forcibly cancels a request — it *waits* for generations to finish,
        which is what preserving the in-flight compute task requires. If the
        bound elapses while work remains, returns False and the caller must not
        restart (the approval flow surfaces this rather than severing work).
        """
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            inflight = self._inflight()
            if inflight is None or inflight < 1:
                return True
            self._sleep(poll_s)
        return (self._inflight() or 0) < 1

    # -- Actuator contract ---------------------------------------------------

    def snapshot_state(self, req: ActionRequest) -> ActuationState:
        # The prior argv (so a bad tune can be rolled back to the old flags).
        return ActuationState(
            kind=req.kind,
            target=str(req.target),
            values={"argv": list(req.params.get("prior_argv", []))},
        )

    def apply(self, req: ActionRequest) -> ActionResult:
        # Drain first so we never sever an in-flight generation.
        drain_timeout = float(req.params.get("drain_timeout_s", 30.0))
        if not req.dry_run:
            drained = self.drain(timeout_s=drain_timeout)
            if not drained:
                return ActionResult(
                    ok=False,
                    kind=req.kind,
                    command="(drain did not complete; restart aborted)",
                    message="requests still in flight after drain timeout",
                    executed=False,
                    error="drain_timeout",
                )
        argv = self.build_argv(req)
        res = self._runner_for(req).run(argv)
        return ActionResult(
            ok=res.ok,
            kind=req.kind,
            command=CommandRunner.render(argv),
            message=res.output,
            executed=res.executed,
            error=None if res.ok else res.output,
        )

    def rollback(self, req: ActionRequest, prior: ActuationState) -> ActionResult:
        argv = list(prior.values.get("argv", [])) or ["llama-server"]
        res = self._runner_for(req).run(argv)
        return ActionResult(
            ok=res.ok,
            kind=req.kind,
            command=CommandRunner.render(argv),
            message="restored prior llama-server flags",
            executed=res.executed,
            error=None if res.ok else res.output,
        )
