"""llama.cpp actuator: tuned restart argv + bounded request-drain."""

from __future__ import annotations

from et_remediation.actions import (
    ActionClass,
    ActionKind,
    ActionRequest,
    ProtectedWorkload,
)
from et_remediation.actuators.base import CommandRunner
from et_remediation.actuators.llamacpp import LlamaCppActuator


def _req(params, dry_run=True):
    return ActionRequest(
        kind=ActionKind.RESTART_LLAMA_SERVER,
        action_class=ActionClass.DISRUPTIVE,
        node_id="n0",
        target="http://localhost:8080",
        params=params,
        protected=ProtectedWorkload(),
        dry_run=dry_run,
    )


def test_build_argv_includes_tuned_flags():
    act = LlamaCppActuator()
    argv = act.build_argv(
        _req({"model": "m.gguf", "n_gpu_layers": 999, "parallel": 4,
              "cache_type_k": "q8_0", "cache_type_v": "q8_0", "mlock": True})
    )
    s = " ".join(argv)
    assert "-m m.gguf" in s
    assert "-ngl 999" in s
    assert "--parallel 4" in s
    assert "--cache-type-k q8_0" in s and "--cache-type-v q8_0" in s
    assert "--mlock" in s


def test_drain_returns_true_when_idle():
    act = LlamaCppActuator(requests_inflight=lambda: 0)
    assert act.drain(timeout_s=1.0) is True


def test_drain_waits_then_succeeds_when_requests_finish():
    seq = iter([2, 1, 0])  # requests drain to zero over polls
    act = LlamaCppActuator(
        requests_inflight=lambda: next(seq, 0),
        sleep=lambda s: None,  # no real waiting
    )
    assert act.drain(timeout_s=5.0, poll_s=0.01) is True


def test_apply_aborts_restart_if_drain_times_out():
    # Requests never finish, executing runner -> apply must NOT restart.
    act = LlamaCppActuator(
        CommandRunner(execute=True),
        requests_inflight=lambda: 3,
        sleep=lambda s: None,
    )
    res = act.apply(_req({"model": "m.gguf", "drain_timeout_s": 0.05}, dry_run=False))
    assert not res.ok and res.error == "drain_timeout"
    assert not res.executed


def test_dry_run_never_executes_even_with_executing_runner():
    act = LlamaCppActuator(CommandRunner(execute=True), requests_inflight=lambda: 0)
    res = act.apply(_req({"model": "m.gguf"}, dry_run=True))
    assert res.ok and not res.executed
