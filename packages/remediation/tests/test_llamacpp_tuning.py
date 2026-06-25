"""Accurate llama.cpp remediation: throughput-based verify, partial-offload fix,
at-ceiling no-op, and demand-gated / headroom-capped restart params."""

from __future__ import annotations

from types import SimpleNamespace

from conftest import diag

from et_remediation import (
    LlamaCppActuator,
    OutcomeKind,
    RemediationConfig,
    RemediationManager,
    RemediationMode,
    default_registry,
)
from et_remediation.actions import ActionContext, ActionKind, ActionRequest, ProtectedWorkload
from et_remediation.actuators.base import CommandRunner
from et_remediation.rootcause import RootCause, map_monitor_verdict
from et_remediation.strategies import _build_fix_offload, _build_restart_llama
from et_remediation.telemetry import summarize
from et_remediation.verify import throughput_recovered


def _ctx(metrics=None, knobs=None) -> ActionContext:
    return ActionContext(
        node_id="n", gpu_index=0, verdict_value="decode_bandwidth_bound",
        metrics=metrics or {}, pre=summarize([]), protected=ProtectedWorkload(),
        knobs=knobs or {},
    )


# -- throughput recovery predicate -------------------------------------------


def test_throughput_recovered_on_real_tps_gain():
    pre = summarize([SimpleNamespace(gen_tokens_per_s=20.0)])
    post = summarize([SimpleNamespace(gen_tokens_per_s=30.0)])
    assert throughput_recovered(pre, post) is True
    # A trivial gain is not recovery.
    assert throughput_recovered(pre, summarize([SimpleNamespace(gen_tokens_per_s=20.3)])) is False
    # Missing post throughput -> never confirm (would roll back).
    assert throughput_recovered(pre, summarize([SimpleNamespace(util_pct=99)])) is False


# -- root-cause mapping ------------------------------------------------------


def test_offload_verdict_maps_to_partial_offload():
    assert map_monitor_verdict("gpu_offload_partial") is RootCause.PARTIAL_GPU_OFFLOAD


def test_decode_at_ceiling_is_noop_cause():
    assert (
        map_monitor_verdict("decode_bandwidth_bound", {"at_practical_ceiling": True})
        is RootCause.AT_PRACTICAL_CEILING
    )
    # Under-batching / host-bound decode stays fixable (SUBOPTIMAL).
    assert (
        map_monitor_verdict("decode_bandwidth_bound", {"under_batching": True})
        is RootCause.SUBOPTIMAL_RUNTIME_FLAGS
    )
    # Back-compat: no metrics -> SUBOPTIMAL (unchanged).
    assert map_monitor_verdict("decode_bandwidth_bound") is RootCause.SUBOPTIMAL_RUNTIME_FLAGS


# -- demand-gated / headroom-capped restart builder --------------------------


def test_restart_holds_parallel_without_demand():
    # Single stream, no deferral -> do NOT add slots (would hurt latency).
    p = _build_restart_llama(_ctx(metrics={"mean_concurrency": 1.0, "max_requests_deferred": 0}))
    assert "parallel" not in p  # no current parallel knob and no demand -> unset
    assert "cont_batching" not in p


def test_restart_raises_parallel_with_demand():
    p = _build_restart_llama(_ctx(metrics={"max_requests_deferred": 2}, knobs={"target_parallel": 6}))
    assert p["parallel"] == 6
    assert p["cont_batching"] is True


def test_restart_does_not_grow_ngl_when_vram_full():
    p = _build_restart_llama(_ctx(metrics={"mem_used_ratio": 0.95}, knobs={"n_gpu_layers": 20}))
    assert p["n_gpu_layers"] == 20  # held at current; not pushed to 999 into a full card


def test_restart_skips_cache_quant_when_already_quantized():
    p = _build_restart_llama(_ctx(knobs={"current_cache_type_k": "q8_0"}))
    assert "cache_type_k" not in p


# -- offload fix builder safety ----------------------------------------------


def test_offload_fix_sets_full_layers_when_it_fits():
    p = _build_fix_offload(_ctx(metrics={"mem_used_ratio": 0.4}, knobs={"model_n_layers": 32}))
    assert p["n_gpu_layers"] == 32


def test_offload_fix_refuses_when_vram_full():
    import pytest

    with pytest.raises(ValueError):
        _build_fix_offload(_ctx(metrics={"mem_used_ratio": 0.95}))


def test_offload_fix_refuses_when_model_too_big():
    import pytest

    with pytest.raises(ValueError):
        _build_fix_offload(_ctx(knobs={"model_size_gb": 40, "vram_total_gb": 24}))


# -- end-to-end through the manager ------------------------------------------


def test_at_ceiling_advises_and_never_restarts():
    cfg = RemediationConfig(mode=RemediationMode.AUTO)
    mgr = RemediationManager(default_registry(), cfg, [LlamaCppActuator()], now_fn=lambda: 0.0)
    out = mgr.observe(
        diag("decode_bandwidth_bound", metrics={"at_practical_ceiling": True}), [], now=0.0
    )
    assert out.kind is OutcomeKind.ADVISED
    assert out.root_cause is RootCause.AT_PRACTICAL_CEILING


def test_partial_offload_opens_restart_approval_with_full_ngl():
    cfg = RemediationConfig(
        mode=RemediationMode.AUTO, knobs={"model": "m.gguf", "model_n_layers": 32}
    )
    mgr = RemediationManager(default_registry(), cfg, [LlamaCppActuator()], now_fn=lambda: 0.0)
    out = mgr.observe(
        diag("gpu_offload_partial", metrics={"mem_used_ratio": 0.4}), [], now=0.0
    )
    assert out.kind is OutcomeKind.APPROVAL_REQUIRED
    assert "-ngl 32" in out.approval.command_preview
    assert not mgr.actuators[0].runner.execute  # nothing executed at request time


# -- actuator renders the new flags ------------------------------------------


def test_actuator_renders_cont_batching_and_flash_attn():
    act = LlamaCppActuator(CommandRunner(execute=False))
    req = ActionRequest(
        kind=ActionKind.RESTART_LLAMA_SERVER, action_class=None, node_id="n", target="0",
        params={"model": "m.gguf", "parallel": 4, "cont_batching": True,
                "flash_attn": True, "ubatch_size": 512},
    )
    s = " ".join(act.build_argv(req))
    assert "--cont-batching" in s
    assert "--flash-attn" in s
    assert "--ubatch-size 512" in s
