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
from et_remediation.strategies import _build_fix_offload, _build_restart_llama, _build_spec_decode
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


# -- flash-attn and ubatch auto-applied by restart builder -------------------


def test_restart_builder_sets_flash_attn_on_by_default():
    # Rendered with an explicit on/off value (current llama.cpp -fa takes an arg).
    p = _build_restart_llama(_ctx())
    assert p.get("flash_attn") == "on"


def test_restart_builder_skips_flash_attn_when_disabled():
    p = _build_restart_llama(_ctx(knobs={"flash_attn": False}))
    assert "flash_attn" not in p


def test_restart_builder_raises_ubatch_only_with_vram_headroom():
    # Clear headroom -> bump ubatch (helps prefill); no headroom -> leave default.
    assert _build_restart_llama(_ctx(metrics={"mem_used_ratio": 0.5})).get("ubatch_size") == 1024
    assert "ubatch_size" not in _build_restart_llama(_ctx(metrics={"mem_used_ratio": 0.9}))
    # Unknown VRAM -> don't risk it.
    assert "ubatch_size" not in _build_restart_llama(_ctx())


def test_restart_builder_respects_ubatch_knob_regardless_of_vram():
    p = _build_restart_llama(_ctx(metrics={"mem_used_ratio": 0.95}, knobs={"ubatch_size": 2048}))
    assert p["ubatch_size"] == 2048


# -- speculative decoding strategy -------------------------------------------


def test_spec_decode_builds_with_draft_model_and_offloads_draft():
    p = _build_spec_decode(_ctx(knobs={"model": "main.gguf", "draft_model": "draft.gguf"}))
    assert p["model_draft"] == "draft.gguf"
    assert p["draft"] == 16  # default
    assert p["flash_attn"] == "on"
    # The draft model MUST be offloaded to the GPU or it serialises on CPU.
    assert p["n_gpu_layers_draft"] == 999


def test_spec_decode_respects_draft_n_and_draft_ngl_knobs():
    p = _build_spec_decode(_ctx(knobs={"draft_model": "d.gguf", "draft_n": 32, "draft_n_gpu_layers": 24}))
    assert p["draft"] == 32
    assert p["n_gpu_layers_draft"] == 24


def test_spec_decode_raises_without_draft_model():
    import pytest
    with pytest.raises(KeyError, match="draft_model"):
        _build_spec_decode(_ctx())


def test_spec_decode_refuses_when_vram_near_full():
    # Adding a second (draft) model to a near-full card would OOM -> advise instead.
    import pytest
    with pytest.raises(ValueError, match="OOM"):
        _build_spec_decode(_ctx(metrics={"mem_used_ratio": 0.9}, knobs={"draft_model": "d.gguf"}))


def test_spec_decode_refuses_nonpositive_draft_count():
    import pytest
    with pytest.raises(ValueError, match="draft token count"):
        _build_spec_decode(_ctx(knobs={"draft_model": "d.gguf", "draft_n": 0}))
    with pytest.raises(ValueError, match="draft token count"):
        _build_spec_decode(_ctx(knobs={"draft_model": "d.gguf", "draft": -3}))


def test_spec_decode_passes_through_acceptance_knobs_only_when_set():
    # Default: no draft-min / draft-p-min keys at all (unchanged behaviour).
    p = _build_spec_decode(_ctx(knobs={"draft_model": "d.gguf"}))
    assert "draft_min" not in p and "draft_p_min" not in p
    # Operator-set: passed through for the actuator to render.
    p = _build_spec_decode(_ctx(knobs={"draft_model": "d.gguf", "draft_min": 1, "draft_p_min": 0.4}))
    assert p["draft_min"] == 1
    assert p["draft_p_min"] == 0.4


def test_actuator_renders_acceptance_knobs():
    act = LlamaCppActuator(CommandRunner(execute=False))
    req = ActionRequest(
        kind=ActionKind.RESTART_LLAMA_SERVER, action_class=None, node_id="n", target="0",
        params={"model": "m.gguf", "model_draft": "d.gguf", "draft": 8,
                "draft_min": 1, "draft_p_min": 0.4},
    )
    s = " ".join(act.build_argv(req))
    assert "--draft-min 1" in s
    assert "--draft-p-min 0.4" in s


def test_command_preview_is_shell_safe_for_paths_with_spaces():
    # The operator may paste command_preview into a shell; a draft path with
    # spaces must stay one token (quoted), not split into two.
    from et_remediation.actuators.base import CommandRunner as CR
    act = LlamaCppActuator(CommandRunner(execute=False))
    req = ActionRequest(
        kind=ActionKind.RESTART_LLAMA_SERVER, action_class=None, node_id="n", target="0",
        params={"model": "/models/my main.gguf", "model_draft": "/models/my draft.gguf"},
    )
    preview = CR.render(act.build_argv(req))
    assert "'/models/my main.gguf'" in preview
    assert "'/models/my draft.gguf'" in preview


def test_at_ceiling_with_full_vram_falls_through_to_advise():
    # Even with a draft model configured, a near-full card can't fit it -> advise.
    cfg = RemediationConfig(
        mode=RemediationMode.AUTO,
        knobs={"model": "main.gguf", "draft_model": "draft.gguf"},
    )
    mgr = RemediationManager(default_registry(), cfg, [LlamaCppActuator()], now_fn=lambda: 0.0)
    out = mgr.observe(
        diag("decode_bandwidth_bound",
             metrics={"at_practical_ceiling": True, "mem_used_ratio": 0.95}),
        [], now=0.0,
    )
    assert out.kind is OutcomeKind.ADVISED
    assert out.root_cause is RootCause.AT_PRACTICAL_CEILING


def test_at_ceiling_without_draft_model_still_advises():
    cfg = RemediationConfig(mode=RemediationMode.AUTO)
    mgr = RemediationManager(default_registry(), cfg, [LlamaCppActuator()], now_fn=lambda: 0.0)
    out = mgr.observe(
        diag("decode_bandwidth_bound", metrics={"at_practical_ceiling": True}), [], now=0.0
    )
    assert out.kind is OutcomeKind.ADVISED
    assert out.root_cause is RootCause.AT_PRACTICAL_CEILING


def test_at_ceiling_with_draft_model_opens_spec_decode_approval():
    cfg = RemediationConfig(
        mode=RemediationMode.AUTO,
        knobs={"model": "main.gguf", "draft_model": "draft.gguf"},
    )
    mgr = RemediationManager(default_registry(), cfg, [LlamaCppActuator()], now_fn=lambda: 0.0)
    out = mgr.observe(
        diag("decode_bandwidth_bound", metrics={"at_practical_ceiling": True}), [], now=0.0
    )
    assert out.kind is OutcomeKind.APPROVAL_REQUIRED
    assert out.root_cause is RootCause.AT_PRACTICAL_CEILING
    prev = out.approval.command_preview
    assert "--model-draft draft.gguf" in prev
    assert "--draft 16" in prev
    # Draft model offloaded to GPU (not stranded on CPU) and flash-attn valued.
    assert "-ngld 999" in prev
    assert "--flash-attn on" in prev


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
    # Current llama.cpp -fa takes an argument; a bare --flash-attn is wrong.
    assert "--flash-attn on" in s
    assert "--ubatch-size 512" in s


def test_actuator_renders_flash_attn_off_and_auto():
    act = LlamaCppActuator(CommandRunner(execute=False))

    def render(val):
        req = ActionRequest(
            kind=ActionKind.RESTART_LLAMA_SERVER, action_class=None, node_id="n", target="0",
            params={"model": "m.gguf", "flash_attn": val},
        )
        return " ".join(act.build_argv(req))

    assert "--flash-attn off" in render(False)
    assert "--flash-attn auto" in render("auto")
    assert "--flash-attn on" in render("on")


def test_actuator_renders_spec_decode_with_draft_gpu_layers():
    act = LlamaCppActuator(CommandRunner(execute=False))
    req = ActionRequest(
        kind=ActionKind.RESTART_LLAMA_SERVER, action_class=None, node_id="n", target="0",
        params={"model": "m.gguf", "model_draft": "draft.gguf", "draft": 16,
                "n_gpu_layers_draft": 999, "flash_attn": "on"},
    )
    s = " ".join(act.build_argv(req))
    assert "--model-draft draft.gguf" in s
    assert "--draft 16" in s
    assert "-ngld 999" in s  # draft model offloaded to GPU, not stranded on CPU
    assert "--flash-attn on" in s
