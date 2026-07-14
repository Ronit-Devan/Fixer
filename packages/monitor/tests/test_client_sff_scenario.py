"""End-to-end regression for the client deployment: RTX PRO 4000 Blackwell SFF
(432 GB/s) serving a Qwen MoE (~3B active) on llama.cpp.

This ties Phases 1.1-1.3 together on the real numbers, and pins the adversarial
degradation contracts, so a future change that reintroduces any of the original
bugs (SFF=672, MoE divided by full weights, prefill read as decode) fails here.
"""

from __future__ import annotations

from et_monitor.analyzer import Thresholds, analyze
from et_monitor.perf import (
    GgufInfo,
    WorkloadSpec,
    bandwidth_for,
    estimate_moe_active_bytes,
    roofline,
)
from et_monitor.types import Snapshot, Verdict

T = Thresholds()

# The client's card and model, as ET would detect them.
SFF_NAME = "NVIDIA RTX PRO 4000 Blackwell SFF Edition"
_MOE = GgufInfo(
    path="qwen.gguf", file_bytes=17_000_000_000, architecture="qwen3moe",
    name="Qwen3 30B A3B", n_layers=48, param_count=30_500_000_000,
    expert_count=128, expert_used_count=8, embedding_length=2048, expert_ffn_len=768,
)


def _client_spec() -> WorkloadSpec:
    active_bytes, _ = estimate_moe_active_bytes(_MOE)
    return WorkloadSpec(
        model_bytes=float(_MOE.file_bytes), active_bytes=active_bytes,
        n_layers=_MOE.n_layers, n_gpu_layers=999,
        mem_bandwidth_gb_s=bandwidth_for(SFF_NAME),
        model_name=_MOE.name, gpu_name=SFF_NAME,
    )


def _win(prompt_tokens, decode_tokens, prefill_s, decode_s, *, util=88.0, n=8):
    out = []
    for i in range(n):
        f = i / (n - 1)
        out.append(Snapshot(
            timestamp_s=float(i), gpu_name=SFF_NAME, util_pct=util,
            mem_used_mb=17500.0, mem_total_mb=24000.0, power_w=68.0, power_limit_w=70.0,
            sm_clock_mhz=2500, sm_clock_max_mhz=2520, temp_c=70.0, llama_reachable=True,
            requests_processing=1.0, requests_deferred=0.0, kv_cache_usage_ratio=0.35,
            gen_tokens_per_s=decode_tokens / decode_s, prompt_tokens_per_s=prompt_tokens / prefill_s,
            prompt_tokens_total=prompt_tokens * f, predicted_tokens_total=decode_tokens * f,
            prompt_seconds_total=prefill_s * f, predicted_seconds_total=decode_s * f,
        ))
    return out


# -- the card + model are read correctly -------------------------------------


def test_sff_bandwidth_is_432_not_672():
    assert bandwidth_for(SFF_NAME) == 432.0


def test_moe_ceiling_is_realistic_not_dense():
    spec = _client_spec()
    assert spec.is_moe
    rl = roofline(spec, gen_tok_s=94.0)
    # ~3B active on 432 GB/s -> a couple-hundred tok/s ceiling, so 94 tok/s is a
    # healthy ~0.4 MBU. A dense-27B reading would put 94 impossibly over ceiling.
    assert 150 < rl.ceiling_tok_s < 260
    assert 0.3 < rl.mbu < 0.5
    assert not rl.partial_offload


# -- the four benchmark scenarios diagnose correctly -------------------------


def test_scenario_30k_cold_is_prefill_bound():
    d = analyze(_win(30000, 420, prefill_s=14.0, decode_s=5.75), T, _client_spec())
    assert d.verdict == Verdict.PREFILL_BOUND
    assert any("prefix cach" in r.lower() or "cache-reuse" in r.lower() for r in d.recommendations)


def test_scenario_30k_warm_is_not_prefill_bound():
    d = analyze(_win(310, 420, prefill_s=0.31, decode_s=5.3, util=85.0), T, _client_spec())
    assert d.verdict != Verdict.PREFILL_BOUND


def test_scenario_short_chat_is_healthy_decode():
    d = analyze(_win(50, 420, prefill_s=0.07, decode_s=4.5, util=85.0), T, _client_spec())
    assert d.verdict != Verdict.PREFILL_BOUND


# -- adversarial: everything degrades gracefully, nothing raises -------------


def test_unknown_gpu_yields_no_bandwidth_and_no_roofline():
    assert bandwidth_for("Some Mystery Accelerator X1") is None
    spec = WorkloadSpec(model_bytes=17e9, n_layers=48, n_gpu_layers=48)  # no bandwidth
    rl = roofline(spec, gen_tok_s=94.0)
    assert rl is not None and rl.mbu is None and rl.at_bandwidth_wall is False


def test_moe_missing_used_count_falls_back_to_dense():
    info = GgufInfo(path="x", file_bytes=17e9, architecture="qwen3moe",
                    n_layers=48, expert_count=128)  # no expert_used_count
    ab, note = estimate_moe_active_bytes(info)
    assert ab is None and "dense" in note.lower()


def test_partial_offload_moe_flags_and_never_crashes():
    spec = _client_spec()
    partial = WorkloadSpec(
        model_bytes=spec.model_bytes, active_bytes=spec.active_bytes,
        n_layers=48, n_gpu_layers=24, mem_bandwidth_gb_s=432.0,
    )
    rl = roofline(partial, gen_tok_s=15.0)
    assert rl.partial_offload is True
    assert rl.offload_fraction == 0.5


def test_dense_model_unchanged_by_moe_path():
    # A dense spec (no active_bytes) still divides by full weights.
    dense = WorkloadSpec(model_bytes=16e9, n_layers=32, n_gpu_layers=32, mem_bandwidth_gb_s=432.0)
    assert dense.per_token_bytes() == 16e9
    assert abs(roofline(dense, None).ideal_tok_s - 432e9 / 16e9) < 1e-6
