"""Roofline-aware diagnosis: partial offload, the single-stream wall, under-
batching, and host-bound decode — the verdicts that resolve the '40% of what?'
question for a llama.cpp box. All require a WorkloadSpec; without one the
analyzer is unchanged (covered by test_analyzer.py)."""

from __future__ import annotations

from et_monitor.analyzer import Thresholds, analyze
from et_monitor.perf import WorkloadSpec
from et_monitor.types import Snapshot, Verdict

T = Thresholds()

# 4.5 GB model on a 672 GB/s card: ideal ~149 tok/s, achievable ceiling ~127.
BW = 672.0
MODEL_BYTES = 4.5e9
IDEAL = BW * 1e9 / MODEL_BYTES  # ~149.3


def _full_spec(**kw) -> WorkloadSpec:
    base = dict(model_bytes=MODEL_BYTES, n_layers=32, n_gpu_layers=32, mem_bandwidth_gb_s=BW)
    base.update(kw)
    return WorkloadSpec(**base)


def _snap(**kw) -> Snapshot:
    base = dict(
        timestamp_s=0.0, gpu_name="test", util_pct=45.0,
        mem_used_mb=20000.0, mem_total_mb=24000.0, power_w=55.0, power_limit_w=70.0,
        sm_clock_mhz=2480, sm_clock_max_mhz=2520, temp_c=55.0,
        llama_reachable=True, requests_processing=1.0, requests_deferred=0.0,
        kv_cache_usage_ratio=0.3, gen_tokens_per_s=60.0, prompt_tokens_per_s=0.0,
    )
    base.update(kw)
    return Snapshot(**base)


def _window(n=10, **kw):
    return [_snap(**kw) for _ in range(n)]


def test_partial_offload_detected_and_actionable():
    spec = _full_spec(n_gpu_layers=16, model_bytes=8e9)  # 16/32 layers on GPU
    d = analyze(_window(util_pct=45.0, gen_tokens_per_s=20.0), T, spec)
    assert d.verdict == Verdict.GPU_OFFLOAD_PARTIAL
    assert d.metrics["partial_offload"] is True
    assert abs(d.metrics["offload_fraction"] - 0.5) < 1e-6
    assert d.confidence >= 0.8
    # 8 GB model fits in 24 GB VRAM -> the fix is "-ngl 999".
    assert any("-ngl 999" in r for r in d.recommendations)


def test_partial_offload_not_flagged_when_idle():
    spec = _full_spec(n_gpu_layers=16)
    d = analyze(_window(util_pct=2.0, requests_processing=0.0, gen_tokens_per_s=0.0), T, spec)
    assert d.verdict == Verdict.IDLE_NO_REQUESTS  # idle wins; no point "fixing" offload


def test_partial_offload_model_too_big_advises_smaller_quant():
    # 40 GB model can't fit a 24 GB card -> don't tell them to -ngl 999.
    spec = _full_spec(n_gpu_layers=20, model_bytes=40e9)
    d = analyze(_window(gen_tokens_per_s=8.0), T, spec)
    assert d.verdict == Verdict.GPU_OFFLOAD_PARTIAL
    assert any("smaller quant" in r.lower() for r in d.recommendations)
    assert not any("-ngl 999" in r for r in d.recommendations)


def test_single_stream_at_the_wall_is_physics_not_a_bug():
    # gen ~120 tok/s vs ideal ~149 -> MBU ~0.80 >= wall; concurrency 1, no deferral.
    d = analyze(
        _window(util_pct=45.0, gen_tokens_per_s=120.0, requests_processing=1.0, requests_deferred=0.0),
        T, _full_spec(),
    )
    assert d.verdict == Verdict.DECODE_BANDWIDTH_BOUND
    assert d.metrics["at_practical_ceiling"] is True
    assert d.metrics["single_stream"] is True
    assert "physics" in d.summary.lower()
    assert d.metrics["mbu"] >= 0.7


def test_under_batched_when_concurrency_without_deferral():
    # 2 in flight, none deferred (so KV-pressure doesn't intercept), util low.
    d = analyze(
        _window(util_pct=45.0, gen_tokens_per_s=60.0, requests_processing=2.0, requests_deferred=0.0),
        T, _full_spec(),
    )
    assert d.verdict == Verdict.DECODE_BANDWIDTH_BOUND
    assert d.metrics["under_batching"] is True
    assert any("--parallel" in r for r in d.recommendations)


def test_host_bound_when_far_below_the_wall_single_stream():
    # Full offload, single stream, gen only ~30 tok/s -> MBU ~0.20: bandwidth is
    # NOT the limit; flag a host/config bottleneck rather than blaming the GPU.
    d = analyze(
        _window(util_pct=45.0, gen_tokens_per_s=30.0, requests_processing=1.0, requests_deferred=0.0),
        T, _full_spec(),
    )
    assert d.verdict == Verdict.DECODE_BANDWIDTH_BOUND
    assert d.metrics["host_or_config_suspect"] is True
    assert d.metrics.get("at_practical_ceiling") is not True
    assert any("flash-attn" in r.lower() for r in d.recommendations)


def test_roofline_metrics_present_on_every_diagnosis_with_spec():
    d = analyze(_window(), T, _full_spec())
    for key in ("mbu", "throughput_pct", "ceiling_tok_s", "offload_fraction"):
        assert key in d.metrics


def test_no_spec_means_no_roofline_keys():
    d = analyze(_window(), T)  # no spec
    assert "mbu" not in d.metrics
    assert d.verdict in (Verdict.DECODE_BANDWIDTH_BOUND, Verdict.HEALTHY, Verdict.MEMORY_HEADROOM)


# --- GPU bandwidth table (Blackwell RTX PRO disambiguation) -----------------
# Regression for the client-SFF bug: the RTX PRO 4000 Blackwell SFF Edition
# (432 GB/s, 18 Gbps) must NOT inherit the full-size 4000's 672 GB/s, and every
# Blackwell-workstation variant must resolve to its verified spec bandwidth.

from et_monitor.perf import bandwidth_for  # noqa: E402


def test_bandwidth_sff_vs_full_size_disambiguation():
    # The whole point: identical prefix, different bandwidth, longest key wins.
    assert bandwidth_for("NVIDIA RTX PRO 4000 Blackwell SFF Edition") == 432.0
    assert bandwidth_for("NVIDIA RTX PRO 4000 Blackwell") == 672.0


def test_bandwidth_blackwell_workstation_table():
    cases = {
        "NVIDIA RTX PRO 6000 Blackwell Workstation Edition": 1792.0,
        "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition": 1792.0,
        "NVIDIA RTX PRO 6000 Blackwell Server Edition": 1597.0,
        "NVIDIA RTX PRO 5000 Blackwell": 1344.0,
        "NVIDIA RTX PRO 4500 Blackwell Workstation Edition": 896.0,
        "NVIDIA RTX PRO 2000 Blackwell": 288.0,
    }
    for name, want in cases.items():
        assert bandwidth_for(name) == want, name


def test_bandwidth_server_6000_not_shadowed_by_workstation():
    # "pro 6000 blackwell server" must out-length "pro 6000 blackwell".
    assert bandwidth_for("NVIDIA RTX PRO 6000 Blackwell Server Edition") == 1597.0
    assert bandwidth_for("NVIDIA RTX PRO 6000 Blackwell Server Edition") != 1792.0


def test_bandwidth_unknown_blackwell_falls_back_not_crashes():
    # A future/unknown Blackwell SKU hits the generic fallback, never None-crashes.
    assert bandwidth_for("NVIDIA RTX PRO 9999 Blackwell Hypothetical") == 672.0
    assert bandwidth_for(None) is None
    assert bandwidth_for("some GPU we have never heard of") is None


def test_bandwidth_non_blackwell_unaffected():
    assert bandwidth_for("NVIDIA GeForce RTX 4090") == 1008.0
    assert bandwidth_for("NVIDIA GeForce RTX 5090") == 1792.0
    assert bandwidth_for("NVIDIA RTX A4000") == 448.0  # not shadowed by "a40"
    assert bandwidth_for("NVIDIA H100 80GB HBM3") == 3350.0


# --- Prefill-bound verdict: the four client benchmark scenarios --------------
# Client box: RTX PRO 4000 Blackwell SFF (432 GB/s) serving a ~30B-A3B MoE.
# active_bytes ~3.8 GB/token -> ideal ~114 tok/s, so the measured 73-94 tok/s
# decode is HEALTHY (near the single-stream wall). The differentiator across
# scenarios is TTFT (cold long-context prefill), not decode.

def _client_spec() -> WorkloadSpec:
    return WorkloadSpec(
        model_bytes=18e9, active_bytes=3.8e9, n_layers=48, n_gpu_layers=999,
        mem_bandwidth_gb_s=432.0,
    )


def test_scenario_short_chat_is_healthy_not_prefill_bound():
    d = analyze(
        _window(util_pct=48.0, gen_tokens_per_s=94.0, prompt_tokens_per_s=2000.0,
                ttft_s=0.07, prompt_tokens=40.0),
        T, _client_spec(),
    )
    assert d.verdict != Verdict.PREFILL_BOUND
    assert d.severity in ("ok", "info")  # at the wall = healthy, nothing to fix
    assert d.metrics["gen_tokens_per_s"] == 94.0  # decode, never end-to-end


def test_scenario_8k_cold_is_prefill_bound():
    d = analyze(
        _window(util_pct=50.0, gen_tokens_per_s=85.0, prompt_tokens_per_s=2500.0,
                ttft_s=3.3, prompt_tokens=8000.0),
        T, _client_spec(),
    )
    assert d.verdict == Verdict.PREFILL_BOUND
    assert "cache-reuse" in " ".join(d.recommendations)
    assert d.metrics.get("prefill_bound") is True


def test_scenario_30k_cold_is_prefill_bound():
    d = analyze(
        _window(util_pct=52.0, gen_tokens_per_s=73.0, prompt_tokens_per_s=2140.0,
                ttft_s=14.0, prompt_tokens=30000.0),
        T, _client_spec(),
    )
    assert d.verdict == Verdict.PREFILL_BOUND
    assert d.metrics.get("est_prefill_s") is not None  # ~14s


def test_scenario_30k_warm_prefix_cache_is_healthy():
    # Warm prefix cache -> TTFT collapses to 0.31s, so NOT prefill-bound even
    # though the prompt is 30K. Decode ~80 tok/s is near the wall -> healthy.
    d = analyze(
        _window(util_pct=48.0, gen_tokens_per_s=80.0, prompt_tokens_per_s=90000.0,
                ttft_s=0.31, prompt_tokens=30000.0, cache_tokens=29900.0),
        T, _client_spec(),
    )
    assert d.verdict != Verdict.PREFILL_BOUND
    assert d.severity in ("ok", "info")


def test_prefill_bound_not_fired_when_decode_is_actually_broken():
    # High TTFT + long prompt but decode is genuinely poor (partial offload):
    # partial-offload must win; we must NOT mislabel a decode bug as prefill.
    spec = WorkloadSpec(
        model_bytes=18e9, active_bytes=3.8e9, n_layers=48, n_gpu_layers=8,
        mem_bandwidth_gb_s=432.0,
    )
    d = analyze(
        _window(util_pct=40.0, gen_tokens_per_s=12.0, prompt_tokens_per_s=300.0,
                ttft_s=10.0, prompt_tokens=30000.0),
        T, spec,
    )
    assert d.verdict == Verdict.GPU_OFFLOAD_PARTIAL


# --- Phase 2: prefix-cache observability drives the prefill verdict ----------
# Same client spec, but TTFT is DERIVED from /slots signals (prompt_tokens +
# cache_tokens + prefill rate), the way production sees it — no direct ttft_s.

def test_prefix_cache_cold_derives_high_ttft_and_fires_prefill_bound():
    d = analyze(
        _window(util_pct=52.0, gen_tokens_per_s=73.0, prompt_tokens_per_s=2140.0,
                prompt_tokens=30000.0, cache_tokens=0.0),  # cold: no reuse
        T, _client_spec(),
    )
    assert d.verdict == Verdict.PREFILL_BOUND
    assert d.metrics["prefix_cache_hit_rate"] == 0.0
    assert d.metrics["ttft_s"] >= 10.0  # ~30000/2140 ≈ 14s derived


def test_prefix_cache_warm_low_ttft_is_healthy():
    d = analyze(
        _window(util_pct=48.0, gen_tokens_per_s=80.0, prompt_tokens_per_s=90000.0,
                prompt_tokens=30000.0, cache_tokens=29900.0),  # warm: ~99.7% reuse
        T, _client_spec(),
    )
    assert d.verdict != Verdict.PREFILL_BOUND
    assert d.metrics["prefix_cache_hit_rate"] > 0.99
