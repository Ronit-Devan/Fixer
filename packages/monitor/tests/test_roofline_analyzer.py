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
