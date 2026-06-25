"""Verdict logic; one test per condition the analyzer must distinguish."""

from __future__ import annotations

from et_monitor.analyzer import Thresholds, analyze
from et_monitor.types import Snapshot, Verdict

T = Thresholds()


def _snap(**kw) -> Snapshot:
    base = dict(
        timestamp_s=0.0,
        gpu_name="test",
        util_pct=50.0,
        mem_used_mb=12000.0,
        mem_total_mb=24000.0,
        power_w=40.0,
        power_limit_w=70.0,
        sm_clock_mhz=2400,
        sm_clock_max_mhz=2520,
        temp_c=45.0,
        llama_reachable=True,
        requests_processing=1.0,
        requests_deferred=0.0,
        kv_cache_usage_ratio=0.3,
        gen_tokens_per_s=50.0,
        prompt_tokens_per_s=0.0,
    )
    base.update(kw)
    # keep timestamps inside the default window
    return Snapshot(**base)


def _window(n=10, **kw):
    return [_snap(**kw) for _ in range(n)]


def test_too_few_samples_is_unknown():
    assert analyze([_snap()], T).verdict == Verdict.UNKNOWN


def test_idle_no_requests():
    w = _window(util_pct=3.0, requests_processing=0.0, gen_tokens_per_s=0.0)
    d = analyze(w, T)
    assert d.verdict == Verdict.IDLE_NO_REQUESTS
    assert d.severity == "info"


def test_decode_bandwidth_bound():
    # actively serving, low concurrency, util below saturation
    w = _window(util_pct=45.0, requests_processing=1.0, gen_tokens_per_s=55.0,
                mem_used_mb=11000.0, kv_cache_usage_ratio=0.3)
    assert analyze(w, T).verdict == Verdict.DECODE_BANDWIDTH_BOUND


def test_memory_headroom_when_saturated_but_low_vram():
    # high util (so not decode-bound) but lots of free VRAM
    w = _window(util_pct=90.0, requests_processing=1.0, mem_used_mb=9000.0,
                mem_total_mb=24000.0, kv_cache_usage_ratio=0.3)
    assert analyze(w, T).verdict == Verdict.MEMORY_HEADROOM


def test_kv_cache_pressure_on_high_ratio():
    w = _window(util_pct=70.0, kv_cache_usage_ratio=0.95, requests_processing=3.0,
                mem_used_mb=22000.0)
    assert analyze(w, T).verdict == Verdict.KV_CACHE_PRESSURE


def test_kv_cache_pressure_on_deferred():
    # Deferral WITH a contended cache (>= kv_defer_pressure_ratio) is real KV pressure.
    w = _window(util_pct=70.0, kv_cache_usage_ratio=0.85, requests_deferred=2.0,
                requests_processing=4.0, mem_used_mb=22000.0)
    assert analyze(w, T).verdict == Verdict.KV_CACHE_PRESSURE


def test_deferred_with_empty_cache_is_under_batching_not_kv_pressure():
    # Requests queued but the cache is near-empty -> too few slots (under-batching),
    # NOT cache pressure. Must NOT recommend "lower --parallel".
    w = _window(util_pct=55.0, kv_cache_usage_ratio=0.25, requests_deferred=2.0,
                requests_processing=4.0, gen_tokens_per_s=60.0, mem_used_mb=12000.0)
    d = analyze(w, T)
    assert d.verdict == Verdict.DECODE_BANDWIDTH_BOUND
    assert any("--parallel" in r and "cont-batching" in r.lower() for r in d.recommendations)


def test_thermal_throttle_beats_everything():
    # under load but clock dragged way down -> throttle wins even if kv is high
    w = _window(util_pct=92.0, sm_clock_mhz=1400, sm_clock_max_mhz=2520,
                kv_cache_usage_ratio=0.95, mem_used_mb=22000.0)
    assert analyze(w, T).verdict == Verdict.THERMAL_THROTTLE
    assert analyze(w, T).severity == "crit"


def test_healthy():
    w = _window(util_pct=85.0, mem_used_mb=18000.0, mem_total_mb=24000.0,
                requests_processing=3.0, kv_cache_usage_ratio=0.5,
                sm_clock_mhz=2500)
    assert analyze(w, T).verdict == Verdict.HEALTHY


def test_gpu_only_mode_idle_without_llama():
    # no llama metrics: idle inferred from low util
    w = _window(util_pct=2.0, llama_reachable=False, requests_processing=None,
                gen_tokens_per_s=None, kv_cache_usage_ratio=None,
                requests_deferred=None)
    assert analyze(w, T).verdict == Verdict.IDLE_NO_REQUESTS
