"""Predictive (early-warning) detection: catch the problem before it lands."""

from __future__ import annotations

from et_monitor.analyzer import Thresholds, analyze
from et_monitor.types import Snapshot, Verdict

T = Thresholds()


def _snap(i: float, **kw) -> Snapshot:
    base = dict(
        timestamp_s=float(i),
        gpu_name="test",
        util_pct=50.0,
        mem_used_mb=10000.0,
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
    return Snapshot(**base)


def test_predicts_thermal_throttle_before_clock_drops():
    # Under load, clock still healthy, but temperature climbing toward throttle.
    temps = [70, 73, 76, 79, 82]
    w = [_snap(i, util_pct=90.0, temp_c=temp, sm_clock_mhz=2400) for i, temp in enumerate(temps)]
    d = analyze(w, T)
    assert d.verdict == Verdict.THERMAL_THROTTLE
    assert d.predicted is True
    assert d.severity == "warn"  # warn (imminent), not crit (already throttling)
    assert d.horizon_s is not None and d.horizon_s <= T.predict_horizon_s


def test_predicts_vram_oom_before_it_crashes():
    mems = [0.80, 0.83, 0.86, 0.89, 0.92]
    w = [_snap(i, util_pct=50.0, mem_used_mb=r * 24000.0) for i, r in enumerate(mems)]
    d = analyze(w, T)
    assert d.verdict == Verdict.VRAM_PRESSURE
    assert d.predicted is True
    assert "predicted_oom_s" in d.metrics


def test_predicts_kv_saturation_before_queueing():
    kvs = [0.70, 0.75, 0.80, 0.85]
    w = [_snap(i, util_pct=70.0, kv_cache_usage_ratio=k) for i, k in enumerate(kvs)]
    d = analyze(w, T)
    assert d.verdict == Verdict.KV_CACHE_PRESSURE
    assert d.predicted is True


def test_flat_window_does_not_predict():
    # Distinct timestamps but no trend -> no early warning, stays healthy.
    w = [
        _snap(i, util_pct=85.0, temp_c=50.0, mem_used_mb=18000.0,
              kv_cache_usage_ratio=0.4, requests_processing=3.0, sm_clock_mhz=2500)
        for i in range(8)
    ]
    d = analyze(w, T)
    assert d.predicted is False
    assert d.verdict == Verdict.HEALTHY


def test_actual_throttle_still_wins_over_prediction_and_is_not_predicted():
    # Clock already dragged down = reactive throttle (crit), not a prediction.
    w = [_snap(i, util_pct=92.0, sm_clock_mhz=1400, temp_c=85.0) for i in range(5)]
    d = analyze(w, T)
    assert d.verdict == Verdict.THERMAL_THROTTLE
    assert d.severity == "crit"
    assert d.predicted is False


def test_noisy_window_does_not_raise_false_warning():
    # Temperature jitters (no real trend) under load -> the r-squared gate
    # rejects the noisy fit, so no spurious "throttle imminent".
    temps = [70, 83, 61, 82, 62, 83]
    w = [_snap(i, util_pct=90.0, temp_c=t) for i, t in enumerate(temps)]
    d = analyze(w, T)
    assert d.predicted is False


def test_too_few_valid_samples_does_not_predict():
    # Only 3 of the samples carry a temperature (the rest are missing) -> the
    # series is below min_trend_samples, so no projection is trusted.
    temps = [70, None, 76, None, 82]
    w = [_snap(i, util_pct=90.0, temp_c=t) for i, t in enumerate(temps)]
    d = analyze(w, T)
    assert d.predicted is False


def test_vram_oom_outranks_thermal_when_both_imminent():
    # Both VRAM and temperature climbing; OOM (workload-killing) must win.
    w = [
        _snap(i, util_pct=90.0, temp_c=70 + 3 * i, mem_used_mb=(0.80 + 0.03 * i) * 24000.0)
        for i in range(5)
    ]
    d = analyze(w, T)
    assert d.verdict == Verdict.VRAM_PRESSURE
