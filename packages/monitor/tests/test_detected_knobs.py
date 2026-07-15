"""The detection->remediation-knobs bridge. Without it the VRAM-budget validator,
KV-quant skip, and 70W power-cap logic ship but never fire on the client box
(they only read knobs, which nothing populated). This is that dead-path fix."""

from __future__ import annotations

from types import SimpleNamespace

from et_monitor.__main__ import detected_knobs
from et_monitor.perf import WorkloadSpec


def test_detected_knobs_from_full_facts():
    spec = WorkloadSpec(
        model_bytes=18e9, kv_bytes_per_token=196608.0, mem_total_bytes=24e9,
        active_bytes=3.8e9, n_layers=48, n_gpu_layers=999, mem_bandwidth_gb_s=432.0,
    )
    gpu = SimpleNamespace(power_limit_w=70.0, power_limit_max_w=70.0)
    props = SimpleNamespace(cache_type_k="f16", ctx_size=32768)
    k = detected_knobs(spec, gpu, props)
    assert k["model_size_gb"] == 18.0
    assert k["vram_total_gb"] == 24.0
    assert abs(k["kv_gb_per_token"] - 196608.0 / 1e9) < 1e-12
    assert k["model_n_layers"] == 48
    assert k["current_power_limit_w"] == 70.0
    assert k["max_power_limit_w"] == 70.0  # 70W SFF -> power-cap logic can now refuse
    assert k["current_cache_type_k"] == "f16"
    assert k["ctx_size"] == 32768


def test_detected_knobs_degrades_when_facts_absent():
    # No spec/gpu/props -> empty (nothing fabricated).
    assert detected_knobs(None, None, None) == {}
    # Partial: spec without KV facts (dense/older) -> no kv_gb_per_token key.
    spec = WorkloadSpec(model_bytes=8e9, n_layers=32, mem_bandwidth_gb_s=672.0)
    k = detected_knobs(spec, None, None)
    assert k["model_size_gb"] == 8.0
    assert "kv_gb_per_token" not in k and "vram_total_gb" not in k


def test_bridged_knobs_carry_the_vram_facts_the_validator_needs():
    # The whole point: the bridge produces exactly the knobs _assert_vram_budget /
    # _build_power_limit read (validated in the remediation suite). Assert the
    # keys the client-safety features depend on are all present.
    spec = WorkloadSpec(
        model_bytes=21e9, kv_bytes_per_token=196608.0, mem_total_bytes=24e9,
        n_layers=48, mem_bandwidth_gb_s=432.0,
    )
    k = detected_knobs(spec, SimpleNamespace(power_limit_w=70.0, power_limit_max_w=70.0),
                       SimpleNamespace(cache_type_k="q8_0", ctx_size=30000))
    for needed in ("model_size_gb", "vram_total_gb", "kv_gb_per_token", "ctx_size",
                   "max_power_limit_w", "current_cache_type_k"):
        assert needed in k, needed
