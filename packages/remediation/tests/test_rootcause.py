"""Canonical root-cause mapping: both products collapse into one taxonomy."""

from __future__ import annotations

from et_remediation.rootcause import (
    RootCause,
    map_engine_verdict,
    map_from_metrics,
    map_monitor_verdict,
)


def test_monitor_thermal_maps_to_throttle():
    assert map_monitor_verdict("thermal_throttle") is RootCause.THERMAL_POWER_THROTTLE


def test_monitor_llama_verdicts_map_to_runtime_flags():
    for v in ("decode_bandwidth_bound", "memory_headroom", "kv_cache_pressure"):
        assert map_monitor_verdict(v) is RootCause.SUBOPTIMAL_RUNTIME_FLAGS


def test_idle_only_actionable_when_vram_resident():
    # Empty idle GPU -> nothing to remediate.
    assert map_monitor_verdict("idle_no_requests", {"mem_used_ratio": 0.0}) is RootCause.NONE
    assert map_monitor_verdict("idle_no_requests", {}) is RootCause.NONE
    # Idle but VRAM resident -> a zombie is holding the card.
    assert (
        map_monitor_verdict("idle_no_requests", {"mem_used_ratio": 0.6})
        is RootCause.IDLE_ZOMBIE_PROCESS
    )


def test_healthy_and_unknown_map_to_none():
    assert map_monitor_verdict("healthy") is RootCause.NONE
    assert map_monitor_verdict("unknown") is RootCause.NONE
    assert map_engine_verdict("healthy") is RootCause.NONE


def test_engine_dataloader_vs_cpu_bound():
    assert map_engine_verdict("dataloader_bound") is RootCause.DATA_PIPELINE_STARVATION
    # HoL blocking flag => CPU-bound preprocessing (non-disruptive remedy).
    assert (
        map_engine_verdict("dataloader_bound", {"hol_blocking_likely": True})
        is RootCause.CPU_BOUND_PREPROCESSING
    )


def test_engine_nccl_maps_to_comm_stall():
    assert map_engine_verdict("nccl_bound") is RootCause.DISTRIBUTED_COMM_STALL


def test_advise_only_engine_verdicts_map_none():
    for v in ("pcie_bound", "checkpoint_bound", "sync_bound", "kernel_launch_bound"):
        assert map_engine_verdict(v) is RootCause.NONE


def test_memory_fragmentation_from_metrics():
    assert map_from_metrics({"fragmentation_ratio": 0.5}) is RootCause.MEMORY_FRAGMENTATION
    assert map_from_metrics({"fragmentation_ratio": 0.1}) is RootCause.NONE
    assert map_from_metrics({}) is RootCause.NONE
