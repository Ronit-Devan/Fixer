"""Canonical, backend-agnostic root-cause taxonomy.

The two ET products diagnose with *different* ``Verdict`` enums:

  * training engine (``gpu_doctor_engine``): DATALOADER_BOUND, NCCL_BOUND,
    CHECKPOINT_BOUND, SYNC_BOUND, PCIE_BOUND, KERNEL_LAUNCH_BOUND, ...
  * inference monitor (``et_monitor``): THERMAL_THROTTLE, IDLE_NO_REQUESTS,
    KV_CACHE_PRESSURE, DECODE_BANDWIDTH_BOUND, MEMORY_HEADROOM, ...

The remediation layer must not care which product spoke. We collapse both into
one ``RootCause`` enum and provide two thin mapping functions. Neither product's
``Verdict`` enum is touched — these maps key off the *string value* a verdict
carries (``Verdict`` subclasses ``str``), so the remediation package never
imports either product. This is the clean interface the design promised.
"""

from __future__ import annotations

from enum import Enum


class RootCause(str, Enum):
    """What is actually wrong, independent of how it was detected."""

    # --- has a non-disruptive remediation (auto-appliable) ---
    THERMAL_POWER_THROTTLE = "thermal_power_throttle"
    IDLE_ZOMBIE_PROCESS = "idle_zombie_process"
    CPU_BOUND_PREPROCESSING = "cpu_bound_preprocessing"
    MEMORY_FRAGMENTATION = "memory_fragmentation"

    # --- remediation needs a process/job restart (disruptive, approval-gated) ---
    DATA_PIPELINE_STARVATION = "data_pipeline_starvation"
    DISTRIBUTED_COMM_STALL = "distributed_comm_stall"
    SUBOPTIMAL_RUNTIME_FLAGS = "suboptimal_runtime_flags"
    # Layers running on CPU (-ngl too low): restart llama-server with the whole
    # model on the GPU. Disruptive (a restart), so always approval-gated.
    PARTIAL_GPU_OFFLOAD = "partial_gpu_offload"

    # --- conditionally remediable (spec decode when draft model is available) ---
    # The box is at the physical single-stream memory-bandwidth wall. A plain flag
    # restart will NOT raise tokens/sec — but speculative decoding can push past
    # it by verifying N draft tokens in one forward pass (amortising weight reads).
    # The SPEC_DECODE_AT_CEILING strategy handles this: it applies when a
    # ``draft_model`` knob is configured and falls back to advise-only when not.
    AT_PRACTICAL_CEILING = "at_practical_ceiling"
    NONE = "none"


# Inference-monitor verdict value -> RootCause.
_MONITOR_MAP: dict[str, RootCause] = {
    "thermal_throttle": RootCause.THERMAL_POWER_THROTTLE,
    "idle_no_requests": RootCause.IDLE_ZOMBIE_PROCESS,
    "decode_bandwidth_bound": RootCause.SUBOPTIMAL_RUNTIME_FLAGS,
    "memory_headroom": RootCause.SUBOPTIMAL_RUNTIME_FLAGS,
    "kv_cache_pressure": RootCause.SUBOPTIMAL_RUNTIME_FLAGS,
    "gpu_offload_partial": RootCause.PARTIAL_GPU_OFFLOAD,
    # VRAM climbing toward OOM (a predictive verdict): try to reclaim stale VRAM
    # non-disruptively before the card OOMs and kills the workload.
    "vram_pressure": RootCause.MEMORY_FRAGMENTATION,
    "healthy": RootCause.NONE,
    "unknown": RootCause.NONE,
}

# Training-engine verdict value -> RootCause.
_ENGINE_MAP: dict[str, RootCause] = {
    "dataloader_bound": RootCause.DATA_PIPELINE_STARVATION,
    "nccl_bound": RootCause.DISTRIBUTED_COMM_STALL,
    # The remaining engine verdicts are real, but their fixes are code/config
    # changes we only *advise* (no safe live actuation), so they map to NONE and
    # fall through to advise-only.
    "pcie_bound": RootCause.NONE,
    "checkpoint_bound": RootCause.NONE,
    "sync_bound": RootCause.NONE,
    "kernel_launch_bound": RootCause.NONE,
    "healthy": RootCause.NONE,
    "unknown": RootCause.NONE,
}


def map_monitor_verdict(verdict_value: str, metrics: dict | None = None) -> RootCause:
    """Map an ``et_monitor`` verdict value (+ optional metrics) to a RootCause.

    ``IDLE_NO_REQUESTS`` only becomes an actionable IDLE_ZOMBIE_PROCESS when the
    GPU is idle *but VRAM is still resident* — i.e. something is holding the card
    without serving. A genuinely empty idle GPU is not a remediation target (the
    fix is utilization, not a knob), so it maps to NONE.
    """
    rc = _MONITOR_MAP.get(verdict_value, RootCause.NONE)
    m = metrics or {}
    if rc is RootCause.IDLE_ZOMBIE_PROCESS:
        resident = m.get("mem_used_ratio")
        # Only treat as a zombie-hold when memory is meaningfully resident.
        if resident is None or resident < 0.10:
            return RootCause.NONE
    # Decode at the physical single-stream wall: a restart can't raise tokens/sec
    # (it's memory-bandwidth limited), so route to the advise-only no-op cause
    # instead of opening a pointless RESTART approval. The analyzer sets
    # ``at_practical_ceiling`` only when MBU is at the wall AND concurrency is 1;
    # under-batching / host-bound decode keep SUBOPTIMAL_RUNTIME_FLAGS (fixable).
    if verdict_value == "decode_bandwidth_bound" and m.get("at_practical_ceiling"):
        return RootCause.AT_PRACTICAL_CEILING
    return rc


def map_engine_verdict(verdict_value: str, metrics: dict | None = None) -> RootCause:
    """Map a ``gpu_doctor_engine`` verdict value (+ optional metrics) to a RootCause.

    A DATALOADER_BOUND trace whose metrics flag head-of-line blocking
    (``hol_blocking_likely``) is specifically a CPU-bound preprocessing problem
    (one slow ``__getitem__`` stalling a worker), which has a *non-disruptive*
    remediation (re-nice / widen affinity of the worker procs). Plain dataloader
    starvation needs a restart with more workers (disruptive).
    """
    rc = _ENGINE_MAP.get(verdict_value, RootCause.NONE)
    if rc is RootCause.DATA_PIPELINE_STARVATION:
        m = metrics or {}
        if m.get("hol_blocking_likely"):
            return RootCause.CPU_BOUND_PREPROCESSING
    return rc


def map_from_metrics(metrics: dict | None) -> RootCause:
    """Detect causes that have no dedicated verdict yet, from raw metrics.

    Memory fragmentation is not a first-class verdict in either product, but a
    monitor/agent can surface a ``fragmentation_ratio`` (reserved-but-unusable
    VRAM). When present and high, that is an actionable, non-disruptive cause.
    """
    m = metrics or {}
    frag = m.get("fragmentation_ratio")
    if frag is not None and frag >= 0.30:
        return RootCause.MEMORY_FRAGMENTATION
    return RootCause.NONE
