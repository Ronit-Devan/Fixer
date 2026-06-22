"""Shared data types for the inference monitor."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Verdict(str, Enum):
    HEALTHY = "healthy"
    IDLE_NO_REQUESTS = "idle_no_requests"
    MEMORY_HEADROOM = "memory_headroom"
    DECODE_BANDWIDTH_BOUND = "decode_bandwidth_bound"
    KV_CACHE_PRESSURE = "kv_cache_pressure"
    THERMAL_THROTTLE = "thermal_throttle"
    VRAM_PRESSURE = "vram_pressure"  # VRAM climbing toward OOM (predictive)
    UNKNOWN = "unknown"


# Human-facing titles, kept out of the logic so the dashboard and CLI agree.
VERDICT_TITLES: dict[Verdict, str] = {
    Verdict.HEALTHY: "Healthy, GPU well used",
    Verdict.IDLE_NO_REQUESTS: "Idle, no inference requests",
    Verdict.MEMORY_HEADROOM: "Memory under-used, room to do more",
    Verdict.DECODE_BANDWIDTH_BOUND: "Decode is memory-bandwidth bound",
    Verdict.KV_CACHE_PRESSURE: "KV cache under pressure",
    Verdict.THERMAL_THROTTLE: "GPU is throttling",
    Verdict.VRAM_PRESSURE: "VRAM filling toward out-of-memory",
    Verdict.UNKNOWN: "Not enough signal yet",
}


@dataclass(frozen=True)
class Snapshot:
    """One unified tick: GPU reading + (optional) llama metrics + derived rates."""

    timestamp_s: float
    # GPU
    gpu_name: str
    util_pct: float | None
    mem_used_mb: float | None
    mem_total_mb: float | None
    power_w: float | None
    power_limit_w: float | None
    sm_clock_mhz: int | None
    sm_clock_max_mhz: int | None
    temp_c: float | None
    # llama-server (None if not reachable)
    llama_reachable: bool = False
    requests_processing: float | None = None
    requests_deferred: float | None = None
    kv_cache_usage_ratio: float | None = None
    # Derived live rates (computed from counter deltas in state.py)
    gen_tokens_per_s: float | None = None
    prompt_tokens_per_s: float | None = None

    @property
    def mem_used_ratio(self) -> float | None:
        if self.mem_used_mb is None or not self.mem_total_mb:
            return None
        return self.mem_used_mb / self.mem_total_mb

    @property
    def clock_ratio(self) -> float | None:
        if self.sm_clock_mhz is None or not self.sm_clock_max_mhz:
            return None
        return self.sm_clock_mhz / self.sm_clock_max_mhz


@dataclass
class Diagnosis:
    verdict: Verdict
    title: str
    severity: str  # "ok" | "info" | "warn" | "crit"
    confidence: float  # 0..1
    summary: str
    evidence: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    # Predictive flag: this verdict was raised from a TREND projection before the
    # problem actually landed (e.g. temperature climbing toward the throttle
    # point), not from the current state already being bad. ``horizon_s`` is the
    # estimated time until the threshold is crossed, so remediation can act early.
    predicted: bool = False
    horizon_s: float | None = None

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict.value,
            "title": self.title,
            "severity": self.severity,
            "confidence": self.confidence,
            "summary": self.summary,
            "evidence": self.evidence,
            "recommendations": self.recommendations,
            "metrics": self.metrics,
            "predicted": self.predicted,
            "horizon_s": self.horizon_s,
        }
