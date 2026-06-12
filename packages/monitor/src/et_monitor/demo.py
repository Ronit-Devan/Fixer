"""Scripted inference timeline for demos and CI.

Drives the Mock GPU sampler and a fake llama-server through a loop of phases so
the dashboard exercises every verdict; idle, decode-bound, memory headroom, KV
pressure, throttling, healthy; without a real GPU or model. Use it for the
investor demo (``et-monitor --demo``) and as the deterministic fixture the
server tests run against.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from et_monitor.gpu import GpuReading, GpuSampler
from et_monitor.llama import LlamaMetrics


@dataclass(frozen=True)
class _Phase:
    name: str
    seconds: float
    util: float
    mem_ratio: float
    clock_ratio: float
    requests: float
    kv: float
    deferred: float
    gen_rate: float  # tokens/s generated during this phase
    prompt_rate: float


# Each phase is ~12s; the whole story is ~72s, then it repeats.
_PHASES: tuple[_Phase, ...] = (
    _Phase("idle", 12, 3, 0.45, 0.30, 0, 0.0, 0, 0, 0),
    _Phase("decode", 16, 46, 0.46, 0.95, 1, 0.30, 0, 55, 0),
    _Phase("headroom", 12, 88, 0.42, 0.98, 1, 0.35, 0, 80, 0),
    _Phase("kv_pressure", 14, 78, 0.92, 0.97, 4, 0.96, 1, 140, 0),
    _Phase("throttle", 10, 92, 0.7, 0.60, 2, 0.6, 0, 120, 0),
    _Phase("healthy", 12, 86, 0.71, 0.97, 3, 0.5, 0, 160, 0),
)
_TOTAL = sum(p.seconds for p in _PHASES)
_MEM_TOTAL_MB = 24564.0
_POWER_LIMIT_W = 70.0
_SM_MAX = 2520


class DemoTimeline:
    """Shared clock both demo adapters read from, plus token-counter integrator."""

    def __init__(self, clock=time.monotonic) -> None:
        self._clock = clock
        self._start = clock()
        self._gen_total = 0.0
        self._prompt_total = 0.0
        self._last_counter_t = self._start

    def phase(self, at: float | None = None) -> _Phase:
        now = (at if at is not None else self._clock())
        offset = (now - self._start) % _TOTAL
        acc = 0.0
        for p in _PHASES:
            acc += p.seconds
            if offset < acc:
                return p
        return _PHASES[-1]

    def advance_counters(self) -> tuple[float, float]:
        """Integrate token counters up to now. Call once per llama read."""
        now = self._clock()
        dt = max(0.0, now - self._last_counter_t)
        p = self.phase(now)
        self._gen_total += p.gen_rate * dt
        self._prompt_total += p.prompt_rate * dt
        self._last_counter_t = now
        return self._gen_total, self._prompt_total


class DemoGpuSampler(GpuSampler):
    backend = "demo"

    def __init__(self, timeline: DemoTimeline) -> None:
        self._tl = timeline

    def gpu_count(self) -> int:
        return 1

    def read(self) -> list[GpuReading]:
        p = self._tl.phase()
        return [
            GpuReading(
                timestamp_s=time.time(),
                index=0,
                name="Demo RTX PRO 4000 Blackwell SFF (24 GB)",
                util_pct=p.util,
                mem_used_mb=_MEM_TOTAL_MB * p.mem_ratio,
                mem_total_mb=_MEM_TOTAL_MB,
                power_w=_POWER_LIMIT_W * (0.12 + 0.8 * p.util / 100),
                power_limit_w=_POWER_LIMIT_W,
                sm_clock_mhz=int(_SM_MAX * p.clock_ratio),
                sm_clock_max_mhz=_SM_MAX,
                temp_c=38 + 0.45 * p.util,
            )
        ]


class DemoLlamaScraper:
    """Quacks like ``LlamaScraper`` but returns scripted metrics."""

    def __init__(self, timeline: DemoTimeline) -> None:
        self._tl = timeline
        self.base_url = "demo://llama-server"
        self.metrics_url = "demo://llama-server/metrics"

    def read(self) -> LlamaMetrics | None:
        p = self._tl.phase()
        gen_total, prompt_total = self._tl.advance_counters()
        return LlamaMetrics(
            timestamp_s=time.time(),
            reachable=True,
            raw={},
            prompt_tokens_total=prompt_total,
            predicted_tokens_total=gen_total,
            predicted_tokens_seconds=p.gen_rate,
            prompt_tokens_seconds=p.prompt_rate,
            kv_cache_usage_ratio=p.kv,
            kv_cache_tokens=p.kv * 4096,
            requests_processing=p.requests,
            requests_deferred=p.deferred,
            decode_total=gen_total,
        )
