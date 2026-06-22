"""Telemetry abstraction the verify-and-rollback loop reasons over.

The remediation layer must observe whether a fix actually produced *recovery*.
It does that over a rolling window of telemetry samples. We do not import
``et_monitor.Snapshot``; instead we read attributes structurally so the same
code works on a real monitor ``Snapshot``, a training-side sample, or a
``FakeSample`` from the simulation harness.

A sample may expose any subset of these (missing/None tolerated):
  util_pct, mem_used_ratio, clock_ratio, temp_c, power_w, power_limit_w,
  requests_processing, kv_cache_usage_ratio, mem_used_mb, mem_total_mb.

``mem_used_ratio`` / ``clock_ratio`` are read as attributes if present (the
monitor exposes them as properties) and otherwise derived.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Iterable, Sequence


def _get(sample: object, attr: str):
    return getattr(sample, attr, None)


def _mem_ratio(sample: object) -> float | None:
    r = _get(sample, "mem_used_ratio")
    if r is not None:
        return float(r)
    used = _get(sample, "mem_used_mb")
    total = _get(sample, "mem_total_mb")
    if used is not None and total:
        return float(used) / float(total)
    return None


def _clock_ratio(sample: object) -> float | None:
    r = _get(sample, "clock_ratio")
    if r is not None:
        return float(r)
    sm = _get(sample, "sm_clock_mhz")
    sm_max = _get(sample, "sm_clock_max_mhz")
    if sm is not None and sm_max:
        return float(sm) / float(sm_max)
    return None


def _vals(samples: Iterable[object], reader) -> list[float]:
    out: list[float] = []
    for s in samples:
        v = reader(s)
        if v is not None:
            out.append(float(v))
    return out


@dataclass(frozen=True)
class WindowSummary:
    """Aggregates over a telemetry window; the unit the recovery checks compare.

    Every field is Optional: a window with no clock readings has
    ``mean_clock_ratio is None``, and recovery predicates treat "unknown" as
    "not proven recovered" so we never confirm a fix on missing data.
    """

    n: int
    mean_util_pct: float | None
    max_util_pct: float | None
    mean_clock_ratio: float | None
    max_temp_c: float | None
    mean_mem_used_ratio: float | None
    min_mem_used_ratio: float | None
    mean_requests_processing: float | None
    max_kv_cache_ratio: float | None


def summarize(samples: Sequence[object]) -> WindowSummary:
    """Reduce a window of samples to a WindowSummary."""
    util = _vals(samples, lambda s: _get(s, "util_pct"))
    clk = _vals(samples, _clock_ratio)
    temp = _vals(samples, lambda s: _get(s, "temp_c"))
    memr = _vals(samples, _mem_ratio)
    reqs = _vals(samples, lambda s: _get(s, "requests_processing"))
    kv = _vals(samples, lambda s: _get(s, "kv_cache_usage_ratio"))
    return WindowSummary(
        n=len(samples),
        mean_util_pct=mean(util) if util else None,
        max_util_pct=max(util) if util else None,
        mean_clock_ratio=mean(clk) if clk else None,
        max_temp_c=max(temp) if temp else None,
        mean_mem_used_ratio=mean(memr) if memr else None,
        min_mem_used_ratio=min(memr) if memr else None,
        mean_requests_processing=mean(reqs) if reqs else None,
        max_kv_cache_ratio=max(kv) if kv else None,
    )
