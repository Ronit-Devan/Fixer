"""Bridge from a confirmed ``IdleEvent`` to an engine ``Diagnosis``.

The flow:
  1. The agent's idle detector emits ``IdleEvent(gpu_index, started_at_s, ...)``.
  2. ``attribute()`` asks an ``EventSource`` for the recent CUDA-API / kernel
     events in [started_at_s − lookback_s, now_s].
  3. Those events are wrapped in a ``Trace`` and handed to ``diagnose()``.
  4. The verdict is folded back into a user-facing alert line.

The whole module is best-effort: empty windows, undersized event counts, or
exceptions from the engine all degrade to ``None`` rather than raising.
Attribution is supplementary signal; it must never take the daemon down.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from gpu_doctor_engine import Diagnosis, Event, Trace, diagnose

if TYPE_CHECKING:
    from gpu_doctor_agent.detector import IdleEvent
    from gpu_doctor_agent.events import EventSource

log = logging.getLogger(__name__)

# Sparse-trace gate thresholds. These MIRROR the engine's warmup_trace_guard
# (gpu_doctor_engine.diagnose._is_warmup_trace) so the agent rejects EXACTLY
# the windows the engine would treat as warm-up — no more, no less.
#
# The previous gate was a flat ``len(events) < MIN_EVENTS`` floor. That was
# stricter than the engine and silently dropped sparse-but-diagnosable
# windows: a checkpoint stall captured as a single 150ms kernel + torch.save
# is 3 events but is NOT warm-up to the engine (its kernel time alone clears
# the 5ms exemption), yet the flat floor refused it and the agent disagreed
# with the engine on the same trace. The engine's guard is an AND of four
# conditions, so a SINGLE substantial dimension (>=5 kernels, OR >=5ms kernel
# time, OR >=50ms span, OR >=85% util) means there is signal to diagnose.
#
# MIN_EVENTS keeps its name and value (the kernel-count exemption) for the
# public re-export and the existing gate tests. The agent/engine consistency
# suite (test_file_attribution.py) is the drift guard if the engine's
# thresholds ever change.
MIN_EVENTS: int = 5  # >= this many GPU kernel events => never warm-up
_WARMUP_MAX_KERNEL_TIME_US: int = 5_000
_WARMUP_MAX_DURATION_US: int = 50_000
_WARMUP_MAX_UTIL: float = 0.85

# Mirror of the engine's category sets (see gpu_doctor_engine/ingest.py).
# Duplicated here intentionally to avoid importing from the engine's private
# submodule path; if the engine ever adds a new category the worst that
# happens is we under-count busy time in the constructed Trace, which only
# affects the `gpu_utilization` field (verdict logic computes idle from the
# events themselves, not from the Trace's pre-aggregated counters).
_GPU_KERNEL_CATS: frozenset[str] = frozenset({"kernel", "gpu_op"})
_GPU_MEMCPY_CATS: frozenset[str] = frozenset({"gpu_memcpy", "gpu_memset"})
_GPU_ALL_CATS: frozenset[str] = _GPU_KERNEL_CATS | _GPU_MEMCPY_CATS
_CPU_CATS: frozenset[str] = frozenset(
    {"cpu_op", "python_function", "user_annotation"}
)

# GPU-only-profile detection threshold — mirror of load_trace's 0.05. Below
# this CPU-event fraction the all-events span is dominated by CUDA-init
# overhead and the GPU-active span is the honest denominator.
_GPU_ONLY_CPU_RATIO: float = 0.05


def _merge_intervals(
    intervals: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Merge overlapping (start, end) intervals. Returns a sorted disjoint list."""
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged: list[tuple[int, int]] = [intervals[0]]
    for start, end in intervals[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def _busy_time_us(events: list[Event], categories: frozenset[str]) -> int:
    """Total wall-clock microseconds spent inside events matching the categories."""
    spans = [(e.ts, e.ts + e.dur) for e in events if e.category in categories]
    return sum(end - start for start, end in _merge_intervals(spans))


def _build_trace(
    events: list[Event], window_start_s: float, window_end_s: float
) -> Trace:
    """Wrap captured events in a Trace the engine will accept.

    ``duration_us`` is derived from the captured events themselves —
    ``max(ts + dur) - min(ts)`` across the list — NOT from the live
    wall-clock window span. Two reasons:

      * For a ``FileEventSource``, the events come from a recorded trace
        whose ``ts`` values are on the profiler's own origin (near 0),
        completely unrelated to the agent's monotonic clock. Using
        ``(now_s - window_start_s) * 1e6`` here would inflate duration by
        many orders of magnitude, collapsing ``gpu_utilization`` to ~0 and
        starving every util-gated rule in the engine.

      * For mock / live sources whose events ARE anchored inside the
        window, the event span is still the correct denominator for
        utilization — it measures the interval the engine actually has
        evidence about, not whatever empty pre-roll the agent slept
        through.

    Verdict logic in ``diagnose()`` computes its own idle intervals from
    the events, so the only field this denominator affects is
    ``gpu_utilization``. ``window_start_s`` / ``window_end_s`` are kept on
    the signature for symmetry with ``attribute()`` (and potential future
    debug logging) but no longer participate in the calculation.

    GPU-only profiles
    -----------------
    The duration computation mirrors ``gpu_doctor_engine.ingest.load_trace``:
    when CPU events are sparse (< ``_GPU_ONLY_CPU_RATIO`` of the window), the
    all-events span is dominated by CUDA-init overhead (``cudaMalloc`` /
    ``cudaLaunchKernel`` that run long before the first kernel), which
    collapses ``gpu_utilization`` toward 0. In that case the GPU-active span is
    the honest denominator. Without this, a GPU-only window the engine
    correctly calls HEALTHY (high util on the GPU span) the agent miscalls
    UNKNOWN (near-zero util on the inflated wall-clock span) — the exact
    "wall-clock bias" the engine already fixed in ``load_trace``. Matching it
    here keeps the agent and engine in agreement on the same events.
    """
    del window_start_s, window_end_s  # see docstring: event-derived now.
    if events:
        span_start = min(e.ts for e in events)
        span_end = max(e.ts + e.dur for e in events)
        cpu_ratio = sum(1 for e in events if e.category in _CPU_CATS) / len(events)
        if cpu_ratio < _GPU_ONLY_CPU_RATIO:
            gpu_events = [e for e in events if e.category in _GPU_ALL_CATS]
            if gpu_events:
                duration_us = max(e.ts + e.dur for e in gpu_events) - min(
                    e.ts for e in gpu_events
                )
            else:
                duration_us = span_end - span_start
        else:
            duration_us = span_end - span_start
        if duration_us < 0:
            duration_us = 0
    else:
        duration_us = 0
    gpu_kernel_time_us = _busy_time_us(events, _GPU_KERNEL_CATS)
    cpu_time_us = _busy_time_us(events, _CPU_CATS)
    return Trace(
        events=sorted(events, key=lambda e: e.ts),
        duration_us=duration_us,
        gpu_kernel_time_us=gpu_kernel_time_us,
        cpu_time_us=cpu_time_us,
    )


def _trace_is_warmup(trace: Trace) -> bool:
    """Mirror of the engine's ``_is_warmup_trace``: too little signal to diagnose.

    Returns True only when ALL of the engine's warm-up conditions hold — fewer
    than ``MIN_EVENTS`` GPU kernel events AND under 5ms of kernel time AND under
    50ms of span AND under 85% utilization. Any single substantial dimension
    means there is diagnostic signal, so the window must reach ``diagnose()``.

    Keeping this identical to the engine's guard is what makes the agent and
    engine agree on whether a window is diagnosable at all.
    """
    kernel_events = [e for e in trace.events if e.category in _GPU_KERNEL_CATS]
    if len(kernel_events) >= MIN_EVENTS:
        return False
    if trace.gpu_kernel_time_us >= _WARMUP_MAX_KERNEL_TIME_US:
        return False
    if trace.duration_us >= _WARMUP_MAX_DURATION_US:
        return False
    if trace.gpu_utilization >= _WARMUP_MAX_UTIL:
        return False
    return True


def attribute(
    event_source: "EventSource",
    idle_event: "IdleEvent",
    now_s: float,
    lookback_s: float = 5.0,
) -> Diagnosis | None:
    """Run the engine over a window ending at ``now_s``.

    Returns the engine's ``Diagnosis``, or ``None`` if attribution failed
    (empty / undersized event window, or an engine exception). The caller
    treats ``None`` as "fall back to the plain Tier-1 alert".

    Pure function: no I/O of its own beyond what the event source does.
    """
    if lookback_s < 0:
        log.warning("attribute called with negative lookback_s=%s; clamping", lookback_s)
        lookback_s = 0.0

    window_start_s = idle_event.started_at_s - lookback_s
    window_end_s = now_s
    if window_end_s < window_start_s:
        # Clock went backwards or now_s precedes the idle event — refuse.
        log.warning(
            "attribute: window_end (%s) precedes window_start (%s); skipping",
            window_end_s,
            window_start_s,
        )
        return None

    try:
        events = event_source.capture(
            idle_event.gpu_index, window_start_s, window_end_s
        )
    except Exception:
        # An event source must not crash the daemon. Log and degrade.
        log.exception("event source raised during capture(); skipping attribution")
        return None

    if not events:
        # Empty window — no signal at all.
        return None

    trace = _build_trace(events, window_start_s, window_end_s)

    if _trace_is_warmup(trace):
        # Too little signal to diagnose. Mirrors the engine's warmup_trace_guard
        # so the agent rejects exactly the windows the engine would, instead of
        # the old flat MIN_EVENTS count that dropped diagnosable sparse windows
        # (e.g. a 3-event checkpoint stall whose kernel time alone has signal).
        return None

    try:
        return diagnose(trace)
    except Exception:
        log.exception("engine.diagnose() raised; skipping attribution")
        return None


def format_verdict_stdout(
    gpu_index: int,
    diagnosis: Diagnosis,
    *,
    at: datetime | None = None,
) -> str:
    """Plain stdout line: wall-clock timestamp, GPU index, and engine verdict."""
    ts = (at or datetime.now(timezone.utc)).isoformat()
    verdict = diagnosis.verdict.value.upper()
    conf_part = (
        ""
        if diagnosis.confidence is None
        else f" confidence={diagnosis.confidence:.2f}"
    )
    return f"{ts} gpu={gpu_index} verdict={verdict}{conf_part} {diagnosis.summary}"


def format_attributed_alert(
    idle_event: "IdleEvent",
    diagnosis: Diagnosis | None,
    sustain_s: float,
) -> str:
    """Render the user-facing alert line for an idle episode.

    With a diagnosis: include the verdict, confidence, and one-line summary.
    Without one (None): fall back to the plain Tier-1 "attribution pending"
    text so the output shape stays identical to the no-attribution path.
    """
    pct = idle_event.mean_util * 100
    if diagnosis is None:
        return (
            f"[bold red]ALERT[/] GPU {idle_event.gpu_index} idle for "
            f"{sustain_s:g}s at {pct:.1f}% — attribution pending (Tier 2)"
        )
    verdict = diagnosis.verdict.value.upper()
    conf_suffix = (
        ""
        if diagnosis.confidence is None
        else f" ({diagnosis.confidence * 100:.0f}%)"
    )
    return (
        f"[bold red]ALERT[/] GPU {idle_event.gpu_index} idle for "
        f"{sustain_s:g}s at {pct:.1f}% — {verdict}{conf_suffix}: "
        f"{diagnosis.summary}"
    )


__all__ = [
    "MIN_EVENTS",
    "attribute",
    "format_attributed_alert",
    "format_verdict_stdout",
]
