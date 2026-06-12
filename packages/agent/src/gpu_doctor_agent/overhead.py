"""Overhead measurement harness for ``TorchHookEventSource``.

The agent's claim — "live profiling adds ~negligible overhead to a real
training step" — has to be falsifiable, not folklore. This module runs a
caller-supplied workload twice (baseline + instrumented), computes wall-clock
means/stddevs, and reports an overhead percentage.

The Colab notebook supplies a real CUDA training step as ``workload_fn``;
local tests supply a trivial CPU lambda to exercise the pure stats path
without torch. The stats helpers themselves are pulled out into pure
functions so they're fully testable on the no-GPU dev box.

WHEN ``overhead_pct`` IS MEANINGFUL
-----------------------------------
``overhead_pct`` is a *ratio* and inherits the pathologies of small
denominators. The numerator absorbs a roughly fixed cost per step that
the profiler pays regardless of workload size: hook dispatch, kineto
event records, per-event Python object allocation, and the
session-open/session-close bracket this harness deliberately repeats
every repeat (see ``measure_overhead``). Call that fixed cost ``F``.
Then for a baseline step of duration ``B``:

    overhead_pct ≈ 100 * F / B

If ``B`` is sub-millisecond — a 256-wide MLP, a 32-element batch, a
``loss.item()`` and not much else — ``F/B`` is order-1 and the
reported percentage is dominated by bookkeeping, not by profiling
*the workload*. Early local microbenchmarks reported +196% overhead
for exactly this reason: the step was ~1 ms, ``F`` was a few hundred
microseconds, and the ratio screamed even though the absolute cost
was tiny. On a realistic ~32 ms training step (e.g. a 4096-wide
model, larger batch), the same ``F`` rounds to ≈0% within run-to-run
noise. Treat any number measured on a sub-10 ms workload as a
profiler-bookkeeping benchmark, NOT as a statement about agent cost.

Rule of thumb: the workload should be at least one order of magnitude
larger than the per-call profiler overhead before the percentage means
anything. In practice that means tens of milliseconds per step or more
— the regime real training jobs actually live in.

WORST-CASE UPPER BOUND
----------------------
What this harness measures is *continuous* profiling: every step runs
with an active profiler session. PRODUCTION DOES NOT DO THIS. The agent
operates in two tiers:

  * Tier-1 (always-on) is NVML sampling, which is out-of-process and
    contributes no torch profiler overhead at all.
  * Tier-2 (BURST capture) opens a ``TorchHookEventSource`` only AFTER
    Tier-1 has already detected a candidate idle episode, profiles for
    a brief window to attribute it, then closes the session.

So the workloads under live profiling are a small fraction of total
training steps, and the steady-state overhead a user actually pays is
strictly lower than whatever ``measure_overhead`` reports here. The
number this harness produces is the worst-case upper bound for the
"if we left the profiler on forever" scenario, not the cost of normal
operation.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Callable, Sequence

from gpu_doctor_agent.torch_source import TorchHookEventSource, TorchUnavailable

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure stats helpers — no torch, no clocks. Tested directly.
# ---------------------------------------------------------------------------


def mean(values: Sequence[float]) -> float:
    """Arithmetic mean. Returns 0.0 on an empty sequence (no division)."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def stddev(values: Sequence[float]) -> float:
    """Population standard deviation. Returns 0.0 for sequences of length < 2.

    We use the population (N-divisor) form, not the sample (N-1) form,
    because the values are a complete enumeration of the timed runs — not
    a sample from a larger population — and the N-divisor result is what
    downstream readers (notebook, logs) intuitively interpret as "spread".
    """
    n = len(values)
    if n < 2:
        return 0.0
    m = mean(values)
    variance = sum((v - m) ** 2 for v in values) / n
    return math.sqrt(variance)


def overhead_pct(baseline: float, instrumented: float) -> float:
    """Relative overhead of instrumented vs baseline, expressed as a percent.

    Returns 0.0 if ``baseline <= 0`` (no meaningful ratio). The sign is
    preserved: a faster instrumented run yields a negative number, which
    is occasionally seen with sub-millisecond workloads due to scheduler
    noise — we report it honestly rather than clamping to zero.
    """
    if baseline <= 0:
        return 0.0
    return ((instrumented - baseline) / baseline) * 100.0


# ---------------------------------------------------------------------------
# Measurement harness — torch.profiler-bound, raises TorchUnavailable
# without torch (the no-torch-test path lives in test_overhead.py).
# ---------------------------------------------------------------------------


def _time_one(workload_fn: Callable[[], None]) -> float:
    """Run ``workload_fn`` once, return wall-clock seconds."""
    t0 = time.perf_counter()
    workload_fn()
    return time.perf_counter() - t0


def measure_overhead(
    workload_fn: Callable[[], None],
    repeats: int = 10,
) -> dict:
    """Time ``workload_fn`` baseline vs. under an active TorchHookEventSource.

    The caller supplies the workload (typically one optimizer step on a
    small model). We run it ``repeats`` times without instrumentation,
    then ``repeats`` times wrapped in a fresh profiler session per call —
    so the measurement reflects the steady-state per-step overhead, not
    one-time session-startup cost amortised across many steps.

    Returns a dict with:

      * ``baseline_mean_s``,    ``baseline_stddev_s``    — uninstrumented
      * ``instrumented_mean_s``,``instrumented_stddev_s`` — with profiler
      * ``overhead_pct``        — (instr - base) / base * 100
      * ``repeats``             — echoed for log clarity
      * ``baseline_times_s``,   ``instrumented_times_s`` — raw samples

    Raises ``TorchUnavailable`` on a torch-less host (no fallback path —
    overhead is meaningless if there's no profiler attached).
    """
    if repeats < 1:
        raise ValueError(f"repeats must be >= 1, got {repeats}")
    if not TorchHookEventSource.is_available():
        raise TorchUnavailable(
            "measure_overhead requires torch; run on a GPU box or Colab."
        )

    # Baseline: workload alone, no profiler ever touched.
    baseline_times: list[float] = [_time_one(workload_fn) for _ in range(repeats)]

    # Instrumented: a FRESH profiler session per run. This is deliberately
    # NOT one long session — the agent's real usage opens a session, drains
    # events, closes it, then reopens; measuring the same shape keeps the
    # overhead number honest.
    instrumented_times: list[float] = []
    for _ in range(repeats):
        src = TorchHookEventSource()
        src.start()
        try:
            t0 = time.perf_counter()
            workload_fn()
            elapsed = time.perf_counter() - t0
        finally:
            src.stop()
        instrumented_times.append(elapsed)

    base_mean = mean(baseline_times)
    instr_mean = mean(instrumented_times)
    return {
        "repeats": repeats,
        "baseline_mean_s": base_mean,
        "baseline_stddev_s": stddev(baseline_times),
        "instrumented_mean_s": instr_mean,
        "instrumented_stddev_s": stddev(instrumented_times),
        "overhead_pct": overhead_pct(base_mean, instr_mean),
        "baseline_times_s": list(baseline_times),
        "instrumented_times_s": list(instrumented_times),
    }


__all__ = [
    "mean",
    "stddev",
    "overhead_pct",
    "measure_overhead",
]
