"""The diagnostic rules engine. This is the actual product wedge."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from gpu_doctor_engine.detectors.attribution import overlap_idle_time
from gpu_doctor_engine.detectors.dataloader import (
    DATALOADER_PATTERNS,
    DataloaderBoundDetector,
)
from gpu_doctor_engine.detectors.checkpoint import (
    CheckpointBoundDetector,
)
from gpu_doctor_engine.detectors.confidence import confidence_from_share
from gpu_doctor_engine.detectors.nccl import NcclBoundDetector
from gpu_doctor_engine.ingest import (
    GPU_ALL_CATS,
    GPU_KERNEL_CATS,
    GPU_MEMCPY_CATS,
    _busy_time_us,
    _merge_intervals,
)
from gpu_doctor_engine.types import Diagnosis, Event, Trace, Verdict

MEMCPY_PATTERNS = ("Memcpy", "memcpy", "HtoD", "DtoH", "MemcpyAsync")
SYNC_PATTERNS = (
    "cudaStreamSynchronize",
    "cudaDeviceSynchronize",
    "cudaEventSynchronize",
    "aten::item",
    "aten::_local_scalar_dense",
)

# pcie_ratio = gpu_memcpy_time_us / gpu_active_us is ACTIVE-normalized, so it is
# unstable when gpu_active_us is tiny: with active=40us and memcpy=20us it reads
# 0.50 from two near-zero numbers and would fire PCIE_BOUND even though a far
# larger share of IDLE is sync/dataloader. PCIE_BOUND must reflect transfer-bound
# COMPUTE, not a near-idle GPU. We therefore require the GPU to have done at least
# this much active work before the pcie_ratio rule is trusted to fire (and before
# a high ratio counts as "pcie dominant" in the healthy gate). Below the floor the
# real story is the idle-attributed causes (sync/dataloader), so we fall through
# to them. This is a stability guard on an existing rule, not a new bottleneck
# threshold or confidence change.
_PCIE_MIN_ACTIVE_US = 1000


@dataclass
class RuleDecision:
    rule: str
    fired: bool
    value: float
    threshold: float
    # passed: this rule's own firing condition held. fired: this rule WON the
    # decision (exactly one per diagnosis). They differ when a rule passes its
    # condition but loses the dominant-cause competition — a fired-but-lost
    # candidate that --explain still surfaces.
    passed: bool = False
    note: str = ""


def _gpu_idle_intervals(
    events: list[Event], trace_start: int, trace_end: int
) -> list[tuple[int, int]]:
    """Return the inverse of GPU-busy intervals: when was the GPU sitting idle?"""
    busy = [(e.ts, e.ts + e.dur) for e in events if e.category in GPU_ALL_CATS]
    busy = _merge_intervals(busy)

    idle: list[tuple[int, int]] = []
    cursor = trace_start
    for b_start, b_end in busy:
        if b_start > cursor:
            idle.append((cursor, b_start))
        cursor = max(cursor, b_end)
    if cursor < trace_end:
        idle.append((cursor, trace_end))
    return idle


def _overlap_time(
    intervals: list[tuple[int, int]],
    events: list[Event],
    patterns: tuple[str, ...],
) -> int:
    """How much of the given intervals overlap with events matching any pattern?"""
    return overlap_idle_time(intervals, events, patterns)


_SYNC_LOOKAHEAD_US = 500


def _sync_attributed_idle(
    idle_intervals: list[tuple[int, int]],
    events: list[Event],
    patterns: tuple[str, ...],
) -> int:
    """Idle time attributable to CPU<->GPU sync stalls.

    An idle interval is attributed to sync when a matching event ends within
    _SYNC_LOOKAHEAD_US before the interval starts.  This captures the
    post-sync dispatch gap: sync unblocks the CPU, the CPU prepares the next
    kernel, and the GPU sits idle during that brief window.

    The simpler 'overlap' model (sync running while GPU is already idle) gives
    false positives for DataLoader traces where a long cudaDeviceSynchronize
    spans the GPU-busy→idle transition but the idle itself is DataLoader-caused.
    Using end-before-start correctly attributes only the gaps that appear
    because the CPU is still dispatching after the sync returned.
    """
    # Build sorted list of sync-event end times for an O(n log n + m) scan.
    sync_ends = sorted(
        e.ts + e.dur
        for e in events
        if any(p.lower() in e.name.lower() for p in patterns)
    )

    total = 0
    k = 0  # index into sync_ends
    n = len(sync_ends)

    for i_start, i_end in idle_intervals:
        # Binary-search lower bound: first sync_end >= i_start - _SYNC_LOOKAHEAD_US
        lo, hi = k, n
        target = i_start - _SYNC_LOOKAHEAD_US
        while lo < hi:
            mid = (lo + hi) // 2
            if sync_ends[mid] < target:
                lo = mid + 1
            else:
                hi = mid

        # Advance the global pointer so future iterations skip old events.
        k = lo

        # Any sync_end in [i_start - LOOKAHEAD, i_start] attributes this interval.
        if k < n and sync_ends[k] <= i_start:
            total += i_end - i_start

    return total


def _checkpoint_dtoh_count(trace: Trace) -> int:
    """Count GPU Device-to-Pageable memcpy events.

    torch.save() pulls every tensor down to unpinned CPU memory before
    pickling, producing a 'Memcpy DtoH (Device -> Pageable)' event per
    tensor. Normal training rarely produces these (it uses Pinned memory
    or stays GPU-resident). A high count is a strong checkpoint signal.
    """
    return sum(
        1
        for e in trace.events
        if e.category == "gpu_memcpy" and "DtoH" in e.name and "Pageable" in e.name
    )


def _interval_overlap(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> int:
    """Total overlap between two sorted, non-overlapping interval lists."""
    total = 0
    i = j = 0
    while i < len(a) and j < len(b):
        a_start, a_end = a[i]
        b_start, b_end = b[j]
        total += max(0, min(a_end, b_end) - max(a_start, b_start))
        if a_end < b_end:
            i += 1
        else:
            j += 1
    return total


def _compute_idle_intervals(
    events: list[Event], trace_start: int, trace_end: int
) -> list[tuple[int, int]]:
    """Idle intervals using only GPU compute kernels as busy (excludes memcpy/memset)."""
    busy = _merge_intervals(
        [(e.ts, e.ts + e.dur) for e in events if e.category in GPU_KERNEL_CATS]
    )
    idle: list[tuple[int, int]] = []
    cursor = trace_start
    for b_start, b_end in busy:
        if b_start > cursor:
            idle.append((cursor, b_start))
        cursor = max(cursor, b_end)
    if cursor < trace_end:
        idle.append((cursor, trace_end))
    return idle


def _hol_blocking_stats(trace: Trace) -> dict | None:
    """Return head-of-line blocking statistics for DataLoader events, or None if absent.

    Computes the distribution of DataLoader fetch durations and checks whether a
    small number of unusually slow samples dominate stall time (MinatoLoader signal).
    """
    durations = [
        e.dur
        for e in trace.events
        if any(p.lower() in e.name.lower() for p in DATALOADER_PATTERNS)
    ]
    if not durations:
        return None

    arr = np.array(durations, dtype=float)
    median_us = float(np.percentile(arr, 50))
    # method='higher' returns an actual observed value (the worst sample in the top 1%)
    # rather than a linear interpolation, which is correct for outlier detection.
    p99_us = float(np.percentile(arr, 99, method="higher"))
    hol_ratio = p99_us / max(median_us, 1.0)

    return {
        "median_us": median_us,
        "p99_us": p99_us,
        "hol_ratio": hol_ratio,
        "hol_blocking_likely": hol_ratio > 10,
        "sample_count": len(durations),
    }


def _is_warmup_trace(trace: Trace) -> bool:
    """Detect traces with too little signal to diagnose (e.g. pure warm-up).

    A trace is warm-up only if ALL of the following hold:
    - Fewer than 5 GPU kernel events
    - AND GPU active span under 5ms
    - AND total duration under 50ms
    - AND GPU utilization below 85%

    Any one of these being substantial means there IS diagnostic signal.
    High utilization especially: a GPU that is ≥85% busy is never a warm-up trace
    regardless of absolute size, and must still reach the healthy-verdict rules.
    """
    kernel_events = [e for e in trace.events if e.category in GPU_KERNEL_CATS]
    if len(kernel_events) >= 5:
        return False
    if trace.gpu_kernel_time_us >= 5_000:
        return False
    if trace.duration_us >= 50_000:
        return False
    if trace.gpu_utilization >= 0.85:
        return False
    return True


def _diagnose_core(trace: Trace) -> tuple[Diagnosis, dict]:
    """Core logic returning (Diagnosis, stats_dict). Use diagnose() for the public API."""
    if not trace.events:
        d = _unknown("Trace contained no events.")
        return d, {}

    util = trace.gpu_utilization

    # Compute all metrics upfront so every rule can be fully evaluated for the decision log.
    trace_start = trace.events[0].ts
    trace_end = max(e.ts + e.dur for e in trace.events)
    idle_intervals = _gpu_idle_intervals(trace.events, trace_start, trace_end)
    idle_us = sum(end - start for start, end in idle_intervals)

    dl_measurement = DataloaderBoundDetector.measure(
        idle_intervals, trace.events, idle_us
    )
    nccl_measurement = NcclBoundDetector.measure(idle_intervals, trace.events, idle_us)
    dataloader_us = dl_measurement.overlap_us
    nccl_us = nccl_measurement.overlap_us
    memcpy_name_us = _overlap_time(idle_intervals, trace.events, MEMCPY_PATTERNS)
    ckpt_measurement = CheckpointBoundDetector.measure(
        idle_intervals, trace.events, idle_us
    )
    checkpoint_us = ckpt_measurement.overlap_us
    sync_us = _sync_attributed_idle(idle_intervals, trace.events, SYNC_PATTERNS)

    # Bug A: category-based PCIe detection.
    # gpu_memcpy events run ON the GPU so they appear in GPU_ALL_CATS and are excluded from
    # idle windows. We detect them two ways:
    #   1. Overlap with *compute-idle* (kernel-only baseline) — catches transfers that fill
    #      gaps between kernels.
    #   2. Ratio of gpu_memcpy time to total GPU active time — catches traces where the GPU
    #      is almost entirely doing copies instead of compute.
    # Take the higher of name-pattern and category-based signals for the candidates stage.
    gpu_memcpy_time_us = _busy_time_us(trace.events, GPU_MEMCPY_CATS)
    compute_idle = _compute_idle_intervals(trace.events, trace_start, trace_end)
    cat_memcpy_merged = _merge_intervals(
        [(e.ts, e.ts + e.dur) for e in trace.events if e.category in GPU_MEMCPY_CATS]
    )
    cat_memcpy_us = _interval_overlap(compute_idle, cat_memcpy_merged)
    memcpy_us = max(memcpy_name_us, cat_memcpy_us)

    gpu_active_us = trace.gpu_kernel_time_us + gpu_memcpy_time_us

    kernels = [e for e in trace.events if e.category in GPU_KERNEL_CATS]
    if kernels:
        avg_kernel_dur = sum(k.dur for k in kernels) / len(kernels)
        tiny_kernel_ratio = sum(1 for k in kernels if k.dur < 50) / len(kernels)
    else:
        avg_kernel_dur = 0.0
        tiny_kernel_ratio = 0.0

    pcie_ratio = gpu_memcpy_time_us / max(gpu_active_us, 1)
    nccl_share = nccl_us / max(idle_us, 1)
    ckpt_share = checkpoint_us / max(idle_us, 1)
    sync_fraction = sync_us / max(idle_us, 1)
    dl_share = dataloader_us / max(idle_us, 1)

    stats: dict = {
        "util": util,
        "idle_us": idle_us,
        "dataloader_us": dataloader_us,
        "nccl_us": nccl_us,
        "memcpy_name_us": memcpy_name_us,
        "cat_memcpy_us": cat_memcpy_us,
        "memcpy_us": memcpy_us,
        "gpu_memcpy_time_us": gpu_memcpy_time_us,
        "gpu_active_us": gpu_active_us,
        "avg_kernel_dur": avg_kernel_dur,
        "tiny_kernel_ratio": tiny_kernel_ratio,
        "checkpoint_us": checkpoint_us,
        "sync_us": sync_us,
        "sync_fraction": sync_fraction,
    }

    decisions: list[RuleDecision] = []
    verdict_diag: Diagnosis | None = None
    rule_key = "unknown"

    # Guard 0: warmup_trace_guard — short-circuit before any rule when the trace has
    # too little signal to diagnose. Fires only when ALL conditions in _is_warmup_trace
    # hold: fewer than 5 kernel events, GPU active < 5ms, wall-clock < 50ms, AND
    # GPU utilization < 85%. High-util traces always reach the healthy-verdict rules.
    is_warmup = _is_warmup_trace(trace)
    decisions.append(
        RuleDecision(
            rule="warmup_trace_guard",
            fired=is_warmup,
            passed=is_warmup,
            value=float(len(kernels)),
            threshold=5.0,
            note=(
                f"kernel_events={len(kernels)} "
                f"gpu_active_us={trace.gpu_kernel_time_us} "
                f"duration_us={trace.duration_us}"
            ),
        )
    )
    if is_warmup:
        stats["rule"] = "warmup_trace_guard"
        stats["decisions"] = [asdict(d) for d in decisions]
        return _unknown(
            "Trace too small to diagnose (warm-up only).",
            metrics={"gpu_util": util, "idle_ms": idle_us / 1000},
        ), stats

    # ------------------------------------------------------------------
    # Decision flow (restructured). Two stages, replacing the old
    # first-match-wins cascade that let a weaker/earlier rule claim a trace
    # a stronger/later rule should win:
    #
    #   1. HEALTHY is a true GATE. A healthy verdict only claims a trace when
    #      NO bottleneck cause is strong enough to fire its own rule.
    #      no_dominant now accounts for sync and checkpoint as well as
    #      nccl/pcie (each bound is the SAME threshold that cause fires at),
    #      so a sync- or checkpoint-dominated trace can no longer be called
    #      healthy. (Kills "healthy while X dominates".)
    #
    #   2. DOMINANT-CAUSE COMPETITION. Every idle-attributed cause that
    #      passes its own firing condition becomes a candidate; the cause
    #      covering the LARGEST share of idle wins (generalising the old
    #      nccl-vs-pcie sort at Rule 6 to all idle causes). These specific
    #      causes outrank the generic DataLoader fallback (DataLoader is the
    #      outer call that contains them). kernel_launch_tiny is a SYMPTOM of
    #      GPU starvation, not a root cause, so it can only win when NO idle
    #      cause fired — otherwise the dominant idle cause wins. (Kills
    #      "symptom beats cause".)
    #
    # Exactly one rule is recorded fired=True (the winner). A rule whose own
    # condition held but lost the competition is recorded passed=True,
    # fired=False so --explain still surfaces it.
    # Warm-up traces are short-circuited above by warmup_trace_guard.
    # ------------------------------------------------------------------
    dtoh_count = _checkpoint_dtoh_count(trace)
    stats["checkpoint_dtoh_count"] = dtoh_count

    # pcie_ratio_fired is hoisted ABOVE no_dominant on purpose: both the firing
    # rule (Stage 2) and the HEALTHY gate must share ONE definition of "PCIe
    # dominates". A high pcie_ratio is only trustworthy once the GPU has done a
    # floor of real work (gpu_active_us >= _PCIE_MIN_ACTIVE_US); below that the
    # ratio is an artifact of two tiny numbers (see _PCIE_MIN_ACTIVE_US) and must
    # neither fire PCIE_BOUND nor block HEALTHY.
    pcie_ratio_fired = pcie_ratio >= 0.50 and gpu_active_us >= _PCIE_MIN_ACTIVE_US

    # no_dominant: HEALTHY is a true GATE — allowed only when no cause is
    # strong enough to fire its own rule. Adding sync_fraction and ckpt_share
    # here (alongside the original nccl/pcie) is the Bug 2 fix: a sync- or
    # checkpoint-dominated trace can no longer slip through as "healthy".
    # Each bound is exactly that cause's own firing condition — nccl 0.50,
    # pcie via the shared pcie_ratio_fired (ratio >= 0.50 AND the active-time
    # floor), sync 0.25 (sync_25), checkpoint 0.25 (checkpoint_25_fallback) — so
    # "strong enough to fire its rule" and "blocks HEALTHY" stay identical.
    no_dominant = (
        nccl_share < 0.50
        and not pcie_ratio_fired
        and sync_fraction < 0.25
        and ckpt_share < 0.25
    )
    # dl_share is gated by util tier ON PURPOSE, because dl_share is
    # dataloader-time / IDLE-time, NOT / total-time:
    #   * healthy_70_no_dominant (util >= 0.70): idle is small, so a high
    #     dl_share just means a healthy OVERLAPPED prefetch is filling those
    #     small gaps — it is NOT a bottleneck and must not block HEALTHY. So
    #     this gate is deliberately left UNGATED on dl_share (this is why
    #     fixtures/healthy.json at util=74.8%, dl_share=71.5% is correctly
    #     HEALTHY).
    #   * borderline_healthy_45 (45-70% util): the GPU is substantially idle,
    #     so a high dl_share means the dataloader is CAUSING that idle and
    #     must block the healthy call — hence the extra dl_share < 0.40 clause
    #     here (unchanged from the previous behaviour).
    borderline_no_dominant = no_dominant and dl_share < 0.40

    # Firing condition for every bottleneck rule — each is exactly the
    # condition the original rule used; only the SELECTION among them changed.
    # (pcie_ratio_fired is defined above — hoisted so the HEALTHY gate reuses the
    # identical definition, including the gpu_active_us floor.)
    ckpt_fired = CheckpointBoundDetector.fired(ckpt_measurement)
    sync_fired = util < 0.70 and sync_fraction >= 0.25 and avg_kernel_dur >= 50
    nccl_fired = NcclBoundDetector.fired(nccl_measurement)
    memcpy_share = memcpy_us / max(idle_us, 1)
    pcie_idle_fired = memcpy_us > 0 and memcpy_share >= NcclBoundDetector.THRESHOLD
    dl_fired = DataloaderBoundDetector.fired(dl_measurement)
    kernel_launch_fired = (
        tiny_kernel_ratio > 0.50 and avg_kernel_dur < 100 and util < 0.60
    )

    # winner: name of the rule that determined the verdict (matching its
    # decision-log entry), or None for the UNKNOWN fallthrough.
    winner = None

    # --- Stage 1: HEALTHY gate (first match wins) -------------------------
    if util >= 0.85:
        winner = "healthy_85"
        rule_key = "util>=0.85"
        verdict_diag = Diagnosis(
            verdict=Verdict.HEALTHY,
            confidence=None,
            summary=f"GPU utilization is {util:.0%}. Looks good.",
            evidence=[f"GPU busy time: {trace.gpu_kernel_time_us / 1000:.1f}ms"],
            recommended_actions=[],
            metrics={"gpu_util": util},
        )
    elif util >= 0.70 and no_dominant:
        winner = "healthy_70_no_dominant"
        rule_key = "Bug B util>=0.70 no dominant specific cause"
        verdict_diag = Diagnosis(
            verdict=Verdict.HEALTHY,
            confidence=None,
            summary=(
                f"GPU utilization is {util:.0%}. "
                "No dominant bottleneck — training looks healthy."
            ),
            evidence=[
                f"GPU busy time: {trace.gpu_kernel_time_us / 1000:.1f}ms",
                f"GPU idle time: {idle_us / 1000:.1f}ms ({idle_us / max(trace_end - trace_start, 1):.0%})",
            ],
            recommended_actions=[],
            metrics={"gpu_util": util},
        )
    elif util >= 0.45 and borderline_no_dominant:
        winner = "borderline_healthy_45"
        rule_key = "borderline util>=0.45 no dominant cause"
        verdict_diag = Diagnosis(
            verdict=Verdict.HEALTHY,
            confidence=None,
            summary=(
                f"GPU utilization is {util:.0%}. "
                "Utilization is moderate with no single dominant bottleneck identified."
            ),
            evidence=[
                f"GPU busy time: {trace.gpu_kernel_time_us / 1000:.1f}ms",
                f"GPU idle time: {idle_us / 1000:.1f}ms "
                f"({idle_us / max(trace_end - trace_start, 1):.0%})",
            ],
            recommended_actions=[
                "Consider profiling a longer trace to confirm.",
                "Increasing batch size may improve utilization if compute-bound.",
            ],
            metrics={"gpu_util": util},
        )
    else:
        # --- Stage 2: dominant-cause competition --------------------------
        # Collect every specific idle-attributed cause that fired, tagged
        # with the ABSOLUTE microseconds of GPU time that cause accounts for
        # (its share's numerator) and the diag/confidence function that verdict
        # already used (preserved verbatim from the old rules).
        #
        # The ranking key is absolute GPU-µs, NOT the share, ON PURPOSE. The
        # shares are not commensurable: pcie_ratio_50 is memcpy/gpu_ACTIVE while
        # the idle causes are <cause>_us/gpu_IDLE — different denominators, so
        # comparing them with `>` let a small copy fraction of a tiny active
        # window out-rank a large idle cause of a huge idle window (S4). Every
        # numerator below is the same currency — microseconds of GPU time
        # attributable to that bottleneck — so the largest absolute time is the
        # genuinely dominant cause. Do NOT "simplify" this back to the share.
        # (Firing gates are unchanged; only the SELECTION key changed.)
        specific = []
        if pcie_ratio_fired:
            specific.append(
                (
                    "pcie_ratio_50",
                    gpu_memcpy_time_us,
                    lambda: _pcie_diag(util, idle_us, gpu_memcpy_time_us, pcie_ratio),
                )
            )
        if ckpt_fired:
            specific.append(
                (
                    "checkpoint_25",
                    checkpoint_us,
                    lambda: CheckpointBoundDetector.build_diagnosis(
                        util, idle_us, checkpoint_us
                    ),
                )
            )
        if sync_fired:
            specific.append(
                (
                    "sync_25",
                    sync_us,
                    lambda: _sync_diag(util, idle_us, sync_us, sync_fraction),
                )
            )
        if nccl_fired:
            specific.append(
                (
                    "nccl_bound_30",
                    nccl_us,
                    lambda: NcclBoundDetector.build_diagnosis(util, idle_us, nccl_us),
                )
            )
        if pcie_idle_fired:
            specific.append(
                (
                    "pcie_idle_30",
                    memcpy_us,
                    lambda: _pcie_diag(util, idle_us, memcpy_us, memcpy_share),
                )
            )

        if specific:
            # Dominant-cause: the cause accounting for the largest absolute
            # amount of GPU time (µs) wins. Stable sort -> decision-log order is
            # the tie-break. (See the commensurability note above: the key is
            # absolute µs, not the unit-mismatched share.)
            specific.sort(key=lambda c: c[1], reverse=True)
            winner, _, _winning_diag = specific[0]
            verdict_diag = _winning_diag()
            rule_key = {
                "pcie_ratio_50": "Bug A gpu_memcpy dominates GPU active time",
                "checkpoint_25": "checkpoint>=25% of idle",
                "sync_25": "sync>=25% of idle",
                "nccl_bound_30": "nccl>=30% of idle",
                "pcie_idle_30": "memcpy>=30% of idle",
            }[winner]
        elif dl_fired:
            # DataLoader is the generic fallback: wins only when no more
            # specific idle cause fired, but still beats the kernel-launch
            # symptom.
            winner = "dataloader_fallback"
            rule_key = "dataloader>=20% of idle"
            hol_stats = _hol_blocking_stats(trace)
            verdict_diag = _dataloader_diag(
                util, idle_us, dataloader_us, dl_share, hol_stats=hol_stats
            )
            stats["hol_stats"] = hol_stats
        elif kernel_launch_fired:
            # Tiny kernels are a SYMPTOM of starvation, so they win only as a
            # last resort — when no idle-attributed cause claimed the trace.
            winner = "kernel_launch_tiny"
            rule_key = "tiny_kernel_ratio>0.5"
            verdict_diag = _kernel_launch_diag(util, avg_kernel_dur, tiny_kernel_ratio)
        # else: winner stays None -> UNKNOWN fallthrough below.

    # --- Decision log: every rule, in a fixed order -----------------------
    # fired=True is the single winner; passed=True marks a rule whose own
    # condition held (a fired-but-lost candidate stays visible in --explain).
    def _record(
        name: str,
        passed: bool,
        value: float,
        threshold: float,
        note: str = "",
    ) -> None:
        decisions.append(
            RuleDecision(
                rule=name,
                fired=(name == winner),
                passed=passed,
                value=value,
                threshold=threshold,
                note=note,
            )
        )

    _record("healthy_85", util >= 0.85, util, 0.85)
    _record(
        "healthy_70_no_dominant",
        util >= 0.70 and no_dominant,
        util,
        0.70,
        note=(
            f"nccl_share={nccl_share:.2f} pcie_ratio={pcie_ratio:.2f} "
            f"sync={sync_fraction:.2f} ckpt={ckpt_share:.2f}"
        ),
    )
    _record(
        "pcie_ratio_50",
        pcie_ratio_fired,
        pcie_ratio,
        0.50,
        note=(
            f"gpu_active_us={gpu_active_us} "
            f"(needs >= {_PCIE_MIN_ACTIVE_US}; ratio is unstable below the floor)"
        ),
    )
    _record(
        "borderline_healthy_45",
        util >= 0.45 and borderline_no_dominant,
        util,
        0.45,
        note=(
            f"nccl_share={nccl_share:.2f} pcie_ratio={pcie_ratio:.2f} "
            f"dl_share={dl_share:.2f} sync={sync_fraction:.2f} ckpt={ckpt_share:.2f}"
        ),
    )
    _record(
        "checkpoint_25",
        ckpt_fired,
        ckpt_share,
        CheckpointBoundDetector.THRESHOLD,
        note=f"dtoh_pageable_count={dtoh_count} (informational only)",
    )
    _record(
        "sync_25",
        sync_fired,
        sync_fraction,
        0.25,
        note=f"avg_kernel_dur={avg_kernel_dur:.0f}us",
    )
    _record(
        "kernel_launch_tiny",
        kernel_launch_fired,
        tiny_kernel_ratio,
        0.50,
        note=f"avg_dur={avg_kernel_dur:.0f}us util={util:.2f} (symptom; loses to any idle cause)",
    )
    _record(
        "nccl_bound_30",
        nccl_fired,
        nccl_share,
        NcclBoundDetector.THRESHOLD,
        note="NCCL collective overlap with GPU idle",
    )
    _record(
        "pcie_idle_30",
        pcie_idle_fired,
        memcpy_share,
        NcclBoundDetector.THRESHOLD,
        note="Memcpy name/category overlap with GPU idle",
    )
    _record("dataloader_fallback", dl_fired, dl_share, 0.20)

    stats["rule"] = rule_key
    stats["decisions"] = [asdict(d) for d in decisions]

    if verdict_diag is None:
        verdict_diag = _unknown(
            f"GPU utilization is {util:.0%} but no clear bottleneck signature was found.",
            metrics={"gpu_util": util, "idle_ms": idle_us / 1000},
        )

    return verdict_diag, stats


def diagnose(trace: Trace) -> Diagnosis:
    """Diagnose a trace and return the Diagnosis. Public API — signature is stable."""
    diag, _ = _diagnose_core(trace)
    return diag


def diagnose_with_stats(trace: Trace) -> tuple[Diagnosis, dict]:
    """Diagnose a trace and return (Diagnosis, stats_dict) for --explain output."""
    return _diagnose_core(trace)


def _dataloader_diag(
    util: float,
    idle_us: int,
    dl_us: int,
    dl_share: float,
    hol_stats: dict | None = None,
) -> Diagnosis:
    actions = [
        "Increase DataLoader num_workers (try 4 or 8).",
        "Set persistent_workers=True to avoid worker re-spawn.",
        "Set pin_memory=True for faster H2D transfer.",
        "Move expensive preprocessing to a separate process or use an iterable dataset.",
        "Profile your __getitem__ and look for slow image decode or disk reads.",
    ]
    metrics: dict = {"gpu_util": util, "dataloader_us": dl_us, "idle_us": idle_us}

    if hol_stats is not None:
        metrics["hol_median_us"] = hol_stats["median_us"]
        metrics["hol_p99_us"] = hol_stats["p99_us"]
        metrics["hol_ratio"] = hol_stats["hol_ratio"]
        metrics["hol_blocking_likely"] = hol_stats["hol_blocking_likely"]
        metrics["hol_sample_count"] = hol_stats["sample_count"]
        if hol_stats["hol_blocking_likely"]:
            actions = [
                "Profile your __getitem__ for outlier samples (large images, slow disk reads,"
                " etc.). One slow sample blocks an entire worker.",
                *actions,
            ]

    return Diagnosis(
        verdict=Verdict.DATALOADER_BOUND,
        confidence=confidence_from_share(dl_share),
        summary=(
            f"GPU is {util:.0%} utilized. The dominant cause is dataloader stalls: "
            f"{dl_us / 1000:.0f}ms ({dl_share:.0%}) of GPU idle time "
            f"overlaps with PyTorch DataLoader activity on the CPU."
        ),
        evidence=[
            f"GPU utilization: {util:.0%}",
            f"Total GPU idle: {idle_us / 1000:.0f}ms",
            f"Dataloader time during idle: {dl_us / 1000:.0f}ms",
        ],
        recommended_actions=actions,
        metrics=metrics,
    )


def _pcie_diag(
    util: float, idle_us: int, mem_us: int, dominance_share: float
) -> Diagnosis:
    return Diagnosis(
        verdict=Verdict.PCIE_BOUND,
        confidence=confidence_from_share(dominance_share),
        summary=(
            f"GPU is {util:.0%} utilized. Host-device transfers dominate: "
            f"{mem_us / 1000:.0f}ms in cuMemcpy."
        ),
        evidence=[
            f"GPU utilization: {util:.0%}",
            f"Memcpy time: {mem_us / 1000:.0f}ms",
        ],
        recommended_actions=[
            "Use pin_memory=True in DataLoader.",
            "Use non_blocking=True on .to(device) calls.",
            "Batch larger transfers instead of many small ones.",
            "Consider keeping intermediate tensors on GPU.",
        ],
        metrics={"gpu_util": util, "memcpy_us": mem_us, "idle_us": idle_us},
    )


def _sync_diag(
    util: float,
    idle_us: int,
    sync_us: int,
    sync_fraction: float,
) -> Diagnosis:
    return Diagnosis(
        verdict=Verdict.SYNC_BOUND,
        confidence=confidence_from_share(sync_fraction),
        summary=(
            f"GPU is {util:.0%} utilized. {sync_fraction:.0%} of idle time is "
            "CPU<->GPU sync stalls (.item()/.cpu()/synchronize), which serialize "
            "the pipeline and drain the kernel queue."
        ),
        evidence=[
            f"GPU utilization: {util:.0%}",
            f"Total GPU idle: {idle_us / 1000:.0f}ms",
            f"Sync stall time during idle: {sync_us / 1000:.0f}ms",
        ],
        recommended_actions=[
            "Remove or defer .item(), .cpu(), .numpy() calls inside the training loop; "
            "accumulate metrics on-GPU and sync once every N steps.",
            "Remove explicit torch.cuda.synchronize() calls from the hot loop.",
            "Use non-blocking transfers and asynchronous logging instead of blocking "
            "device->host copies.",
            "Batch host-side metric computation to cut host-device round trips.",
        ],
        metrics={
            "gpu_util": util,
            "sync_us": sync_us,
            "sync_fraction": sync_fraction,
            "idle_us": idle_us,
        },
    )


def _kernel_launch_diag(util: float, avg_us: float, tiny_ratio: float) -> Diagnosis:
    return Diagnosis(
        verdict=Verdict.KERNEL_LAUNCH_BOUND,
        confidence=confidence_from_share(tiny_ratio),
        summary=(
            f"GPU is {util:.0%} utilized. Kernels are too small: average {avg_us:.0f}us, "
            f"{tiny_ratio:.0%} are under 50us. Launch overhead dominates."
        ),
        evidence=[
            f"GPU utilization: {util:.0%}",
            f"Average kernel duration: {avg_us:.0f}us",
            f"Tiny kernel ratio: {tiny_ratio:.0%}",
        ],
        recommended_actions=[
            "Increase batch size to make kernels bigger.",
            "Use torch.compile() to fuse small kernels.",
            "Use CUDA Graphs for repetitive kernel sequences.",
            "Check for unnecessary .item() or .cpu() calls causing sync points.",
        ],
        metrics={
            "gpu_util": util,
            "avg_kernel_us": avg_us,
            "tiny_kernel_ratio": tiny_ratio,
        },
    )


def _unknown(msg: str, metrics: dict | None = None) -> Diagnosis:
    return Diagnosis(
        verdict=Verdict.UNKNOWN,
        confidence=0.0,
        summary=msg,
        evidence=[],
        recommended_actions=[
            "Try recording a longer trace (>30 seconds).",
            "Verify torch.profiler captured both CPU and CUDA activities.",
        ],
        metrics=metrics or {},
    )
