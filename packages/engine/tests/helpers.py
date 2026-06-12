"""Shared factory helpers for building synthetic traces in tests.

These are plain functions (not pytest fixtures) so they can be imported
directly by any test module.
"""

from __future__ import annotations

from gpu_doctor_engine.types import Event, Trace


def make_kernel_event(ts: int, dur: int, name: str = "volta_sgemm") -> Event:
    """Return a synthetic GPU kernel event."""
    return Event(name=name, category="kernel", pid=1, tid=1, ts=ts, dur=dur)


def make_cpu_event(name: str, ts: int, dur: int) -> Event:
    """Return a synthetic CPU op event."""
    return Event(name=name, category="cpu_op", pid=0, tid=0, ts=ts, dur=dur)


def make_trace(
    events: list[Event],
    duration_us: int,
    gpu_kernel_time_us: int,
    cpu_time_us: int = 0,
) -> Trace:
    """Build a Trace with the given events, sorted by start time."""
    return Trace(
        events=sorted(events, key=lambda e: e.ts),
        duration_us=duration_us,
        gpu_kernel_time_us=gpu_kernel_time_us,
        cpu_time_us=cpu_time_us,
    )


# ---------------------------------------------------------------------------
# High-level synthetic trace builders for boundary tests
# ---------------------------------------------------------------------------

_DUMMY_NAME = "aten::_dummy_filler"  # name that matches zero pattern lists


def trace_at_util(util: float, total_us: int = 1_000_000) -> Trace:
    """Trace at precise GPU util with NO suspect patterns. For HEALTHY threshold tests.

    Uses 10 evenly-spaced kernel events so the warmup guard never fires
    (n_kernels >= 5) and gpu_kernel_time_us is set exactly to util * total_us.
    """
    kernel_us = int(util * total_us)
    n_kernels = 10
    per_kernel = kernel_us // n_kernels
    spacing = total_us // n_kernels
    events = [
        make_kernel_event(ts=i * spacing, dur=per_kernel) for i in range(n_kernels)
    ]
    return make_trace(events, duration_us=total_us, gpu_kernel_time_us=kernel_us)


def trace_with_dataloader_share(
    util: float, dl_share: float, total_us: int = 1_000_000
) -> Trace:
    """Trace at given util with DataLoader events covering given share of idle."""
    kernel_us = int(util * total_us)
    idle_us = total_us - kernel_us
    dl_us = int(dl_share * idle_us)
    remaining_us = idle_us - dl_us
    events: list[Event] = [make_kernel_event(ts=0, dur=kernel_us)]
    if dl_us > 0:
        events.append(make_cpu_event("DataLoader__next_data", ts=kernel_us, dur=dl_us))
    if remaining_us > 0:
        events.append(
            make_cpu_event(_DUMMY_NAME, ts=kernel_us + dl_us, dur=remaining_us)
        )
    return make_trace(events, duration_us=total_us, gpu_kernel_time_us=kernel_us)


def trace_with_memcpy_ratio(memcpy_ratio: float, total_gpu_us: int = 500_000) -> Trace:
    """Trace where gpu_memcpy_time / (kernel_time + memcpy_time) = ratio.

    Uses gpu_memcpy category so pcie_ratio_50 is the relevant rule.  A large
    idle gap (4× the GPU work time) ensures specific_30 cannot false-fire when
    the ratio is below 0.50: cat_memcpy_us / idle_us stays well under 0.30.
    """
    kernel_us = int((1.0 - memcpy_ratio) * total_gpu_us)
    memcpy_us = total_gpu_us - kernel_us
    idle_gap_us = total_gpu_us * 4
    total_us = total_gpu_us + idle_gap_us
    events: list[Event] = [
        Event(name="volta_sgemm", category="kernel", pid=1, tid=1, ts=0, dur=kernel_us),
        Event(
            name="gpu_memcpy_htod",
            category="gpu_memcpy",
            pid=1,
            tid=1,
            ts=kernel_us,
            dur=memcpy_us,
        ),
        make_cpu_event(_DUMMY_NAME, ts=total_gpu_us, dur=idle_gap_us),
    ]
    return make_trace(events, duration_us=total_us, gpu_kernel_time_us=kernel_us)


def trace_with_checkpoint_share(
    util: float,
    checkpoint_share: float,
    total_us: int = 1_000_000,
    dtoh_count: int = 100,
) -> Trace:
    """Trace at given util with checkpoint events covering given share of idle.

    Uses 'torch.save' (a CHECKPOINT_PATTERNS match) for the checkpoint span,
    plus `dtoh_count` synthetic 'Memcpy DtoH (Device -> Pageable)' events
    (default 100, well above the strong-signal threshold of 50). Tests that
    want to exercise the fallback share-only path can pass dtoh_count=0.
    """
    kernel_us = int(util * total_us)
    idle_us = total_us - kernel_us
    ckpt_us = int(checkpoint_share * idle_us)
    remaining_us = idle_us - ckpt_us
    events: list[Event] = [make_kernel_event(ts=0, dur=kernel_us)]
    if ckpt_us > 0:
        events.append(make_cpu_event("torch.save", ts=kernel_us, dur=ckpt_us))
    if remaining_us > 0:
        events.append(
            make_cpu_event(_DUMMY_NAME, ts=kernel_us + ckpt_us, dur=remaining_us)
        )
    for i in range(dtoh_count):
        events.append(
            Event(
                name="Memcpy DtoH (Device -> Pageable)",
                category="gpu_memcpy",
                pid=1,
                tid=1,
                ts=kernel_us + i,
                dur=0,
            )
        )
    return make_trace(events, duration_us=total_us, gpu_kernel_time_us=kernel_us)


def trace_with_tiny_kernels(
    n_kernels: int = 20, kernel_dur_us: int = 30, gap_us: int = 200
) -> Trace:
    """N tiny kernels separated by gaps. For KERNEL_LAUNCH_BOUND tests.

    All kernels have dur < 50 µs so tiny_kernel_ratio = 1.0, and gaps keep
    utilisation low (< 60%).  duration_us is computed from event timestamps so
    the util field is consistent with what the engine sees from events.
    """
    slot = kernel_dur_us + gap_us
    events = [
        make_kernel_event(ts=i * slot, dur=kernel_dur_us) for i in range(n_kernels)
    ]
    gpu_kernel_time_us = n_kernels * kernel_dur_us
    trace_end = (n_kernels - 1) * slot + kernel_dur_us
    return make_trace(
        events, duration_us=trace_end, gpu_kernel_time_us=gpu_kernel_time_us
    )


def trace_with_nccl_share(
    util: float, nccl_share: float, total_us: int = 1_000_000
) -> Trace:
    """Trace at given util with NCCL events covering given share of idle.

    Uses 'ncclAllReduce' (a NCCL_PATTERNS match).
    """
    kernel_us = int(util * total_us)
    idle_us = total_us - kernel_us
    nccl_us = int(nccl_share * idle_us)
    remaining_us = idle_us - nccl_us
    events: list[Event] = [make_kernel_event(ts=0, dur=kernel_us)]
    if nccl_us > 0:
        events.append(make_cpu_event("ncclAllReduce", ts=kernel_us, dur=nccl_us))
    if remaining_us > 0:
        events.append(
            make_cpu_event(_DUMMY_NAME, ts=kernel_us + nccl_us, dur=remaining_us)
        )
    return make_trace(events, duration_us=total_us, gpu_kernel_time_us=kernel_us)


def trace_warmup(
    n_kernels: int = 2, kernel_dur_us: int = 100, total_us: int = 5_000
) -> Trace:
    """Trace small enough to trigger warmup guard (when n_kernels < 5).

    The warmup guard fires when ALL FOUR hold: < 5 kernel events, gpu_active
    < 5 ms, duration < 50 ms, util < 0.85.  Callers can break one condition
    (e.g. pass n_kernels=6) to test the guard being skipped.
    """
    spacing = total_us // max(n_kernels, 1)
    events = [
        make_kernel_event(ts=i * spacing, dur=kernel_dur_us) for i in range(n_kernels)
    ]
    gpu_kernel_time_us = n_kernels * kernel_dur_us
    return make_trace(
        events, duration_us=total_us, gpu_kernel_time_us=gpu_kernel_time_us
    )
