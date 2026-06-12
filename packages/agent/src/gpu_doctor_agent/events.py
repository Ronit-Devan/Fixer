"""Pluggable event sources for Tier-2 attribution.

The agent's idle detector emits an `IdleEvent` when a GPU has been below the
idle threshold for the sustain window. At that moment the daemon needs a list
of recent CUDA-API / kernel events to feed to the engine's `diagnose()`. The
abstraction here decouples WHERE those events come from (eBPF, CUPTI, a
recorded trace file, a mock) from HOW they're consumed (attribution.py).

All implementations must return `gpu_doctor_engine.Event` objects whose
`category` field is one the engine recognizes:

  - ``"kernel"`` or ``"gpu_op"``  — GPU compute kernels.
  - ``"gpu_memcpy"`` / ``"gpu_memset"`` — GPU memory ops.
  - ``"cuda_runtime"``             — CUDA runtime API calls on the CPU side.
  - ``"cpu_op"`` / ``"python_function"`` / ``"user_annotation"`` — host ops,
    DataLoader iter, NCCL collective wrappers, sync calls (``aten::item``,
    ``cudaDeviceSynchronize``), etc.

Timestamps (`ts`) and durations (`dur`) MUST be in microseconds; that's the
engine's native unit and what `load_trace` produces from Chrome-trace JSON.
The `capture()` window is in seconds (monotonic) to match the agent loop's
clock — implementations are responsible for the unit conversion.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

from gpu_doctor_engine import Event, load_trace

log = logging.getLogger(__name__)


class EventSource(ABC):
    """Abstract pluggable source of engine-compatible events.

    A `capture()` call corresponds to "give me the events relevant to the
    idle window I just observed". Implementations must be side-effect-free
    on the daemon's hot path beyond their own cache, must never raise on
    ordinary missing-data conditions (return an empty list instead), and
    must produce engine-compatible Event objects with categories the engine
    understands (see module docstring).
    """

    @abstractmethod
    def capture(
        self, gpu_index: int, start_s: float, end_s: float
    ) -> list[Event]:
        """Return events occurring in the wall-clock window [start_s, end_s].

        Window bounds are seconds (typically monotonic). Returned `Event.ts`
        and `Event.dur` are microseconds (engine convention).

        Empty list is a legal return — the attribution layer treats that as
        "insufficient signal" and skips emitting a verdict.
        """


# ---------------------------------------------------------------------------
# Mock implementation: preset events, in-memory window filter.
# ---------------------------------------------------------------------------


class MockEventSource(EventSource):
    """Deterministic in-memory source. Drop-in for tests and dev demos.

    Two mutually-exclusive construction modes:

    - **Scenario mode** (``MockEventSource(scenario=...)``): events are
      synthesized PER ``capture()`` call and anchored to the requested window
      via ``window_start_us = int(start_s * 1_000_000)``. The engine's
      detectors only care about relative gaps and durations, so re-anchoring
      keeps the verdict stable while letting the events land inside whatever
      window the live agent loop happens to be on (monotonic clock or
      otherwise). Events whose end (``ts + dur``) would overshoot ``end_us``
      are trimmed so the closed-interval window contract still holds.

    - **Literal events mode** (``MockEventSource(events=[...])``): a fixed
      ``list[Event]`` is held verbatim and ``capture()`` returns the subset
      whose ``ts`` falls inside ``[start_us, end_us]``. Use this when the
      test's whole point is asserting window-filter exclusion behavior.

    ``gpu_index`` is accepted for interface compatibility but ignored — the
    mock has no notion of per-GPU partitioning.
    """

    _SCENARIOS: tuple[str, ...] = ("sync_bound", "dataloader_bound")

    def __init__(
        self,
        events: list[Event] | None = None,
        *,
        scenario: str | None = None,
    ) -> None:
        if events is not None and scenario is not None:
            raise ValueError("provide either `events` or `scenario`, not both")
        if scenario is not None and scenario not in self._SCENARIOS:
            raise ValueError(
                f"unknown scenario {scenario!r}; valid: {self._SCENARIOS}"
            )
        self._scenario: str | None = scenario
        # Defensive copy: callers must not be able to mutate our state.
        # Empty in scenario mode — those events don't exist until capture().
        self._events: list[Event] = list(events or [])

    @property
    def events(self) -> list[Event]:
        """Read-only view of the literal preset events.

        Returns ``[]`` in scenario mode — synthesized events are produced
        on-demand by ``capture()`` and never held on the instance.
        """
        return list(self._events)

    def capture(
        self, gpu_index: int, start_s: float, end_s: float
    ) -> list[Event]:
        if end_s < start_s:
            return []
        start_us: int = int(start_s * 1_000_000)
        end_us: int = int(end_s * 1_000_000)
        if self._scenario is not None:
            synthesized: list[Event]
            if self._scenario == "sync_bound":
                synthesized = build_sync_bound_events(window_start_us=start_us)
            else:  # "dataloader_bound" — _SCENARIOS is guarded in __init__.
                synthesized = build_dataloader_bound_events(
                    window_start_us=start_us
                )
            # Trim anything that overshoots the requested window. The default
            # 5 s lookback dwarfs the ~200 ms synthesized span, but a caller
            # asking for a sub-span window must still get a clean truncation.
            return [e for e in synthesized if e.ts + e.dur <= end_us]
        return [e for e in self._events if start_us <= e.ts <= end_us]


# ---------------------------------------------------------------------------
# File implementation: load a Chrome-trace JSON once, filter by window.
# ---------------------------------------------------------------------------


class FileEventSource(EventSource):
    """Replay a recorded PyTorch Profiler trace as if it were live attribution.

    On first `capture()` call, loads the trace through the engine's public
    `load_trace()` and caches the parsed events. Subsequent calls return the
    same cached list verbatim.

    Why `capture()` ignores its window/gpu_index arguments
    ------------------------------------------------------
    A recorded trace is a *self-contained episode* on the profiler's own time
    origin — `ts` values are microseconds since the profiler started, not
    monotonic wall-clock seconds aligned with the agent loop. The live-clock
    window `[started_at_s - lookback_s, now_s]` simply has no meaning for
    these timestamps: any naive intersection would return ``[]`` and force
    attribution to ``None``, never reaching the engine.

    For file replay the recorded trace IS the idle episode. The whole point
    is "diagnose this captured trace", so `capture()` returns every parsed
    event regardless of the window. ``gpu_index`` is likewise ignored —
    single-file traces are not per-GPU partitioned.

    Load failures are swallowed: a missing or malformed file logs a warning
    and `capture()` returns ``[]`` so attribution degrades to ``None``
    rather than crashing the daemon. The load is attempted exactly once;
    subsequent calls reuse the cached failure state (still ``[]``).
    """

    def __init__(self, path: str | Path) -> None:
        self._path: Path = Path(path)
        self._events: list[Event] | None = None
        self._load_attempted: bool = False

    @property
    def path(self) -> Path:
        return self._path

    def _ensure_loaded(self) -> list[Event]:
        if self._load_attempted:
            return self._events or []
        self._load_attempted = True
        try:
            trace = load_trace(self._path)
        except Exception:
            log.warning(
                "FileEventSource: failed to load trace %s; "
                "attribution will degrade to None",
                self._path,
                exc_info=True,
            )
            self._events = []
            return self._events
        self._events = list(trace.events)
        return self._events

    def capture(
        self, gpu_index: int, start_s: float, end_s: float
    ) -> list[Event]:
        """Return ALL events from the cached trace; window is ignored.

        See the class docstring for the rationale: a recorded trace's
        timestamps are on the profiler's own origin and do not align with
        the agent's monotonic clock, so the live-window intersection used
        by `MockEventSource` would silently filter out every event. The
        recorded trace itself defines the episode under attribution.

        ``gpu_index``, ``start_s``, ``end_s`` are accepted for interface
        compatibility but intentionally unused.
        """
        return self._ensure_loaded()


# ---------------------------------------------------------------------------
# Synthesizers: construct event lists the engine actually classifies.
# ---------------------------------------------------------------------------
#
# Each builder is anchored by `window_start_us` so tests / live demos that
# need events inside a moving window (the agent's monotonic clock is unrelated
# to whatever epoch the events were authored at) can call e.g.
# `build_sync_bound_events(window_start_us=int(start_s * 1_000_000))`.
#
# Shapes mirror the engine's own test fixtures (test_diagnose.py:
# test_sync_bound_verdict, test_dataloader_bound_verdict). Five kernel events
# clear the engine's `warmup_trace_guard` (kernel_events >= 5) and the
# 8 ms kernel duration clears `avg_kernel_dur >= 50µs` for sync_25.

_DEFAULT_KERNEL_DUR_US = 8_000
_DEFAULT_SLOT_US = 40_000
_DEFAULT_REPEATS = 5


def build_sync_bound_events(
    window_start_us: int = 0,
    *,
    repeats: int = _DEFAULT_REPEATS,
    kernel_dur_us: int = _DEFAULT_KERNEL_DUR_US,
    slot_us: int = _DEFAULT_SLOT_US,
) -> list[Event]:
    """Synthesize an event list the engine diagnoses as ``SYNC_BOUND``.

    `repeats` kernel events alternate with `aten::item` sync calls whose end
    time aligns with the start of each idle gap (within the engine's 500µs
    sync-attribution lookahead). The defaults clear all engine guards:

      - 5 kernel events  → warmup_trace_guard returns False.
      - 8 ms kernels     → avg_kernel_dur ≥ 50µs (kernel_launch guard).
      - 100% of idle is sync-attributed → sync_fraction = 1.0 ≥ 0.25.
      - Low util         → util < 0.70 (no healthy_70/85 rule fires).
    """
    if repeats < 1:
        raise ValueError(f"repeats must be >= 1, got {repeats}")
    if kernel_dur_us <= 0 or slot_us <= 0:
        raise ValueError("kernel_dur_us and slot_us must be > 0")
    if kernel_dur_us >= slot_us:
        raise ValueError("kernel_dur_us must be < slot_us (need idle gap)")

    out: list[Event] = []
    sync_anchor_us = 100  # aten::item lasts 100µs, ending exactly at idle start
    for i in range(repeats):
        base = window_start_us + i * slot_us
        out.append(
            Event(
                name="volta_sgemm",
                category="kernel",
                pid=1,
                tid=1,
                ts=base,
                dur=kernel_dur_us,
            )
        )
        # aten::item ends at base + kernel_dur_us — exactly when GPU goes idle.
        # The engine's _sync_attributed_idle catches sync ends within 500µs
        # before an idle interval starts, so this attributes the full idle gap.
        out.append(
            Event(
                name="aten::item",
                category="cpu_op",
                pid=0,
                tid=0,
                ts=base + kernel_dur_us - sync_anchor_us,
                dur=sync_anchor_us,
            )
        )
    return out


def build_dataloader_bound_events(
    window_start_us: int = 0,
    *,
    repeats: int = _DEFAULT_REPEATS,
    kernel_dur_us: int = _DEFAULT_KERNEL_DUR_US,
    slot_us: int = _DEFAULT_SLOT_US,
) -> list[Event]:
    """Synthesize an event list the engine diagnoses as ``DATALOADER_BOUND``.

    `repeats` kernel events alternate with `DataLoader__next_data` CPU events
    that exactly fill the idle gap between kernels. dl_share = 100% of idle
    is well above the 20% threshold for the dataloader_fallback rule.
    """
    if repeats < 1:
        raise ValueError(f"repeats must be >= 1, got {repeats}")
    if kernel_dur_us <= 0 or slot_us <= 0:
        raise ValueError("kernel_dur_us and slot_us must be > 0")
    if kernel_dur_us >= slot_us:
        raise ValueError("kernel_dur_us must be < slot_us (need idle gap)")

    out: list[Event] = []
    dl_dur = slot_us - kernel_dur_us
    for i in range(repeats):
        base = window_start_us + i * slot_us
        out.append(
            Event(
                name="volta_sgemm",
                category="kernel",
                pid=1,
                tid=1,
                ts=base,
                dur=kernel_dur_us,
            )
        )
        out.append(
            Event(
                name="DataLoader__next_data",
                category="cpu_op",
                pid=0,
                tid=0,
                ts=base + kernel_dur_us,
                dur=dl_dur,
            )
        )
    return out


__all__ = [
    "EventSource",
    "MockEventSource",
    "FileEventSource",
    "build_sync_bound_events",
    "build_dataloader_bound_events",
]
