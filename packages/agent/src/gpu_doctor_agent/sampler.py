"""Per-GPU sampling abstraction.

Two implementations:
  - NvmlSampler: real NVML reads (Linux/NVIDIA hosts only).
  - MockNvmlSampler: deterministic, seedable, scripted — for tests and
    dev machines without a GPU.

Both produce a flat list[Sample] per tick (one per selected GPU).
A None util/mem/clock/power value is the sentinel for a transient read
failure on that GPU at that tick — the loop continues; the detector and
buffer must tolerate Nones.
"""

from __future__ import annotations

import logging
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Sequence

from gpu_doctor_agent.config import AgentConfig

log = logging.getLogger(__name__)


# Named utilization patterns the MockNvmlSampler plays back through, used by
# the CLI's --scenario flag to drive the FSM end-to-end without code edits.
# Each pattern is a sequence of fractional utilizations in [0, 1].
#
# Playback semantics are hold-last (clamp), not cyclic: once the pattern is
# exhausted the sampler repeats the FINAL element forever. This makes scenario
# outcomes terminal and --max-iters-independent — e.g. "idle" sustains idle, so
# the detector confirms exactly once regardless of how long the loop runs.
# Each pattern below is therefore chosen so its LAST value reflects the
# scenario's intended steady state (idle stays low; flapping & recovering end
# high so the system settles HEALTHY; busy is constant).
SCENARIOS: dict[str, list[float]] = {
    # Stays above the default idle threshold (0.20). Detector never trips.
    "busy": [0.85],
    # Brief warm-up, then sustained idle (terminal=0.05). Detector confirms once.
    "idle": [0.85, 0.85, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
    # Alternating dips that never sustain. Ends HIGH so playback settles healthy
    # — no late dip can re-arm the detector. Detector emits 0 events.
    "flapping": [0.05, 0.85, 0.05, 0.85, 0.05, 0.85],
    # Idle long enough to confirm, then recovers past the hysteresis threshold
    # and stays there (terminal=0.85). Exactly one event, then settles healthy.
    "recovering": [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.85, 0.85],
}


class NvmlUnavailable(RuntimeError):
    """pynvml is missing or NVML failed to initialize on this host."""


@dataclass(frozen=True)
class Sample:
    timestamp_s: float
    gpu_index: int
    util_pct: float | None  # 0..1, or None if read failed
    mem_used_mb: float | None
    mem_total_mb: float | None
    sm_clock_mhz: int | None
    power_w: float | None


class Sampler(ABC):
    @abstractmethod
    def sample(self) -> list[Sample]:
        """Take one sample per selected GPU. Never raises for a single-GPU error."""

    @abstractmethod
    def gpu_count(self) -> int:
        """Number of GPUs this sampler will report on per tick."""


# ---------------------------------------------------------------------------
# NVML
# ---------------------------------------------------------------------------


class NvmlSampler(Sampler):
    """Real-NVML sampler. Lazy-inits NVML on first use."""

    def __init__(
        self,
        gpu_indices: Sequence[int] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._requested_indices = (
            tuple(gpu_indices) if gpu_indices is not None else None
        )
        self._clock = clock
        self._initialized = False
        self._handles: list[tuple[int, object]] = []  # [(gpu_index, handle), ...]
        self._pynvml: object | None = None

    def _ensure_init(self) -> None:
        if self._initialized:
            return
        try:
            import pynvml  # type: ignore[import-not-found]
        except ImportError as e:
            raise NvmlUnavailable(f"pynvml not installed: {e}") from e
        try:
            pynvml.nvmlInit()
        except Exception as e:  # NVMLError subclasses live in pynvml
            raise NvmlUnavailable(f"nvmlInit failed: {e}") from e

        try:
            total = pynvml.nvmlDeviceGetCount()
        except Exception as e:
            raise NvmlUnavailable(f"nvmlDeviceGetCount failed: {e}") from e

        if self._requested_indices is None:
            indices: list[int] = list(range(total))
        else:
            indices = [i for i in self._requested_indices if 0 <= i < total]
            if not indices:
                raise NvmlUnavailable(
                    f"None of requested GPU indices {self._requested_indices} exist "
                    f"(NVML reports {total} GPUs)"
                )

        handles: list[tuple[int, object]] = []
        for i in indices:
            try:
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
            except Exception as e:
                # Skip the GPU we can't open, but keep going — partial init beats none.
                log.warning("NVML: failed to open GPU %d: %s", i, e)
                continue
            handles.append((i, h))

        if not handles:
            raise NvmlUnavailable("No GPU handles could be opened")

        self._pynvml = pynvml
        self._handles = handles
        self._initialized = True

    def gpu_count(self) -> int:
        self._ensure_init()
        return len(self._handles)

    def sample(self) -> list[Sample]:
        self._ensure_init()
        assert self._pynvml is not None
        pynvml = self._pynvml
        now = self._clock()
        out: list[Sample] = []
        for gpu_index, handle in self._handles:
            # Each GPU isolated: any failure on one becomes a Sample-with-Nones,
            # never a raise. One sick GPU must not kill the loop.
            try:
                out.append(_read_one_nvml(pynvml, handle, gpu_index, now))
            except Exception as e:  # defensive: _read_one_nvml already catches
                log.warning("NVML: unexpected failure sampling GPU %d: %s", gpu_index, e)
                out.append(
                    Sample(
                        timestamp_s=now,
                        gpu_index=gpu_index,
                        util_pct=None,
                        mem_used_mb=None,
                        mem_total_mb=None,
                        sm_clock_mhz=None,
                        power_w=None,
                    )
                )
        return out


def _read_one_nvml(pynvml: object, handle: object, gpu_index: int, now: float) -> Sample:
    """All per-GPU NVML calls, each individually guarded.

    A missing optional metric (clock, power) becomes None rather than failing
    the whole sample.
    """
    util_pct: float | None
    try:
        u = pynvml.nvmlDeviceGetUtilizationRates(handle)  # type: ignore[attr-defined]
        util_pct = float(u.gpu) / 100.0
    except Exception as e:
        log.warning("NVML util read failed (GPU %d): %s", gpu_index, e)
        util_pct = None

    mem_used_mb: float | None
    mem_total_mb: float | None
    try:
        m = pynvml.nvmlDeviceGetMemoryInfo(handle)  # type: ignore[attr-defined]
        mem_used_mb = float(m.used) / (1024 * 1024)
        mem_total_mb = float(m.total) / (1024 * 1024)
    except Exception as e:
        log.warning("NVML mem read failed (GPU %d): %s", gpu_index, e)
        mem_used_mb = None
        mem_total_mb = None

    sm_clock_mhz: int | None
    try:
        sm_clock_mhz = int(
            pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)  # type: ignore[attr-defined]
        )
    except Exception:
        sm_clock_mhz = None  # optional

    power_w: float | None
    try:
        power_w = float(pynvml.nvmlDeviceGetPowerUsage(handle)) / 1000.0  # type: ignore[attr-defined]
    except Exception:
        power_w = None  # optional

    return Sample(
        timestamp_s=now,
        gpu_index=gpu_index,
        util_pct=util_pct,
        mem_used_mb=mem_used_mb,
        mem_total_mb=mem_total_mb,
        sm_clock_mhz=sm_clock_mhz,
        power_w=power_w,
    )


# ---------------------------------------------------------------------------
# Mock
# ---------------------------------------------------------------------------


class MockNvmlSampler(Sampler):
    """Deterministic, scripted sampler. Drop-in for NvmlSampler in tests and dev.

    `util_pattern` is a sequence of fractional utilizations the sampler plays
    back across ticks. Playback is hold-last (clamp): once the pattern is
    exhausted the FINAL element is repeated forever, so scenario outcomes are
    terminal and the live binary's event counts don't depend on how long the
    loop ran. Each GPU is offset by its index so they don't move in lockstep
    until both have clamped to the last element.
    """

    def __init__(
        self,
        num_gpus: int = 1,
        util_pattern: Sequence[float] | None = None,
        clock: Callable[[], float] = time.monotonic,
        seed: int = 0,
        mem_total_mb: float = 40000.0,
    ) -> None:
        if num_gpus <= 0:
            raise ValueError(f"num_gpus must be > 0, got {num_gpus}")
        self._num_gpus = num_gpus
        self._pattern: tuple[float, ...] = (
            tuple(util_pattern) if util_pattern else (0.85,)
        )
        for u in self._pattern:
            if not (0.0 <= u <= 1.0):
                raise ValueError(f"util_pattern values must be in [0,1], got {u}")
        self._clock = clock
        self._rng = random.Random(seed)
        self._tick = 0
        self._mem_total_mb = mem_total_mb

    @classmethod
    def from_scenario(
        cls,
        name: str,
        gpu_count: int = 1,
        seed: int = 0,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> "MockNvmlSampler":
        if name not in SCENARIOS:
            raise ValueError(
                f"unknown scenario {name!r}; valid: {sorted(SCENARIOS)}"
            )
        return cls(
            num_gpus=gpu_count,
            util_pattern=SCENARIOS[name],
            clock=clock,
            seed=seed,
        )

    def gpu_count(self) -> int:
        return self._num_gpus

    def sample(self) -> list[Sample]:
        now = self._clock()
        out: list[Sample] = []
        last = len(self._pattern) - 1
        for g in range(self._num_gpus):
            # Hold-last (clamp): past the pattern's end, repeat pattern[-1].
            # Wrapping here would make the binary's event counts depend on
            # --max-iters (a late-cycle dip can re-arm the detector); clamping
            # makes the scenario's terminal value the scenario's verdict.
            util = self._pattern[min(self._tick + g, last)]
            # Mem walk: small deterministic-ish jitter that does NOT touch the
            # tick-driven utilization sequence (seed reused across ticks is OK
            # because the RNG only feeds non-determinism into ancillary fields).
            mem_used = self._mem_total_mb * (0.5 + 0.1 * self._rng.random())
            out.append(
                Sample(
                    timestamp_s=now,
                    gpu_index=g,
                    util_pct=util,
                    mem_used_mb=mem_used,
                    mem_total_mb=self._mem_total_mb,
                    sm_clock_mhz=1500,
                    power_w=250.0,
                )
            )
        self._tick += 1
        return out


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_sampler(
    config: AgentConfig,
    mock: bool,
    *,
    clock: Callable[[], float] = time.monotonic,
    scenario: str = "busy",
) -> Sampler:
    """Return the configured sampler.

    If mock is True, always returns a MockNvmlSampler built from `scenario`.
    Otherwise tries NvmlSampler; if NVML is unavailable, logs a clear warning
    and falls back to a scenario-driven MockNvmlSampler so dev environments
    (and the NVML fallback path) still produce realistic-looking util traces.
    """
    if mock:
        return MockNvmlSampler.from_scenario(scenario, gpu_count=1, clock=clock)
    try:
        sampler = NvmlSampler(gpu_indices=config.gpu_indices, clock=clock)
        sampler.gpu_count()  # forces init; raises NvmlUnavailable if so
        return sampler
    except NvmlUnavailable as e:
        log.warning(
            "NVML unavailable (%s); falling back to MockNvmlSampler "
            "(scenario=%s). Pass --mock to silence this warning.",
            e,
            scenario,
        )
        return MockNvmlSampler.from_scenario(scenario, gpu_count=1, clock=clock)
