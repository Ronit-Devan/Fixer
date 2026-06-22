"""GPU sampling for the live inference monitor.

Three backends, tried in order so the app *just works* on any machine:

  1. ``pynvml`` (NVIDIA Management Library); richest signal, no subprocess.
  2. ``nvidia-smi`` CSV query; always present with the driver; the robust
     fallback when the ``pynvml`` / ``nvidia-ml-py`` wheel won't import
     (common on fresh Windows boxes).
  3. ``MockGpuSampler``; deterministic synthetic data so the dashboard runs
     on a laptop with no GPU at all (demos, dev, CI).

A failed read of one metric on one GPU degrades that field to ``None`` rather
than raising; one sick sensor must never take the monitor down. This mirrors
the isolation discipline in ``gpu_doctor_agent.sampler``.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class GpuReading:
    """One GPU's state at one tick. ``None`` means that sensor failed to read."""

    timestamp_s: float
    index: int
    name: str
    util_pct: float | None  # 0..100
    mem_used_mb: float | None
    mem_total_mb: float | None
    power_w: float | None
    power_limit_w: float | None
    sm_clock_mhz: int | None
    sm_clock_max_mhz: int | None
    temp_c: float | None

    @property
    def mem_used_ratio(self) -> float | None:
        if self.mem_used_mb is None or not self.mem_total_mb:
            return None
        return self.mem_used_mb / self.mem_total_mb

    @property
    def power_ratio(self) -> float | None:
        if self.power_w is None or not self.power_limit_w:
            return None
        return self.power_w / self.power_limit_w

    @property
    def clock_ratio(self) -> float | None:
        if self.sm_clock_mhz is None or not self.sm_clock_max_mhz:
            return None
        return self.sm_clock_mhz / self.sm_clock_max_mhz


class GpuSampler(ABC):
    backend: str = "abstract"

    @abstractmethod
    def read(self) -> list[GpuReading]:
        """One reading per GPU. Never raises for a per-GPU failure."""

    @abstractmethod
    def gpu_count(self) -> int: ...


# ---------------------------------------------------------------------------
# pynvml
# ---------------------------------------------------------------------------


class NvmlGpuSampler(GpuSampler):
    backend = "nvml"

    def __init__(self, indices: Sequence[int] | None = None) -> None:
        import pynvml  # raises ImportError if unavailable

        pynvml.nvmlInit()
        self._pynvml = pynvml
        total = pynvml.nvmlDeviceGetCount()
        wanted = list(range(total)) if indices is None else list(indices)
        self._handles: list[tuple[int, object]] = []
        for i in wanted:
            if 0 <= i < total:
                self._handles.append((i, pynvml.nvmlDeviceGetHandleByIndex(i)))
        if not self._handles:
            raise RuntimeError("NVML initialised but no GPU handles opened")

    def gpu_count(self) -> int:
        return len(self._handles)

    def read(self) -> list[GpuReading]:
        nv = self._pynvml
        now = time.time()
        out: list[GpuReading] = []
        for index, h in self._handles:
            out.append(self._read_one(nv, h, index, now))
        return out

    def _read_one(self, nv, h, index: int, now: float) -> GpuReading:
        def _try(fn, default=None):
            try:
                return fn()
            except Exception:  # noqa: BLE001; any NVML hiccup degrades to None
                return default

        name = _try(lambda: _decode(nv.nvmlDeviceGetName(h)), f"GPU {index}")
        util = _try(lambda: float(nv.nvmlDeviceGetUtilizationRates(h).gpu))
        mem = _try(lambda: nv.nvmlDeviceGetMemoryInfo(h))
        mem_used = float(mem.used) / 1024 / 1024 if mem else None
        mem_total = float(mem.total) / 1024 / 1024 if mem else None
        power = _try(lambda: float(nv.nvmlDeviceGetPowerUsage(h)) / 1000.0)
        plimit = _try(
            lambda: float(nv.nvmlDeviceGetEnforcedPowerLimit(h)) / 1000.0
        )
        sm = _try(lambda: int(nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_SM)))
        sm_max = _try(
            lambda: int(nv.nvmlDeviceGetMaxClockInfo(h, nv.NVML_CLOCK_SM))
        )
        temp = _try(
            lambda: float(nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU))
        )
        return GpuReading(
            timestamp_s=now,
            index=index,
            name=name or f"GPU {index}",
            util_pct=util,
            mem_used_mb=mem_used,
            mem_total_mb=mem_total,
            power_w=power,
            power_limit_w=plimit,
            sm_clock_mhz=sm,
            sm_clock_max_mhz=sm_max,
            temp_c=temp,
        )


def _decode(v) -> str:
    return v.decode() if isinstance(v, bytes) else str(v)


# ---------------------------------------------------------------------------
# nvidia-smi fallback
# ---------------------------------------------------------------------------

_SMI_FIELDS = (
    "index",
    "name",
    "utilization.gpu",
    "memory.used",
    "memory.total",
    "power.draw",
    "power.limit",
    "clocks.sm",
    "clocks.max.sm",
    "temperature.gpu",
)


class NvidiaSmiGpuSampler(GpuSampler):
    """Parse ``nvidia-smi --query-gpu`` CSV. Works wherever the driver is."""

    backend = "nvidia-smi"

    def __init__(self, indices: Sequence[int] | None = None) -> None:
        self._smi = shutil.which("nvidia-smi")
        if not self._smi:
            raise RuntimeError("nvidia-smi not found on PATH")
        self._indices = set(indices) if indices is not None else None
        # One validating read at construction; cache the GPU count from it so
        # gpu_count() never spawns another subprocess (the physical GPU count
        # does not change at runtime). Avoids 2 extra nvidia-smi forks at startup.
        first = self.read()
        if not first:
            raise RuntimeError("nvidia-smi returned no GPUs")
        self._count = len(first)

    def gpu_count(self) -> int:
        return self._count

    def read(self) -> list[GpuReading]:
        now = time.time()
        try:
            proc = subprocess.run(
                [
                    self._smi,
                    f"--query-gpu={','.join(_SMI_FIELDS)}",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
        except Exception as e:  # noqa: BLE001
            log.warning("nvidia-smi invocation failed: %s", e)
            return []
        out: list[GpuReading] = []
        for line in proc.stdout.strip().splitlines():
            cells = [c.strip() for c in line.split(",")]
            if len(cells) < len(_SMI_FIELDS):
                continue
            idx = _to_int(cells[0])
            if idx is None:
                continue
            if self._indices is not None and idx not in self._indices:
                continue
            out.append(
                GpuReading(
                    timestamp_s=now,
                    index=idx,
                    name=cells[1] or f"GPU {idx}",
                    util_pct=_to_float(cells[2]),
                    mem_used_mb=_to_float(cells[3]),
                    mem_total_mb=_to_float(cells[4]),
                    power_w=_to_float(cells[5]),
                    power_limit_w=_to_float(cells[6]),
                    sm_clock_mhz=_to_int(cells[7]),
                    sm_clock_max_mhz=_to_int(cells[8]),
                    temp_c=_to_float(cells[9]),
                )
            )
        return out


def _to_float(s: str) -> float | None:
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def _to_int(s: str) -> int | None:
    f = _to_float(s)
    return int(f) if f is not None else None


# ---------------------------------------------------------------------------
# Mock
# ---------------------------------------------------------------------------


class MockGpuSampler(GpuSampler):
    """Deterministic synthetic GPU. Drives a believable inference timeline for
    demos and tests when no real GPU is present.

    The util/mem pattern is supplied by the caller (see ``demo`` module); by
    default the GPU sits mostly idle with occasional bursts, which is the
    realistic shape for a single-box inference server.
    """

    backend = "mock"

    def __init__(
        self,
        name: str = "Mock RTX PRO 4000 Blackwell SFF",
        mem_total_mb: float = 24564.0,
        power_limit_w: float = 70.0,
        sm_clock_max_mhz: int = 2520,
    ) -> None:
        self._name = name
        self._mem_total = mem_total_mb
        self._plimit = power_limit_w
        self._sm_max = sm_clock_max_mhz
        # Mutated by the demo driver each tick; defaults to idle.
        self.util_pct: float = 3.0
        self.mem_used_mb: float = mem_total_mb * 0.45
        self.power_w: float = power_limit_w * 0.12
        self.sm_clock_mhz: int = int(sm_clock_max_mhz * 0.3)
        self.temp_c: float = 38.0

    def gpu_count(self) -> int:
        return 1

    def read(self) -> list[GpuReading]:
        return [
            GpuReading(
                timestamp_s=time.time(),
                index=0,
                name=self._name,
                util_pct=self.util_pct,
                mem_used_mb=self.mem_used_mb,
                mem_total_mb=self._mem_total,
                power_w=self.power_w,
                power_limit_w=self._plimit,
                sm_clock_mhz=self.sm_clock_mhz,
                sm_clock_max_mhz=self._sm_max,
                temp_c=self.temp_c,
            )
        ]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_gpu_sampler(
    *, force_mock: bool = False, indices: Sequence[int] | None = None
) -> GpuSampler:
    """Return the best available sampler, with a clear log line for which won."""
    if force_mock:
        log.info("GPU backend: mock (forced)")
        return MockGpuSampler()
    for ctor, label in (
        (lambda: NvmlGpuSampler(indices), "nvml"),
        (lambda: NvidiaSmiGpuSampler(indices), "nvidia-smi"),
    ):
        try:
            sampler = ctor()
            log.info("GPU backend: %s (%d GPU(s))", label, sampler.gpu_count())
            return sampler
        except Exception as e:  # noqa: BLE001
            log.info("GPU backend %s unavailable: %s", label, e)
    log.warning("No real GPU backend available; falling back to mock data.")
    return MockGpuSampler()
