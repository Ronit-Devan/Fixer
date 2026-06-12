"""Core data types."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Verdict(str, Enum):
    HEALTHY = "healthy"
    DATALOADER_BOUND = "dataloader_bound"
    PCIE_BOUND = "pcie_bound"
    KERNEL_LAUNCH_BOUND = "kernel_launch_bound"
    NCCL_BOUND = "nccl_bound"
    CHECKPOINT_BOUND = "checkpoint_bound"
    SYNC_BOUND = "sync_bound"
    UNKNOWN = "unknown"


@dataclass
class Event:
    name: str
    category: str
    pid: int
    tid: int
    ts: int
    dur: int
    args: dict = field(default_factory=dict)


@dataclass
class Trace:
    events: list[Event]
    duration_us: int
    gpu_kernel_time_us: int
    cpu_time_us: int

    @property
    def gpu_utilization(self) -> float:
        if self.duration_us == 0:
            return 0.0
        return self.gpu_kernel_time_us / self.duration_us


@dataclass
class Diagnosis:
    verdict: Verdict
    confidence: float | None
    summary: str
    evidence: list[str]
    recommended_actions: list[str]
    metrics: dict
