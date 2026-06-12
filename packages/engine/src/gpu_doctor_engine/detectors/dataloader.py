"""DATALOADER_BOUND idle-window detector."""

from __future__ import annotations

from gpu_doctor_engine.detectors.attribution import IdleShareMeasurement, overlap_idle_time
from gpu_doctor_engine.types import Event

DATALOADER_PATTERNS: tuple[str, ...] = (
    "DataLoader",
    "enumerate(DataLoader)",
    "_MultiProcessingDataLoaderIter",
    "_SingleProcessDataLoaderIter",
    "_next_data",
    "_get_iterator",
    "fetch",
    "collate",
)

DATALOADER_IDLE_SHARE_THRESHOLD: float = 0.20


class DataloaderBoundDetector:
    """Fires when DataLoader CPU activity covers >= 20% of GPU idle time."""

    PATTERNS = DATALOADER_PATTERNS
    THRESHOLD = DATALOADER_IDLE_SHARE_THRESHOLD

    @classmethod
    def measure(
        cls,
        idle_intervals: list[tuple[int, int]],
        events: list[Event],
        idle_us: int,
    ) -> IdleShareMeasurement:
        overlap_us = overlap_idle_time(idle_intervals, events, cls.PATTERNS)
        share = overlap_us / max(idle_us, 1)
        return IdleShareMeasurement(
            overlap_us=overlap_us, idle_us=idle_us, share=share
        )

    @classmethod
    def fired(cls, measurement: IdleShareMeasurement) -> bool:
        return measurement.fired_at(cls.THRESHOLD)
