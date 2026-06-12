"""Modular bottleneck detectors (idle-window attribution)."""

from gpu_doctor_engine.detectors.attribution import (
    IdleShareMeasurement,
    overlap_idle_time,
)
from gpu_doctor_engine.detectors.confidence import confidence_from_share
from gpu_doctor_engine.detectors.checkpoint import (
    CHECKPOINT_IDLE_SHARE_THRESHOLD,
    CHECKPOINT_PATTERNS,
    CheckpointBoundDetector,
)
from gpu_doctor_engine.detectors.dataloader import (
    DATALOADER_IDLE_SHARE_THRESHOLD,
    DATALOADER_PATTERNS,
    DataloaderBoundDetector,
)
from gpu_doctor_engine.detectors.nccl import (
    NCCL_IDLE_SHARE_THRESHOLD,
    NCCL_PATTERNS,
    NcclBoundDetector,
)

__all__ = [
    "CHECKPOINT_IDLE_SHARE_THRESHOLD",
    "CHECKPOINT_PATTERNS",
    "CheckpointBoundDetector",
    "DATALOADER_IDLE_SHARE_THRESHOLD",
    "DATALOADER_PATTERNS",
    "DataloaderBoundDetector",
    "IdleShareMeasurement",
    "NCCL_IDLE_SHARE_THRESHOLD",
    "NCCL_PATTERNS",
    "NcclBoundDetector",
    "confidence_from_share",
    "overlap_idle_time",
]
