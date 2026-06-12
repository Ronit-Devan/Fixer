"""CHECKPOINT_BOUND idle-window detector."""

from __future__ import annotations

from gpu_doctor_engine.detectors.attribution import IdleShareMeasurement, overlap_idle_time
from gpu_doctor_engine.detectors.confidence import confidence_from_share
from gpu_doctor_engine.types import Diagnosis, Event, Verdict

# Name patterns for checkpoint overlap attribution (idle-window share).
CHECKPOINT_PATTERNS: tuple[str, ...] = (
    "torch.save",
    "save_checkpoint",
    "state_dict",
    "aten::copy_",
)

CHECKPOINT_IDLE_SHARE_THRESHOLD: float = 0.25


class CheckpointBoundDetector:
    """Fires when checkpoint CPU activity covers >= 25% of GPU idle time."""

    PATTERNS = CHECKPOINT_PATTERNS
    THRESHOLD = CHECKPOINT_IDLE_SHARE_THRESHOLD

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

    @classmethod
    def build_diagnosis(
        cls,
        util: float,
        idle_us: int,
        checkpoint_us: int,
    ) -> Diagnosis:
        ckpt_share = checkpoint_us / max(idle_us, 1)
        return Diagnosis(
            verdict=Verdict.CHECKPOINT_BOUND,
            confidence=confidence_from_share(ckpt_share),
            summary=(
                f"GPU is {util:.0%} utilized. Synchronous checkpointing dominates: "
                f"{checkpoint_us / 1000:.0f}ms ({ckpt_share:.0%}) of GPU idle time "
                f"overlaps with torch.save / checkpoint operations."
            ),
            evidence=[
                f"GPU utilization: {util:.0%}",
                f"Total GPU idle: {idle_us / 1000:.0f}ms",
                f"Checkpoint time during idle: {checkpoint_us / 1000:.0f}ms",
            ],
            recommended_actions=[
                "Use asynchronous checkpointing (DeepSpeed Universal Checkpointing or DataStates-LLM).",
                "Reduce checkpoint frequency if your training is stable.",
                "Save to local NVMe first, then async-upload to object storage.",
                "Use torch.distributed.checkpoint for multi-GPU training.",
            ],
            metrics={
                "gpu_util": util,
                "checkpoint_us": checkpoint_us,
                "idle_us": idle_us,
                "ckpt_share": ckpt_share,
            },
        )
