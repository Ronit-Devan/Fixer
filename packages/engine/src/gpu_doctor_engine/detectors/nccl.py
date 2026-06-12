"""NCCL_BOUND idle-window detector."""

from __future__ import annotations

from gpu_doctor_engine.detectors.attribution import IdleShareMeasurement, overlap_idle_time
from gpu_doctor_engine.detectors.confidence import confidence_from_share
from gpu_doctor_engine.types import Diagnosis, Event, Verdict

# Collective ops the engine attributes to NCCL (case-insensitive name match).
NCCL_PATTERNS: tuple[str, ...] = (
    "nccl",
    "AllReduce",
    "AllGather",
    "ReduceScatter",
    "Broadcast",
    "c10d::",
)

NCCL_IDLE_SHARE_THRESHOLD: float = 0.30


class NcclBoundDetector:
    """Fires when NCCL collective CPU activity covers >= 30% of GPU idle time."""

    PATTERNS = NCCL_PATTERNS
    THRESHOLD = NCCL_IDLE_SHARE_THRESHOLD

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
        nccl_us: int,
    ) -> Diagnosis:
        nccl_share = nccl_us / max(idle_us, 1)
        return Diagnosis(
            verdict=Verdict.NCCL_BOUND,
            confidence=confidence_from_share(nccl_share),
            summary=(
                f"GPU is {util:.0%} utilized. The dominant cause is collective "
                f"communication: {nccl_us / 1000:.0f}ms in NCCL operations "
                f"({nccl_us / max(idle_us, 1):.0%} of GPU idle time)."
            ),
            evidence=[
                f"GPU utilization: {util:.0%}",
                f"Total GPU idle: {idle_us / 1000:.0f}ms",
                f"NCCL time during idle: {nccl_us / 1000:.0f}ms",
            ],
            recommended_actions=[
                "Check for stragglers (one rank slower than others).",
                "Consider gradient accumulation to reduce all-reduce frequency.",
                "Verify NCCL is using the fastest interconnect (NVLink, not PCIe).",
                "Try ZeRO-3 or FSDP communication overlap if not already enabled.",
            ],
            metrics={
                "gpu_util": util,
                "nccl_us": nccl_us,
                "idle_us": idle_us,
                "nccl_share": nccl_share,
            },
        )
