"""Recalibrated confidence from dominance share (idle overlap or symptom fraction)."""

from __future__ import annotations

# Tier boundaries: share of idle (or tiny_kernel_fraction) -> confidence.
_CONFIDENCE_TIERS: tuple[tuple[float, float], ...] = (
    (0.90, 0.95),
    (0.75, 0.90),
    (0.60, 0.82),
    (0.40, 0.72),
    (0.25, 0.60),
)


def confidence_from_share(share: float) -> float:
    """Map how dominant a cause is (0–1) to a calibrated confidence score.

    Used for all BOUND verdicts: idle-window shares (DataLoader, NCCL, PCIe
    idle overlap, checkpoint, sync) and ``tiny_kernel_ratio`` for
    KERNEL_LAUNCH_BOUND.
    """
    for threshold, confidence in _CONFIDENCE_TIERS:
        if share >= threshold:
            return confidence
    # Below 0.25 but still fired (e.g. DataLoader at >=0.20 idle share).
    return 0.60
