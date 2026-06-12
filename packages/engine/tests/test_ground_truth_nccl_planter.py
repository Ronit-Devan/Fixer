"""NCCL synthetic planter runs without CUDA."""

from __future__ import annotations

from gpu_doctor_engine import Verdict

from accuracy.ground_truth import plant_nccl_bound


def test_plant_nccl_bound_synthetic_matches_without_cuda() -> None:
    result = plant_nccl_bound()
    assert result.match is True
    assert result.actual == Verdict.NCCL_BOUND
    assert result.expected == Verdict.NCCL_BOUND
    assert result.trace_path is not None
    assert "nccl_share=" in result.decisive_metric
