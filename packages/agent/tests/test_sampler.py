"""Sampler tests: mock determinism and per-GPU error isolation."""

from __future__ import annotations

import itertools

import pytest

from gpu_doctor_agent.sampler import (
    MockNvmlSampler,
    Sample,
    Sampler,
)


class _FakeClock:
    def __init__(self) -> None:
        self._t = 0.0

    def __call__(self) -> float:
        return self._t

    def advance(self, dt: float) -> None:
        self._t += dt


def test_mock_returns_one_sample_per_gpu() -> None:
    clock = _FakeClock()
    s = MockNvmlSampler(num_gpus=4, clock=clock)
    assert s.gpu_count() == 4
    samples = s.sample()
    assert len(samples) == 4
    assert [x.gpu_index for x in samples] == [0, 1, 2, 3]


def test_mock_holds_last_value_after_pattern_exhausted() -> None:
    clock = _FakeClock()
    pattern = [0.1, 0.5, 0.9]
    s = MockNvmlSampler(num_gpus=1, util_pattern=pattern, clock=clock, seed=42)
    # Playback is hold-last (clamp): the pattern plays out, then the FINAL
    # value (0.9) is repeated forever. This makes scenario outcomes terminal
    # so the live binary's event counts don't depend on how many ticks ran.
    seen = []
    for _ in range(7):
        seen.append(s.sample()[0].util_pct)
    assert seen == [0.1, 0.5, 0.9, 0.9, 0.9, 0.9, 0.9]


def test_mock_two_runs_same_seed_match() -> None:
    pattern = [0.1, 0.3, 0.6, 0.8]
    # Inject identical fake clocks so timestamps are deterministic across runs.
    a = MockNvmlSampler(num_gpus=2, util_pattern=pattern, clock=_FakeClock(), seed=7)
    b = MockNvmlSampler(num_gpus=2, util_pattern=pattern, clock=_FakeClock(), seed=7)
    for _ in range(5):
        assert a.sample() == b.sample()


def test_mock_rejects_util_outside_unit_interval() -> None:
    with pytest.raises(ValueError):
        MockNvmlSampler(num_gpus=1, util_pattern=[1.5])
    with pytest.raises(ValueError):
        MockNvmlSampler(num_gpus=1, util_pattern=[-0.1])


def test_mock_rejects_zero_gpus() -> None:
    with pytest.raises(ValueError):
        MockNvmlSampler(num_gpus=0)


class _FlakyTwoGpuSampler(Sampler):
    """Per-GPU isolation harness: GPU 0 succeeds, GPU 1 raises.

    Mirrors the NvmlSampler contract: per-GPU read failures must NOT raise out
    of `sample()` — they must materialize as a Sample with sentinel Nones.
    """

    def __init__(self) -> None:
        self._ticks = itertools.count()

    def gpu_count(self) -> int:
        return 2

    def sample(self) -> list[Sample]:
        now = float(next(self._ticks))
        out: list[Sample] = []
        for i in range(2):
            try:
                if i == 1:
                    raise RuntimeError("simulated NVML failure on GPU 1")
                out.append(
                    Sample(
                        timestamp_s=now,
                        gpu_index=i,
                        util_pct=0.7,
                        mem_used_mb=1000.0,
                        mem_total_mb=40000.0,
                        sm_clock_mhz=1500,
                        power_w=250.0,
                    )
                )
            except Exception:
                out.append(
                    Sample(
                        timestamp_s=now,
                        gpu_index=i,
                        util_pct=None,
                        mem_used_mb=None,
                        mem_total_mb=None,
                        sm_clock_mhz=None,
                        power_w=None,
                    )
                )
        return out


def test_per_gpu_error_isolation_contract() -> None:
    """If one GPU's read explodes, the other GPU still produces a sample."""
    sampler = _FlakyTwoGpuSampler()
    samples = sampler.sample()
    assert len(samples) == 2
    by_idx = {s.gpu_index: s for s in samples}
    assert by_idx[0].util_pct == 0.7  # healthy GPU still reported
    assert by_idx[1].util_pct is None  # sick GPU degraded, not raised
    assert by_idx[1].mem_used_mb is None
