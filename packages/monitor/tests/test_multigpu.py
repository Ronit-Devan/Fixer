"""Multi-GPU monitoring: every card is tracked, diagnosed, and costed.

Proves the audit's #1 usability fix — a DGX/multi-GPU box is no longer 7/8
invisible and idle cost is no longer under-reported by the GPU count.
"""

from __future__ import annotations

import time

from et_monitor.gpu import GpuReading, GpuSampler
from et_monitor.state import Monitor, MonitorConfig
from et_monitor.types import Verdict


class MultiMock(GpuSampler):
    backend = "multimock"

    def __init__(self, utils: list[float], mem_ratio: float = 0.5) -> None:
        self._utils = utils
        self._mem_ratio = mem_ratio

    def gpu_count(self) -> int:
        return len(self._utils)

    def read(self) -> list[GpuReading]:
        now = time.time()
        return [
            GpuReading(
                timestamp_s=now, index=i, name=f"GPU{i}", util_pct=u,
                mem_used_mb=24000.0 * self._mem_ratio, mem_total_mb=24000.0,
                power_w=40.0, power_limit_w=70.0, sm_clock_mhz=2400,
                sm_clock_max_mhz=2520, temp_c=50.0,
            )
            for i, u in enumerate(self._utils)
        ]


def test_all_gpus_are_tracked_and_diagnosed():
    # 3 idle cards, 1 busy card.
    mon = Monitor(MultiMock([2.0, 2.0, 90.0, 2.0]), None, MonitorConfig(gpu_hourly_usd=1.0))
    for _ in range(10):
        mon.tick()

    assert mon.gpus() == [0, 1, 2, 3]
    snap = mon.snapshot()
    assert snap["gpu_count"] == 4
    assert len(snap["gpus"]) == 4
    assert snap["gpus"][0]["is_primary"] is True

    # Per-GPU diagnosis: the busy card is not idle, the others are.
    d = mon.diagnosis_all()
    assert d[0].verdict == Verdict.IDLE_NO_REQUESTS
    assert d[2].verdict != Verdict.IDLE_NO_REQUESTS


def test_fleet_idle_cost_sums_across_gpus_not_just_card0():
    # 3 of 4 cards idle: a single-GPU monitor would report 1 card's idle cost;
    # the fleet correctly reports 3 cards' worth (the under-reporting fix).
    mon = Monitor(MultiMock([2.0, 2.0, 90.0, 2.0]), None, MonitorConfig(gpu_hourly_usd=1.0))
    for _ in range(10):
        mon.tick()
    session = mon.snapshot()["session"]
    # 3 idle GPU-seconds-streams * 10s = 30 idle GPU-seconds of 40 total.
    assert session["idle_seconds"] == 30.0
    assert session["idle_fraction"] == 0.75
    assert session["wasted_usd_so_far"] == round(30 / 3600, 2)  # 3x a single card


def test_report_breaks_down_per_gpu():
    mon = Monitor(MultiMock([2.0, 90.0]), None, MonitorConfig(gpu_hourly_usd=2.0))
    for _ in range(5):
        mon.tick()
    rep = mon.report()
    assert rep["gpu_count"] == 2
    assert len(rep["gpus"]) == 2
    assert {g["index"] for g in rep["gpus"]} == {0, 1}
    # The idle card carries all the idle fraction; the busy card carries none.
    by_idx = {g["index"]: g for g in rep["gpus"]}
    assert by_idx[0]["idle_fraction"] == 1.0
    assert by_idx[1]["idle_fraction"] == 0.0


def test_report_wasted_matches_snapshot_for_many_small_gpus():
    # 10 small idle cards: each card's wasted-$ rounds to $0.00, but the FLEET
    # total must not — report() must sum raw (matching snapshot), not sum rounded.
    mon = Monitor(MultiMock([2.0] * 10), None, MonitorConfig(gpu_hourly_usd=1.0))
    for _ in range(14):
        mon.tick()
    rep = mon.report()
    snap = mon.snapshot()
    assert rep["wasted_usd_so_far"] == snap["session"]["wasted_usd_so_far"]
    assert rep["wasted_usd_so_far"] > 0  # not collapsed to $0.00 by per-card rounding


class _UtilNoneSampler(GpuSampler):
    backend = "util-none"

    def gpu_count(self) -> int:
        return 1

    def read(self) -> list[GpuReading]:
        return [GpuReading(time.time(), 0, "GPU0", util_pct=None, mem_used_mb=None,
                           mem_total_mb=None, power_w=None, power_limit_w=None,
                           sm_clock_mhz=None, sm_clock_max_mhz=None, temp_c=None)]


def test_unknown_util_is_not_billed_as_idle():
    # A failed util read (None) is UNKNOWN, not idle — don't over-report wasted-$.
    mon = Monitor(_UtilNoneSampler(), None, MonitorConfig(gpu_hourly_usd=1.0))
    for _ in range(10):
        mon.tick()
    assert mon.snapshot()["session"]["idle_seconds"] == 0.0


def test_single_gpu_snapshot_shape_unchanged():
    # Back-compat: a 1-GPU box still exposes the same top-level shape.
    mon = Monitor(MultiMock([2.0]), None, MonitorConfig(gpu_hourly_usd=1.0))
    for _ in range(10):
        mon.tick()
    s = mon.snapshot()
    assert s["latest"]["gpu_name"] == "GPU0"
    assert s["session"]["idle_seconds"] == 10.0  # one card, identical to before
