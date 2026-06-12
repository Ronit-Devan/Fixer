"""Monitor orchestration: rate derivation and idle/$ accounting."""

from __future__ import annotations

from et_monitor.gpu import MockGpuSampler
from et_monitor.state import Monitor, MonitorConfig, _rate


def test_rate_basic():
    assert _rate(100.0, 160.0, 2.0) == 30.0


def test_rate_handles_counter_reset():
    # server restarted: total dropped -> no negative spike
    assert _rate(500.0, 10.0, 1.0) is None


def test_rate_handles_missing():
    assert _rate(None, 10.0, 1.0) is None
    assert _rate(10.0, None, 1.0) is None
    assert _rate(10.0, 20.0, 0.0) is None


def test_idle_accounting_counts_idle_ticks():
    sampler = MockGpuSampler()
    sampler.util_pct = 2.0  # below idle threshold
    mon = Monitor(sampler, None, MonitorConfig(interval_s=1.0, gpu_hourly_usd=1.0))
    for _ in range(10):
        mon.tick()
    snap = mon.snapshot()
    assert snap["session"]["idle_fraction"] == 1.0
    assert snap["session"]["idle_seconds"] == 10.0
    # 10s at 100% idle, $1/hr => 10/3600 dollars
    assert snap["session"]["wasted_usd_so_far"] == round(10 / 3600, 2)


def test_busy_ticks_are_not_idle():
    sampler = MockGpuSampler()
    sampler.util_pct = 90.0
    mon = Monitor(sampler, None, MonitorConfig(interval_s=1.0, gpu_hourly_usd=1.0))
    for _ in range(5):
        mon.tick()
    assert mon.snapshot()["session"]["idle_fraction"] == 0.0


def test_snapshot_shape():
    mon = Monitor(MockGpuSampler(), None, MonitorConfig())
    mon.tick()
    snap = mon.snapshot()
    assert snap["backend"] == "mock"
    assert snap["latest"]["gpu_name"]
    assert "projected_monthly_idle_usd" in snap["session"]
