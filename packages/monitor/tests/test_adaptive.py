"""Wave B efficiency: adaptive back-off, real-dt accounting, tail-walk window."""

from __future__ import annotations

from et_monitor.gpu import MockGpuSampler
from et_monitor.state import Monitor, MonitorConfig, _GpuTrack
from et_monitor.types import Diagnosis, Verdict


def _diag(verdict: Verdict, severity: str) -> Diagnosis:
    return Diagnosis(verdict, verdict.value, severity, 0.8, "s")


def _mon(**cfg) -> Monitor:
    return Monitor(MockGpuSampler(), None, MonitorConfig(**cfg))


def _set_verdict(mon: Monitor, verdict: Verdict, severity: str = "ok") -> None:
    """Set the primary GPU track's last diagnosis (what the scheduler reads)."""
    tr = mon._tracks.get(0)
    if tr is None:
        tr = _GpuTrack(0, mon.config, 0.0)
        mon._tracks[0] = tr
        mon._primary = 0
    tr.last_diag = _diag(verdict, severity)


def test_backs_off_when_quiescent():
    mon = _mon(interval_s=1.0, max_interval_s=5.0, stable_ticks_to_backoff=3)
    i = mon.config.interval_s
    intervals = []
    for _ in range(8):
        _set_verdict(mon, Verdict.HEALTHY, "ok")
        i = mon._next_interval(i)
        intervals.append(i)
    # Holds base for the first few quiescent ticks, then grows toward the ceiling.
    assert intervals[0] == 1.0
    assert intervals[-1] == 5.0
    assert max(intervals) <= mon.config.max_interval_s


def test_snaps_back_to_base_on_attention_verdict():
    mon = _mon(interval_s=1.0, max_interval_s=5.0, stable_ticks_to_backoff=2)
    i = 1.0
    for _ in range(6):  # back off first
        _set_verdict(mon, Verdict.HEALTHY, "ok")
        i = mon._next_interval(i)
    assert i > 1.0
    # A throttle verdict must snap straight back to the fast base rate.
    _set_verdict(mon, Verdict.THERMAL_THROTTLE, "crit")
    assert mon._next_interval(i) == 1.0


def test_idle_verdict_is_quiescent_but_decode_is_not():
    mon = _mon(interval_s=1.0, max_interval_s=4.0, stable_ticks_to_backoff=1)
    i = 1.0
    for _ in range(4):
        _set_verdict(mon, Verdict.IDLE_NO_REQUESTS, "info")
        i = mon._next_interval(i)
    assert i > 1.0  # plainly-idle box backs off
    # Actively decoding is NOT quiescent — stay fast to catch developing issues.
    _set_verdict(mon, Verdict.DECODE_BANDWIDTH_BOUND, "info")
    assert mon._next_interval(i) == 1.0


def test_adaptive_can_be_disabled():
    mon = _mon(interval_s=1.0, max_interval_s=5.0, adaptive_sampling=False)
    _set_verdict(mon, Verdict.HEALTHY, "ok")
    for _ in range(10):
        assert mon._next_interval(5.0) == 1.0  # fixed rate, never grows


def test_does_not_back_off_while_remediation_verifying():
    class _RM:
        def status(self):
            return {"state": "verifying"}

    mon = _mon(interval_s=1.0, max_interval_s=5.0, stable_ticks_to_backoff=1)
    mon.remediation_manager = _RM()
    i = 1.0
    for _ in range(4):
        _set_verdict(mon, Verdict.HEALTHY, "ok")
        i = mon._next_interval(i)
    assert i == 1.0  # must stay fast so the verify window resolves promptly


def test_accounting_uses_real_dt():
    sampler = MockGpuSampler()
    sampler.util_pct = 2.0  # idle
    mon = Monitor(sampler, None, MonitorConfig(interval_s=1.0, gpu_hourly_usd=1.0))
    for _ in range(5):
        mon.tick(dt=4.0)  # each tick represents 4 real seconds
    snap = mon.snapshot()
    assert snap["session"]["idle_seconds"] == 20.0  # 5 * 4, not 5 * interval_s
    assert snap["session"]["uptime_s"] == 20.0


def test_default_tick_keeps_nominal_dt():
    sampler = MockGpuSampler()
    sampler.util_pct = 2.0
    mon = Monitor(sampler, None, MonitorConfig(interval_s=1.0))
    for _ in range(10):
        mon.tick()  # no dt -> nominal interval_s, preserving legacy semantics
    assert mon.snapshot()["session"]["idle_seconds"] == 10.0


def test_window_tail_walk_matches_naive():
    mon = Monitor(MockGpuSampler(), None, MonitorConfig(window_seconds=30))
    for _ in range(20):
        mon.tick()
    track = mon._tracks[0]
    naive = [s for s in track.history if s.timestamp_s >= (__import__("time").time() - 30)]
    assert [s.timestamp_s for s in mon._window()] == [s.timestamp_s for s in naive]


def test_perf_readout_in_snapshot():
    mon = Monitor(MockGpuSampler(), None, MonitorConfig())
    mon.tick()
    perf = mon.snapshot()["perf"]
    assert "tick_cost_ms" in perf
    assert perf["interval_s"] == mon.config.interval_s
    assert perf["adaptive"] is True
