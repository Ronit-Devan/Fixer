"""Long-running robustness: NTP clock steps, server restarts, GPU driver loss."""

from __future__ import annotations

import et_monitor.state as state_mod
from et_monitor.gpu import GpuReading, MockGpuSampler, NvmlGpuSampler
from et_monitor.llama import LlamaMetrics
from et_monitor.state import Monitor, MonitorConfig, _GpuTrack
from et_monitor.types import Snapshot


def _snap(ts: float) -> Snapshot:
    return Snapshot(timestamp_s=ts, gpu_name="g", util_pct=50.0, mem_used_mb=1.0,
                    mem_total_mb=2.0, power_w=1.0, power_limit_w=2.0, sm_clock_mhz=1,
                    sm_clock_max_mhz=2, temp_c=1.0)


def test_window_excludes_future_samples_after_backward_clock_step(monkeypatch):
    # Clock is "now" = 1000. The deque holds a stale pre-step sample stamped in
    # the future (1005) plus three in-window samples. window() must return the
    # in-window ones in monotonic order and drop the future one.
    monkeypatch.setattr(state_mod.time, "time", lambda: 1000.0)
    tr = _GpuTrack(0, MonitorConfig(window_seconds=30), started_at_s=0.0)
    for ts in (1005.0, 998.0, 999.0, 1000.0):  # 1005 = stale 'future' sample
        tr.history.append(_snap(ts))
    times = [s.timestamp_s for s in tr.window()]
    assert times == [998.0, 999.0, 1000.0]            # future sample excluded
    assert times == sorted(times)                      # strictly monotonic


def _scraper(metrics):
    class _S:
        def __init__(self, seq):
            self._seq = list(seq)
            self._i = 0

        def read(self):
            m = self._seq[min(self._i, len(self._seq) - 1)]
            self._i += 1
            return m
    return _S(metrics)


def _lm(ts, pred):
    return LlamaMetrics(timestamp_s=ts, reachable=True, raw={}, predicted_tokens_total=pred,
                        prompt_tokens_total=0.0, requests_processing=1.0)


def test_token_rate_survives_backward_clock_step():
    # Baseline must not advance to a backward-stamped sample, so rates resume
    # correctly against the last good reading instead of being corrupted.
    seq = [_lm(100.0, 1000.0), _lm(101.0, 1100.0), _lm(100.5, 1150.0), _lm(102.0, 1300.0)]
    mon = Monitor(MockGpuSampler(), _scraper(seq), MonitorConfig())
    for _ in range(3):  # prime, rate, backward blip (rate None)
        mon.tick()
    snap = mon.tick()                    # recovers against the 101.0 baseline
    assert snap.gen_tokens_per_s == 200.0  # (1300-1100)/(102-101), NOT a skewed value


def test_token_rate_resumes_after_server_restart_counter_reset():
    # Counter resets to a small value (restart). Baseline DOES advance (time went
    # forward), so rates resume after the reset instead of freezing forever.
    seq = [_lm(110.0, 5000.0), _lm(111.0, 10.0), _lm(112.0, 210.0)]
    mon = Monitor(MockGpuSampler(), _scraper(seq), MonitorConfig())
    mon.tick()           # baseline 5000
    s2 = mon.tick()      # counter reset -> rate None
    assert s2.gen_tokens_per_s is None
    s3 = mon.tick()      # resumes from the reset baseline
    assert s3.gen_tokens_per_s == 200.0


def _dead():
    return [GpuReading(timestamp_s=0.0, index=0, name="x", util_pct=None, mem_used_mb=None,
                       mem_total_mb=None, power_w=None, power_limit_w=None, sm_clock_mhz=None,
                       sm_clock_max_mhz=None, temp_c=None)]


def _live():
    r = _dead()[0]
    return [GpuReading(**{**r.__dict__, "util_pct": 50.0})]


class _FakeNv:
    def __init__(self):
        self.init_calls = 0

    def nvmlInit(self):
        self.init_calls += 1

    def nvmlShutdown(self):
        pass

    def nvmlDeviceGetCount(self):
        return 1

    def nvmlDeviceGetHandleByIndex(self, i):
        return object()


def test_nvml_recovers_after_sustained_dead_reads():
    s = object.__new__(NvmlGpuSampler)
    s._pynvml = _FakeNv()
    s._wanted = None
    s._handles = [(0, object())]
    s._dead_ticks = 0
    # Three consecutive all-None reads trip a re-init attempt.
    for _ in range(3):
        s._track_liveness_and_maybe_recover(_dead())
    assert s._pynvml.init_calls == 1     # recovery attempted exactly once
    assert s._dead_ticks == 0            # reset after a successful re-init
    # A live read keeps it healthy with no further re-inits.
    s._track_liveness_and_maybe_recover(_live())
    assert s._pynvml.init_calls == 1
    assert s._dead_ticks == 0
