"""The live monitor: sampling loop, rolling history, and the money readout.

``Monitor`` owns a background thread that, every ``interval_s``:
  1. reads the GPU (NVML / nvidia-smi / mock),
  2. scrapes llama-server /metrics (if configured / reachable),
  3. derives live token rates from counter deltas,
  4. appends a unified ``Snapshot`` to a ring buffer,
  5. accumulates idle time and the dollar cost of that idle time.

The web layer never samples directly; it reads ``snapshot()`` / ``history()``
/ ``diagnosis()`` off this object, which are cheap and thread-safe enough for a
single-writer / many-reader poll model (the GIL makes the list ops atomic and
we copy on read).
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass

from et_monitor.analyzer import Thresholds, analyze
from et_monitor.gpu import GpuSampler
from et_monitor.llama import LlamaMetrics, LlamaScraper
from et_monitor.types import Diagnosis, Snapshot

log = logging.getLogger(__name__)


@dataclass
class MonitorConfig:
    interval_s: float = 1.0
    history_seconds: int = 1800  # 30 min of 1s samples retained for charts
    window_seconds: int = 30  # verdict is computed over this trailing window
    gpu_hourly_usd: float = 0.0  # for the wasted-$ readout; 0 disables it
    thresholds: Thresholds = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.thresholds is None:
            self.thresholds = Thresholds()


def _rate(prev: float | None, cur: float | None, dt: float) -> float | None:
    """Per-second rate from two counter samples; None if not derivable.

    Guards the counter reset case (server restart drops the total) by treating
    a decrease as 'no rate' rather than a negative spike.
    """
    if prev is None or cur is None or dt <= 0:
        return None
    if cur < prev:
        return None
    return (cur - prev) / dt


class Monitor:
    def __init__(
        self,
        gpu_sampler: GpuSampler,
        llama_scraper: LlamaScraper | None,
        config: MonitorConfig | None = None,
        alert_manager=None,
    ) -> None:
        self.gpu = gpu_sampler
        self.llama = llama_scraper
        self.config = config or MonitorConfig()
        self.alert_manager = alert_manager
        maxlen = max(1, int(self.config.history_seconds / self.config.interval_s))
        self._history: deque[Snapshot] = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._prev_llama: LlamaMetrics | None = None
        # Session-level accounting.
        self.started_at_s: float = time.time()
        self._idle_seconds: float = 0.0
        self._total_seconds: float = 0.0
        self._util_sum: float = 0.0
        self._util_n: int = 0
        self._util_peak: float = 0.0
        self._verdict_seconds: dict[str, float] = {}

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="et-monitor", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2 * self.config.interval_s + 1)
            self._thread = None

    def _run(self) -> None:
        while not self._stop.is_set():
            tick = time.time()
            try:
                self.tick()
            except Exception:  # noqa: BLE001; the loop must never die
                log.exception("monitor tick failed")
            elapsed = time.time() - tick
            self._stop.wait(max(0.0, self.config.interval_s - elapsed))

    # -- sampling ------------------------------------------------------------

    def tick(self) -> Snapshot:
        """Take one unified sample, update history + accounting, return it."""
        readings = self.gpu.read()
        # Single-box product: GPU 0 is the device of record. (Multi-GPU is a
        # later iteration; the dashboard and verdicts assume one card.)
        gr = readings[0] if readings else None

        lm = self.llama.read() if self.llama else None
        gen_tps = prompt_tps = None
        if lm is not None and self._prev_llama is not None:
            dt = lm.timestamp_s - self._prev_llama.timestamp_s
            gen_tps = _rate(
                self._prev_llama.predicted_tokens_total,
                lm.predicted_tokens_total,
                dt,
            )
            prompt_tps = _rate(
                self._prev_llama.prompt_tokens_total,
                lm.prompt_tokens_total,
                dt,
            )
        if lm is not None:
            self._prev_llama = lm

        snap = Snapshot(
            timestamp_s=tick_time(gr, lm),
            gpu_name=gr.name if gr else "(no GPU)",
            util_pct=gr.util_pct if gr else None,
            mem_used_mb=gr.mem_used_mb if gr else None,
            mem_total_mb=gr.mem_total_mb if gr else None,
            power_w=gr.power_w if gr else None,
            power_limit_w=gr.power_limit_w if gr else None,
            sm_clock_mhz=gr.sm_clock_mhz if gr else None,
            sm_clock_max_mhz=gr.sm_clock_max_mhz if gr else None,
            temp_c=gr.temp_c if gr else None,
            llama_reachable=lm is not None,
            requests_processing=lm.requests_processing if lm else None,
            requests_deferred=lm.requests_deferred if lm else None,
            kv_cache_usage_ratio=lm.kv_cache_usage_ratio if lm else None,
            gen_tokens_per_s=gen_tps,
            prompt_tokens_per_s=prompt_tps,
        )

        with self._lock:
            self._history.append(snap)
            self._account(snap)

        # Diagnose once per tick (outside the lock; diagnosis() re-acquires it)
        # and use it for both verdict-time accounting and alerting.
        try:
            diag = self.diagnosis()
        except Exception:  # noqa: BLE001
            diag = None
        if diag is not None:
            with self._lock:
                self._verdict_seconds[diag.verdict.value] = (
                    self._verdict_seconds.get(diag.verdict.value, 0.0)
                    + self.config.interval_s
                )
            if self.alert_manager is not None:
                try:
                    self.alert_manager.observe(
                        diag, snap.timestamp_s, gpu_name=snap.gpu_name
                    )
                except Exception:  # noqa: BLE001
                    log.exception("alert evaluation failed")
        return snap

    def _account(self, snap: Snapshot) -> None:
        """Roll up idle time and its dollar cost. Caller holds the lock."""
        dt = self.config.interval_s
        self._total_seconds += dt
        t = self.config.thresholds
        if snap.llama_reachable:
            idle = (snap.requests_processing or 0) < 1
        else:
            idle = (snap.util_pct or 0) < t.idle_util_pct
        if idle:
            self._idle_seconds += dt
        if snap.util_pct is not None:
            self._util_sum += snap.util_pct
            self._util_n += 1
            self._util_peak = max(self._util_peak, snap.util_pct)

    # -- reads (thread-safe) -------------------------------------------------

    def _window(self) -> list[Snapshot]:
        cutoff = time.time() - self.config.window_seconds
        with self._lock:
            return [s for s in self._history if s.timestamp_s >= cutoff]

    def snapshot(self) -> dict:
        with self._lock:
            latest = self._history[-1] if self._history else None
        idle_frac = (
            self._idle_seconds / self._total_seconds
            if self._total_seconds
            else 0.0
        )
        wasted_usd = idle_frac * (self._total_seconds / 3600) * self.config.gpu_hourly_usd
        proj_monthly = idle_frac * 730 * self.config.gpu_hourly_usd  # 730h/mo
        return {
            "backend": self.gpu.backend,
            "latest": _snap_to_dict(latest) if latest else None,
            "session": {
                "started_at_s": self.started_at_s,
                "uptime_s": round(self._total_seconds, 1),
                "idle_fraction": round(idle_frac, 4),
                "idle_seconds": round(self._idle_seconds, 1),
                "gpu_hourly_usd": self.config.gpu_hourly_usd,
                "wasted_usd_so_far": round(wasted_usd, 2),
                "projected_monthly_idle_usd": round(proj_monthly, 2),
            },
        }

    def history(self) -> list[dict]:
        with self._lock:
            return [_snap_to_dict(s) for s in self._history]

    def report(self) -> dict:
        """Session summary; the shareable 'how much GPU did we waste' artifact."""
        with self._lock:
            total = self._total_seconds
            idle = self._idle_seconds
            avg_util = self._util_sum / self._util_n if self._util_n else 0.0
            peak_util = self._util_peak
            latest = self._history[-1] if self._history else None
            verdict_seconds = dict(self._verdict_seconds)
        idle_frac = idle / total if total else 0.0
        price = self.config.gpu_hourly_usd
        # Verdict time breakdown, largest first.
        breakdown = [
            {
                "verdict": v,
                "seconds": round(secs, 1),
                "fraction": round(secs / total, 4) if total else 0.0,
            }
            for v, secs in sorted(
                verdict_seconds.items(), key=lambda kv: kv[1], reverse=True
            )
        ]
        return {
            "gpu_name": latest.gpu_name if latest else "(no GPU)",
            "host_label": "",
            "generated_at_s": time.time(),
            "started_at_s": self.started_at_s,
            "uptime_s": round(total, 1),
            "uptime_hours": round(total / 3600, 2),
            "avg_util_pct": round(avg_util, 1),
            "peak_util_pct": round(peak_util, 1),
            "idle_fraction": round(idle_frac, 4),
            "idle_hours": round(idle / 3600, 2),
            "gpu_hourly_usd": price,
            "wasted_usd_so_far": round(idle_frac * (total / 3600) * price, 2),
            "projected_monthly_idle_usd": round(idle_frac * 730 * price, 2),
            "projected_yearly_idle_usd": round(idle_frac * 8760 * price, 2),
            "verdict_breakdown": breakdown,
        }

    def diagnosis(self) -> Diagnosis:
        return analyze(self._window(), self.config.thresholds)


def tick_time(gr, lm) -> float:
    if gr is not None:
        return gr.timestamp_s
    if lm is not None:
        return lm.timestamp_s
    return time.time()


def _snap_to_dict(s: Snapshot) -> dict:
    return {
        "t": s.timestamp_s,
        "gpu_name": s.gpu_name,
        "util_pct": s.util_pct,
        "mem_used_mb": s.mem_used_mb,
        "mem_total_mb": s.mem_total_mb,
        "mem_used_ratio": s.mem_used_ratio,
        "power_w": s.power_w,
        "power_limit_w": s.power_limit_w,
        "sm_clock_mhz": s.sm_clock_mhz,
        "sm_clock_max_mhz": s.sm_clock_max_mhz,
        "temp_c": s.temp_c,
        "llama_reachable": s.llama_reachable,
        "requests_processing": s.requests_processing,
        "requests_deferred": s.requests_deferred,
        "kv_cache_usage_ratio": s.kv_cache_usage_ratio,
        "gen_tokens_per_s": s.gen_tokens_per_s,
        "prompt_tokens_per_s": s.prompt_tokens_per_s,
    }
