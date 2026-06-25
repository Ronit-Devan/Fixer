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
from et_monitor.llama import LlamaMetrics, LlamaProps, LlamaScraper
from et_monitor.perf import WorkloadSpec, roofline
from et_monitor.types import Diagnosis, Snapshot, Verdict

# Verdicts quiescent enough to back the sample rate off on: a healthy box or a
# plainly-idle one. Everything else (decode/memory/KV/throttle/unknown) keeps the
# loop at the fast base rate so a developing problem is caught promptly.
_QUIESCENT_VERDICTS: frozenset[Verdict] = frozenset(
    {Verdict.HEALTHY, Verdict.IDLE_NO_REQUESTS}
)

log = logging.getLogger(__name__)


@dataclass
class MonitorConfig:
    interval_s: float = 1.0  # base (fast) sample interval — the floor when active
    history_seconds: int = 1800  # 30 min of 1s samples retained for charts
    window_seconds: int = 30  # verdict is computed over this trailing window
    gpu_hourly_usd: float = 0.0  # for the wasted-$ readout; 0 disables it
    thresholds: Thresholds = None  # type: ignore[assignment]
    # Adaptive sampling: when the box is provably quiescent (healthy or plainly
    # idle, with no remediation in flight) the loop multiplicatively backs off
    # the sample rate up to ``max_interval_s`` — cutting steady-state CPU /
    # subprocess / NVML cost several-fold — and snaps straight back to the fast
    # base interval the instant anything changes or warrants attention. This is
    # the single biggest overhead win and it preserves fast reaction to real
    # transitions. Set ``adaptive_sampling=False`` for a fixed deterministic rate.
    adaptive_sampling: bool = True
    max_interval_s: float = 5.0
    stable_ticks_to_backoff: int = 5  # consecutive quiescent ticks before growing
    # Static hardware+model facts for the decode roofline (MBU / single-stream
    # ceiling / partial-offload detection). None => the analyzer falls back to its
    # util-based heuristics. Captured once at setup (et-monitor detect).
    workload_spec: WorkloadSpec | None = None

    def __post_init__(self) -> None:
        if self.thresholds is None:
            self.thresholds = Thresholds()
        # Keep the back-off ceiling from starving the diagnosis window of the
        # minimum samples it needs (min_samples=3): never let the interval grow
        # so large that the trailing window can't hold a few readings.
        safe_ceiling = max(self.interval_s, self.window_seconds / 5.0)
        self.max_interval_s = max(self.interval_s, min(self.max_interval_s, safe_ceiling))


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


class _GpuTrack:
    """Per-GPU rolling history + session accounting + last diagnosis.

    One of these exists per physical GPU the sampler reports. Factoring the
    per-card state out is what makes the monitor work on a DGX (8 cards) or any
    multi-GPU node instead of silently watching only card 0.
    """

    def __init__(self, index: int, config: MonitorConfig, started_at_s: float) -> None:
        self.index = index
        self.config = config
        self.started_at_s = started_at_s
        maxlen = max(1, int(config.history_seconds / config.interval_s))
        self.history: deque[Snapshot] = deque(maxlen=maxlen)
        self.idle_seconds = 0.0
        self.total_seconds = 0.0
        self.util_sum = 0.0
        self.util_n = 0
        self.util_peak = 0.0
        self.verdict_seconds: dict[str, float] = {}
        self.last_diag: Diagnosis | None = None
        self.last_verdict: Verdict | None = None
        self.stable_count = 0

    def append_and_account(self, snap: Snapshot, dt: float) -> None:
        self.history.append(snap)
        self.total_seconds += dt
        t = self.config.thresholds
        if snap.llama_reachable:
            idle = (snap.requests_processing or 0) < 1
        elif snap.util_pct is not None:
            idle = snap.util_pct < t.idle_util_pct
        else:
            # A failed util read is UNKNOWN, not idle — coercing None->0 used to
            # silently bill an unprovable idle and over-report wasted-$.
            idle = False
        if idle:
            self.idle_seconds += dt
        if snap.util_pct is not None:
            self.util_sum += snap.util_pct
            self.util_n += 1
            self.util_peak = max(self.util_peak, snap.util_pct)

    def window(self) -> list[Snapshot]:
        """Trailing window, walked newest-first (O(window)).

        Skips any sample stamped in the *future* relative to ``now``: an NTP step
        backward can leave stale samples timestamped after the current clock, and
        returning them would feed a non-monotonic time series to the trend
        regression (corrupting predictive verdicts). They age out of the deque
        within a window's worth of ticks, so the skip is transient."""
        now = time.time()
        cutoff = now - self.config.window_seconds
        out: list[Snapshot] = []
        for s in reversed(self.history):
            if s.timestamp_s > now:
                continue  # stale 'future' sample after a backward clock step
            if s.timestamp_s < cutoff:
                break
            out.append(s)
        out.reverse()
        return out

    def idle_fraction(self) -> float:
        return self.idle_seconds / self.total_seconds if self.total_seconds else 0.0


class Monitor:
    def __init__(
        self,
        gpu_sampler: GpuSampler,
        llama_scraper: LlamaScraper | None,
        config: MonitorConfig | None = None,
        alert_manager=None,
        remediation_manager=None,
        *,
        host_label: str = "",
        remediation_factory=None,
        alert_factory=None,
    ) -> None:
        self.gpu = gpu_sampler
        self.llama = llama_scraper
        self.config = config or MonitorConfig()
        self.host_label = host_label
        # Optional ET alert + remediation layers. The single-instance params apply
        # to the PRIMARY (lowest-index) GPU — backward compatible with the
        # single-GPU product. The *_factory(index) callables build one manager
        # PER GPU for a multi-GPU box, so each card verifies/approves/breaks
        # independently and a fleet stays per-(node,gpu) keyed.
        self.alert_manager = alert_manager
        self.remediation_manager = remediation_manager
        self.remediation_factory = remediation_factory
        self.alert_factory = alert_factory
        self._rem_mgrs: dict[int, object] = {}
        self._alert_mgrs: dict[int, object] = {}
        # Box-wide desired remediation mode (the kill-switch), persisted here so a
        # flip during the startup window — before any per-GPU manager has been
        # lazily created — still applies to every manager built later.
        self._rem_mode: object | None = None
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._prev_llama: LlamaMetrics | None = None  # box-level (one llama-server)
        self.started_at_s: float = time.time()
        # Per-GPU state, created lazily as cards are first seen.
        self._tracks: dict[int, _GpuTrack] = {}
        self._primary: int | None = None
        # Adaptive-sampling + overhead state (box-level cadence).
        self._interval_s: float = self.config.interval_s
        self._tick_cost_ema_s: float | None = None
        self._stable_count: int = 0

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
        last_tick: float | None = None
        while not self._stop.is_set():
            tick_start = time.time()
            # Account the REAL time since the last tick (not a fixed nominal),
            # so idle-seconds / wasted-$ / verdict-time stay correct under the
            # variable cadence the adaptive scheduler produces.
            # Guard against a backward clock step (NTP correction on an edge box):
            # a negative dt would corrupt idle/uptime/verdict-time accounting. Treat
            # a backward jump as zero elapsed for this tick.
            dt = self._interval_s if last_tick is None else max(0.0, tick_start - last_tick)
            last_tick = tick_start
            try:
                self.tick(dt=dt)
            except Exception:  # noqa: BLE001; the loop must never die
                log.exception("monitor tick failed")
            elapsed = time.time() - tick_start
            self._record_tick_cost(elapsed)
            self._interval_s = self._next_interval(self._interval_s)
            self._stop.wait(max(0.0, self._interval_s - elapsed))

    def _next_interval(self, current: float) -> float:
        """Adaptive cadence: grow only when EVERY GPU is quiescent.

        On a multi-GPU box one busy/at-risk card keeps the whole loop fast — we
        never slow sampling while any GPU needs attention or any remediation is
        verifying.
        """
        if not self.config.adaptive_sampling:
            return self.config.interval_s
        tracks = list(self._tracks.values()) or None
        if tracks is None:
            return self.config.interval_s
        all_quiescent = all(
            tr.last_diag is not None
            and tr.last_diag.verdict in _QUIESCENT_VERDICTS
            and tr.last_diag.verdict == tr.last_verdict
            for tr in tracks
        )
        for tr in tracks:
            tr.last_verdict = tr.last_diag.verdict if tr.last_diag is not None else None
        if not all_quiescent or self._any_remediation_verifying():
            self._stable_count = 0
            return self.config.interval_s
        self._stable_count += 1
        if self._stable_count >= self.config.stable_ticks_to_backoff:
            return min(self.config.max_interval_s, max(current * 2.0, self.config.interval_s))
        return current

    # -- per-GPU manager resolution (back-compat single + multi-GPU factory) --

    def _rem_for(self, index: int):
        if self.remediation_factory is not None:
            if index not in self._rem_mgrs:
                mgr = self.remediation_factory(index)
                # Apply any box-wide mode set before this GPU first appeared, so
                # a startup-window kill-switch is honored by lazily-built managers.
                if self._rem_mode is not None:
                    try:
                        mgr.set_mode(self._rem_mode)
                    except Exception:  # noqa: BLE001
                        pass
                self._rem_mgrs[index] = mgr
            return self._rem_mgrs[index]
        if self.remediation_manager is not None and index == self._primary:
            return self.remediation_manager
        return None

    def set_remediation_mode(self, mode) -> None:
        """Box-wide kill-switch: record the desired mode AND flip every manager
        (existing and future). Routed here from the HTTP endpoint so a flip during
        the startup window isn't silently lost."""
        self._rem_mode = mode
        for m in self.remediation_managers().values():
            try:
                m.set_mode(mode)
            except Exception:  # noqa: BLE001
                pass

    def remediation_mode_value(self):
        """The current box-wide mode value, even before any manager exists."""
        mgrs = self.remediation_managers()
        if mgrs:
            return next(iter(mgrs.values())).mode.value
        return getattr(self._rem_mode, "value", None)

    def _alert_for(self, index: int):
        if self.alert_factory is not None:
            if index not in self._alert_mgrs:
                self._alert_mgrs[index] = self.alert_factory(index)
            return self._alert_mgrs[index]
        if self.alert_manager is not None and index == self._primary:
            return self.alert_manager
        return None

    def _all_rem_managers(self) -> list:
        mgrs = list(self._rem_mgrs.values())
        if self.remediation_factory is None and self.remediation_manager is not None:
            mgrs.append(self.remediation_manager)
        return mgrs

    def _any_remediation_verifying(self) -> bool:
        for rm in self._all_rem_managers():
            try:
                if rm.status().get("state") == "verifying":
                    return True
            except Exception:  # noqa: BLE001
                pass
        return False

    def _record_tick_cost(self, elapsed_s: float) -> None:
        prev = self._tick_cost_ema_s
        self._tick_cost_ema_s = elapsed_s if prev is None else (0.8 * prev + 0.2 * elapsed_s)

    # -- sampling ------------------------------------------------------------

    def tick(self, dt: float | None = None) -> Snapshot:
        """Sample EVERY GPU on the box, diagnose + remediate each independently.

        Returns the primary (lowest-index) GPU's snapshot for backward
        compatibility. ``dt`` is the accounting time this tick represents; the
        live loop passes real elapsed time, a direct caller defaults to the base
        interval (so synchronous tick() keeps 'one tick == interval_s').
        """
        dt = self.config.interval_s if dt is None else dt
        readings = self.gpu.read()

        # llama-server metrics are box-level (one server) — read once and attach
        # to every GPU's snapshot. Token rates derive from box-level counters.
        lm = self.llama.read() if self.llama else None
        # Static launch config (ctx/slots/cache types) — cached after first read,
        # so this is a dict lookup on every tick but the first few. Optional on the
        # scraper (demo/custom scrapers may not implement it), so probe by attr.
        _read_props = getattr(self.llama, "read_props", None) if self.llama else None
        props: LlamaProps | None = _read_props() if _read_props else None
        gen_tps = prompt_tps = None
        if lm is not None and self._prev_llama is not None:
            ldt = lm.timestamp_s - self._prev_llama.timestamp_s
            gen_tps = _rate(self._prev_llama.predicted_tokens_total, lm.predicted_tokens_total, ldt)
            prompt_tps = _rate(self._prev_llama.prompt_tokens_total, lm.prompt_tokens_total, ldt)
        # Advance the rate baseline only when time did not go backward, so token
        # rates and snapshot timestamps stay monotonic across an NTP step. A
        # counter RESET (server restart) still advances it (its timestamp moves
        # forward), so rates resume cleanly after the restart.
        if lm is not None and (
            self._prev_llama is None or lm.timestamp_s >= self._prev_llama.timestamp_s
        ):
            self._prev_llama = lm

        if not readings:
            readings = [None]  # synthesize a "(no GPU)" track so the loop still runs

        primary_snap: Snapshot | None = None
        for gr in readings:
            index = gr.index if gr is not None else 0
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
                ctx_size=props.ctx_size if props else None,
                total_slots=props.total_slots if props else None,
                cache_type_k=props.cache_type_k if props else None,
                cache_type_v=props.cache_type_v if props else None,
                cont_batching=props.cont_batching if props else None,
            )
            with self._lock:
                track = self._tracks.get(index)
                if track is None:
                    track = _GpuTrack(index, self.config, self.started_at_s)
                    self._tracks[index] = track
                    self._primary = min(self._tracks)
                track.append_and_account(snap, dt)
                window = track.window()  # one tail-walk per GPU, reused below

            try:
                diag = analyze(window, self.config.thresholds, self.config.workload_spec)
            except Exception:  # noqa: BLE001
                diag = None
            track.last_diag = diag
            if diag is not None:
                with self._lock:
                    track.verdict_seconds[diag.verdict.value] = (
                        track.verdict_seconds.get(diag.verdict.value, 0.0) + dt
                    )
                self._dispatch(index, diag, window, snap)
            if index == self._primary:
                primary_snap = snap

        return primary_snap if primary_snap is not None else snap

    def _dispatch(self, index: int, diag: Diagnosis, window: list[Snapshot], snap: Snapshot) -> None:
        """Feed this GPU's diagnosis to its alert + remediation managers."""
        am = self._alert_for(index)
        if am is not None:
            try:
                am.observe(diag, snap.timestamp_s, gpu_name=snap.gpu_name)
            except Exception:  # noqa: BLE001
                log.exception("alert evaluation failed (gpu %d)", index)
        rm = self._rem_for(index)
        if rm is not None:
            try:
                rm.observe(diag, window, snap.timestamp_s)
            except Exception:  # noqa: BLE001
                log.exception("remediation evaluation failed (gpu %d)", index)

    # -- reads (thread-safe) -------------------------------------------------

    def gpus(self) -> list[int]:
        with self._lock:
            return sorted(self._tracks)

    def remediation_managers(self) -> dict[int, object]:
        """All active remediation managers keyed by GPU index (for the API).

        Includes factory-built per-GPU managers, plus the single back-compat
        manager (mapped to the primary GPU) when no factory is configured.
        """
        out = dict(self._rem_mgrs)
        if (
            self.remediation_factory is None
            and self.remediation_manager is not None
            and self._primary is not None
        ):
            out[self._primary] = self.remediation_manager
        return out

    def _primary_track(self) -> _GpuTrack | None:
        with self._lock:
            if self._primary is None:
                return None
            return self._tracks.get(self._primary)

    def _window(self) -> list[Snapshot]:
        tr = self._primary_track()
        if tr is None:
            return []
        with self._lock:
            return tr.window()

    def snapshot(self) -> dict:
        """Live snapshot. ``latest``/``session`` are the primary GPU + FLEET
        aggregate (back-compat); ``gpus`` carries every card for multi-GPU UIs."""
        with self._lock:
            tracks = [self._tracks[i] for i in sorted(self._tracks)]
            primary = self._tracks.get(self._primary) if self._primary is not None else None
            latest = primary.history[-1] if (primary and primary.history) else None
            session = self._session_locked(tracks)
            gpus = [self._gpu_snapshot_locked(tr) for tr in tracks]
        tick_ms = (
            round(self._tick_cost_ema_s * 1000, 2)
            if self._tick_cost_ema_s is not None
            else None
        )
        return {
            "backend": self.gpu.backend,
            "host_label": self.host_label,
            "gpu_count": len(tracks),
            "latest": _snap_to_dict(latest) if latest else None,
            # Static workload facts + the single-stream ceiling, so the dashboard
            # can frame tokens/sec against the card's limit even while idle.
            "workload": _workload_dict(self.config.workload_spec),
            # Self-reported overhead so "minimal overhead" is observable, not
            # folklore: smoothed per-tick wall cost + the live (adaptive) cadence.
            "perf": {
                "tick_cost_ms": tick_ms,
                "interval_s": round(self._interval_s, 2),
                "adaptive": self.config.adaptive_sampling,
            },
            "session": session,
            "gpus": gpus,
        }

    def _session_locked(self, tracks: list[_GpuTrack]) -> dict:
        """FLEET-aggregate session: summed idle GPU-seconds + summed $ across all
        cards (so an 8-GPU box reports 8 cards' idle cost, not card 0's)."""
        price = self.config.gpu_hourly_usd
        total = sum(tr.total_seconds for tr in tracks)
        idle = sum(tr.idle_seconds for tr in tracks)
        uptime = max((tr.total_seconds for tr in tracks), default=0.0)  # wall time
        idle_frac = idle / total if total else 0.0
        wasted = sum(tr.idle_fraction() * (tr.total_seconds / 3600) * price for tr in tracks)
        proj_monthly = sum(tr.idle_fraction() * 730 * price for tr in tracks)
        return {
            "started_at_s": self.started_at_s,
            "uptime_s": round(uptime, 1),
            "idle_fraction": round(idle_frac, 4),
            "idle_seconds": round(idle, 1),
            "gpu_hourly_usd": price,
            "wasted_usd_so_far": round(wasted, 2),
            "projected_monthly_idle_usd": round(proj_monthly, 2),
        }

    def _gpu_snapshot_locked(self, tr: _GpuTrack) -> dict:
        latest = tr.history[-1] if tr.history else None
        return {
            "index": tr.index,
            "latest": _snap_to_dict(latest) if latest else None,
            "diagnosis": tr.last_diag.to_dict() if tr.last_diag is not None else None,
            "idle_fraction": round(tr.idle_fraction(), 4),
            "is_primary": tr.index == self._primary,
        }

    def history(self, index: int | None = None) -> list[dict]:
        """Per-GPU history (default: primary GPU, for the single-GPU dashboard)."""
        with self._lock:
            idx = index if index is not None else self._primary
            tr = self._tracks.get(idx) if idx is not None else None
            return [_snap_to_dict(s) for s in tr.history] if tr else []

    def report(self) -> dict:
        """Shareable 'how much GPU did we waste' artifact — fleet-aggregate
        headline + per-GPU breakdown."""
        with self._lock:
            tracks = [self._tracks[i] for i in sorted(self._tracks)]
            primary = self._tracks.get(self._primary) if self._primary is not None else None
            verdict_seconds: dict[str, float] = {}
            for tr in tracks:
                for v, secs in tr.verdict_seconds.items():
                    verdict_seconds[v] = verdict_seconds.get(v, 0.0) + secs
            util_sum = sum(tr.util_sum for tr in tracks)
            util_n = sum(tr.util_n for tr in tracks)
            peak_util = max((tr.util_peak for tr in tracks), default=0.0)
            uptime = max((tr.total_seconds for tr in tracks), default=0.0)
            total_gpu_s = sum(tr.total_seconds for tr in tracks)
            idle_gpu_s = sum(tr.idle_seconds for tr in tracks)
            gpu_name = (
                primary.history[-1].gpu_name if (primary and primary.history) else "(no GPU)"
            )
            per_gpu = [self._gpu_report_locked(tr) for tr in tracks]
            # Sum the RAW per-track dollars and round ONCE — summing the already
            # rounded per-GPU values under-reports the fleet headline (each small
            # card rounds to $0.00) and diverges from snapshot()'s session block.
            wasted = sum(tr.idle_fraction() * (tr.total_seconds / 3600) for tr in tracks)
            proj_month = sum(tr.idle_fraction() * 730 for tr in tracks)
            proj_year = sum(tr.idle_fraction() * 8760 for tr in tracks)
        price = self.config.gpu_hourly_usd
        idle_frac = idle_gpu_s / total_gpu_s if total_gpu_s else 0.0
        breakdown = [
            {
                "verdict": v,
                "seconds": round(secs, 1),
                "fraction": round(secs / total_gpu_s, 4) if total_gpu_s else 0.0,
            }
            for v, secs in sorted(verdict_seconds.items(), key=lambda kv: kv[1], reverse=True)
        ]
        return {
            "gpu_name": gpu_name,
            "host_label": self.host_label,
            "gpu_count": len(per_gpu),
            "generated_at_s": time.time(),
            "started_at_s": self.started_at_s,
            "uptime_s": round(uptime, 1),
            "uptime_hours": round(uptime / 3600, 2),
            "avg_util_pct": round(util_sum / util_n, 1) if util_n else 0.0,
            "peak_util_pct": round(peak_util, 1),
            "idle_fraction": round(idle_frac, 4),
            "idle_hours": round(idle_gpu_s / 3600, 2),
            "gpu_hourly_usd": price,
            "wasted_usd_so_far": round(wasted * price, 2),
            "projected_monthly_idle_usd": round(proj_month * price, 2),
            "projected_yearly_idle_usd": round(proj_year * price, 2),
            "verdict_breakdown": breakdown,
            "gpus": per_gpu,
        }

    def _gpu_report_locked(self, tr: _GpuTrack) -> dict:
        price = self.config.gpu_hourly_usd
        frac = tr.idle_fraction()
        name = tr.history[-1].gpu_name if tr.history else "(no GPU)"
        return {
            "index": tr.index,
            "gpu_name": name,
            "avg_util_pct": round(tr.util_sum / tr.util_n, 1) if tr.util_n else 0.0,
            "peak_util_pct": round(tr.util_peak, 1),
            "idle_fraction": round(frac, 4),
            "wasted_usd_so_far": round(frac * (tr.total_seconds / 3600) * price, 2),
            "projected_monthly_idle_usd": round(frac * 730 * price, 2),
            "projected_yearly_idle_usd": round(frac * 8760 * price, 2),
        }

    def diagnosis(self, index: int | None = None) -> Diagnosis:
        """Diagnosis for one GPU (default: primary, for the single-GPU API)."""
        with self._lock:
            idx = index if index is not None else self._primary
            tr = self._tracks.get(idx) if idx is not None else None
            window = tr.window() if tr else []
        return analyze(window, self.config.thresholds, self.config.workload_spec)

    def diagnosis_all(self) -> dict[int, Diagnosis]:
        with self._lock:
            items = [(i, self._tracks[i].window()) for i in sorted(self._tracks)]
        spec = self.config.workload_spec
        return {i: analyze(w, self.config.thresholds, spec) for i, w in items}


def _workload_dict(spec: WorkloadSpec | None) -> dict | None:
    """The persisted workload spec + its single-stream ceiling, for the UI."""
    if spec is None:
        return None
    rl = roofline(spec, None)
    return {
        "model_name": spec.model_name,
        "gpu_name": spec.gpu_name,
        "mem_bandwidth_gb_s": spec.mem_bandwidth_gb_s,
        "model_gb": round(spec.model_bytes / 1e9, 2) if spec.model_bytes else None,
        "n_layers": spec.n_layers,
        "n_gpu_layers": spec.n_gpu_layers,
        "offload_fraction": round(spec.offload_fraction, 3),
        "ceiling_tok_s": round(rl.ceiling_tok_s, 1) if (rl and rl.ceiling_tok_s) else None,
        "ideal_tok_s": round(rl.ideal_tok_s, 1) if (rl and rl.ideal_tok_s) else None,
    }


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
        "ctx_size": s.ctx_size,
        "total_slots": s.total_slots,
        "cache_type_k": s.cache_type_k,
        "cache_type_v": s.cache_type_v,
        "cont_batching": s.cont_batching,
    }
