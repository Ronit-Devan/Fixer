"""Inference-aware verdict engine.

This is the inference-side analogue of ``gpu_doctor_engine.diagnose``. Where the
training engine attributes GPU *idle* to DataLoader / NCCL / checkpoint stalls
from a PyTorch trace, this attributes the live state of an *inference* box to
the causes that actually matter when you serve a model with llama.cpp:

  IDLE_NO_REQUESTS       GPU sitting idle; you're paying for capacity you
                         aren't using. The core "GPU idleness" money story.
  MEMORY_HEADROOM        Lots of VRAM free; you could run a larger / higher
                         precision model, a bigger context, or more parallel
                         slots on the same card.
  DECODE_BANDWIDTH_BOUND Actively generating at low concurrency: single-stream
                         decode is memory-bandwidth bound, so util plateaus
                         below 100%. Batching concurrent requests raises
                         throughput at the same draw.
  KV_CACHE_PRESSURE      KV cache near full and/or requests being deferred -
                         users are queueing.
  THERMAL_THROTTLE       SM clock dragged well below max while under load.
  HEALTHY                Well used, nothing to do.

Pure function over a recent window of Snapshots. No I/O. Like the training
engine, every rule is evaluated and the dominant condition wins; a per-rule
decision log is returned for transparency.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean

from et_monitor.trends import r_squared, time_to_threshold
from et_monitor.types import VERDICT_TITLES, Diagnosis, Snapshot, Verdict


@dataclass(frozen=True)
class Thresholds:
    idle_util_pct: float = 12.0  # below this = idle
    busy_util_pct: float = 80.0  # at/above this under load = saturated/healthy
    mem_headroom_ratio: float = 0.55  # below this = meaningful VRAM free
    kv_pressure_ratio: float = 0.90  # at/above this = cache nearly full
    throttle_clock_ratio: float = 0.70  # below this under load = throttling
    throttle_util_pct: float = 60.0  # "under load" floor for throttle check
    min_samples: int = 3  # need this many ticks to call anything
    # --- predictive (early-warning) detection ---
    predict_horizon_s: float = 60.0  # only warn on a crossing within this horizon
    throttle_temp_c: float = 84.0  # temperature that tends to trigger HW throttle
    vram_pressure_ratio: float = 0.95  # VRAM fraction treated as OOM-imminent
    min_trend_samples: int = 4  # need this many readings to trust a trend
    min_trend_r2: float = 0.6  # reject noisy fits: the trend must explain >= this


def _vals(window: list[Snapshot], attr: str) -> list[float]:
    out = []
    for s in window:
        v = getattr(s, attr)
        if v is not None:
            out.append(float(v))
    return out


def _mean_or(window: list[Snapshot], attr: str, default: float | None = None):
    vs = _vals(window, attr)
    return mean(vs) if vs else default


def _make(
    verdict: Verdict,
    severity: str,
    confidence: float,
    summary: str,
    evidence: list[str],
    recommendations: list[str],
    metrics: dict,
    *,
    predicted: bool = False,
    horizon_s: float | None = None,
) -> Diagnosis:
    return Diagnosis(
        verdict=verdict,
        title=VERDICT_TITLES[verdict],
        severity=severity,
        confidence=round(max(0.0, min(1.0, confidence)), 2),
        summary=summary,
        evidence=evidence,
        recommendations=recommendations,
        metrics=metrics,
        predicted=predicted,
        horizon_s=round(horizon_s, 1) if horizon_s is not None else None,
    )


def _series(window: list[Snapshot], attr: str) -> tuple[list[float], list[float]]:
    """(timestamps, values) for samples where ``attr`` and the timestamp exist."""
    times: list[float] = []
    vals: list[float] = []
    for s in window:
        v = getattr(s, attr, None)
        if v is not None and s.timestamp_s is not None:
            times.append(float(s.timestamp_s))
            vals.append(float(v))
    return times, vals


def _predict(
    window: list[Snapshot], t: Thresholds, mean_util: float, llama_on: bool, metrics: dict
) -> Diagnosis | None:
    """Early-warning verdicts from trends, before the problem actually lands.

    Ordered by danger: an imminent OOM (which would kill the workload) outranks
    an imminent throttle, which outranks imminent KV saturation. Each fires only
    when the projected crossing is within ``predict_horizon_s``. Confidence rises
    as the crossing gets nearer. Returns None if nothing is imminent.
    """
    if len(window) < t.min_trend_samples:
        return None

    def _conf(horizon: float) -> float:
        return 0.5 + 0.45 * (1.0 - min(horizon, t.predict_horizon_s) / t.predict_horizon_s)

    def _project(times: list[float], vals: list[float], target: float, rising: bool) -> float | None:
        """Horizon to cross ``target`` — only if there are enough points, the fit
        is good enough (not noise), and the crossing is within the horizon."""
        if len(times) < t.min_trend_samples:
            return None
        r2 = r_squared(times, vals)
        if r2 is None or r2 < t.min_trend_r2:
            return None
        tt = time_to_threshold(times, vals, target, rising=rising)
        return tt if (tt is not None and tt <= t.predict_horizon_s) else None

    # 1) VRAM climbing toward OOM — the workload-killing failure.
    mt, mv = _series(window, "mem_used_ratio")
    tt_oom = _project(mt, mv, t.vram_pressure_ratio, True)
    if tt_oom is not None:
        return _make(
            Verdict.VRAM_PRESSURE, "warn", _conf(tt_oom),
            f"VRAM is climbing and is on track to hit {t.vram_pressure_ratio:.0%} "
            f"(OOM risk) in ~{tt_oom:.0f}s. Acting now avoids an out-of-memory crash.",
            [f"VRAM now: {mv[-1]:.0%}", f"Projected OOM in: ~{tt_oom:.0f}s"],
            [
                "Free unused/leaked allocations or stop a stale process holding VRAM.",
                "If this is real demand, lower --ctx-size/--parallel or move to a bigger card before it OOMs.",
            ],
            {**metrics, "predicted_oom_s": round(tt_oom, 1)},
            predicted=True, horizon_s=tt_oom,
        )

    # 2) Thermal throttle imminent (under load): temp rising to the throttle
    #    point, or SM clock already sliding toward the throttle floor.
    if mean_util >= t.throttle_util_pct:
        ttemp, vtemp = _series(window, "temp_c")
        tt_temp = _project(ttemp, vtemp, t.throttle_temp_c, True)
        tclk, vclk = _series(window, "clock_ratio")
        tt_clk = _project(tclk, vclk, t.throttle_clock_ratio, False)
        horizons = [h for h in (tt_temp, tt_clk) if h is not None]
        if horizons:
            horizon = min(horizons)
            why = "temperature" if tt_temp is not None and tt_temp == horizon else "SM clock"
            return _make(
                Verdict.THERMAL_THROTTLE, "warn", _conf(horizon),
                f"On track to throttle in ~{horizon:.0f}s ({why} trend) while under "
                "load. Raising the power limit / improving cooling now keeps tokens/sec.",
                [
                    f"Mean utilization: {mean_util:.0f}%",
                    *([f"Temp now: {vtemp[-1]:.0f}°C, throttle ~{t.throttle_temp_c:.0f}°C"] if vtemp else []),
                    f"Projected throttle in: ~{horizon:.0f}s",
                ],
                [
                    "Raise the power limit (nvidia-smi -pl) if the card/PSU allow.",
                    "Improve airflow / cooling, or pre-emptively lower sustained load (--parallel).",
                ],
                {**metrics, "predicted_throttle_s": round(horizon, 1)},
                predicted=True, horizon_s=horizon,
            )

    # 3) KV cache filling toward saturation (callers will start queueing).
    if llama_on:
        tk, vk = _series(window, "kv_cache_usage_ratio")
        tt_kv = _project(tk, vk, t.kv_pressure_ratio, True)
        if tt_kv is not None:
            return _make(
                Verdict.KV_CACHE_PRESSURE, "warn", _conf(tt_kv),
                f"KV cache is filling and is on track to hit {t.kv_pressure_ratio:.0%} "
                f"in ~{tt_kv:.0f}s; requests will start queueing. Scale out or tune now.",
                [f"KV now: {vk[-1]:.0%}", f"Projected saturation in: ~{tt_kv:.0f}s"],
                [
                    "Enlarge the cache (--ctx-size) or quantize it (--cache-type-k/v q8_0).",
                    "If sustained, scale out a second server before callers are deferred.",
                ],
                {**metrics, "predicted_kv_full_s": round(tt_kv, 1)},
                predicted=True, horizon_s=tt_kv,
            )

    return None


def analyze(window: list[Snapshot], thresholds: Thresholds | None = None) -> Diagnosis:
    """Return the dominant diagnosis for a recent window of snapshots."""
    t = thresholds or Thresholds()

    if len(window) < t.min_samples:
        return _make(
            Verdict.UNKNOWN,
            "info",
            0.0,
            "Collecting samples; give it a few seconds.",
            [],
            [],
            {},
        )

    llama_on = any(s.llama_reachable for s in window)
    mean_util = _mean_or(window, "util_pct", 0.0) or 0.0
    mean_mem = _mean_or(window, "mem_used_ratio")
    mean_clock = _mean_or(window, "clock_ratio")
    max_kv = max(_vals(window, "kv_cache_usage_ratio"), default=0.0)
    max_deferred = max(_vals(window, "requests_deferred"), default=0.0)
    gen_tps = _mean_or(window, "gen_tokens_per_s", 0.0) or 0.0

    # "Active" = actually serving a request. With llama metrics we know exactly
    # (requests_processing >= 1); without them we proxy from GPU utilization.
    if llama_on:
        active_ticks = sum(
            1 for s in window if (s.requests_processing or 0) >= 1
        )
    else:
        active_ticks = sum(
            1 for s in window if (s.util_pct or 0) >= t.idle_util_pct
        )
    active_frac = active_ticks / len(window)

    metrics = {
        "mean_util_pct": round(mean_util, 1),
        "mem_used_ratio": round(mean_mem, 3) if mean_mem is not None else None,
        "clock_ratio": round(mean_clock, 3) if mean_clock is not None else None,
        "max_kv_cache_ratio": round(max_kv, 3),
        "max_requests_deferred": max_deferred,
        "active_fraction": round(active_frac, 3),
        "gen_tokens_per_s": round(gen_tps, 1),
        "llama_connected": llama_on,
    }

    # --- Rule 1: THERMAL_THROTTLE; clock dragged down while under load. -----
    if (
        mean_clock is not None
        and mean_clock < t.throttle_clock_ratio
        and mean_util >= t.throttle_util_pct
    ):
        return _make(
            Verdict.THERMAL_THROTTLE,
            "crit",
            min(1.0, (t.throttle_clock_ratio - mean_clock) / t.throttle_clock_ratio + 0.4),
            f"GPU is {mean_util:.0f}% busy but SM clock is only "
            f"{mean_clock:.0%} of max; it is throttling, so you are leaving "
            "tokens/sec on the table.",
            [
                f"Mean utilization: {mean_util:.0f}%",
                f"SM clock vs max: {mean_clock:.0%}",
                *(
                    [f"Temperature: {_mean_or(window, 'temp_c'):.0f}°C"]
                    if _vals(window, "temp_c")
                    else []
                ),
            ],
            [
                "Confirm it: run  nvidia-smi -q -d TEMPERATURE,CLOCK  and look for 'HW Slowdown: Active' or SM clocks pinned below base.",
                "If it's heat: improve airflow, clean dust filters, or move the SFF box somewhere cooler. Small chassis heat-soak under sustained inference.",
                "If it's a power cap: check it with  nvidia-smi -q -d POWER , then (if the card and PSU allow) raise it with  sudo nvidia-smi -pl <watts> .",
                "If cooling can't improve, reduce sustained load: lower --parallel on llama-server so the card runs cooler.",
            ],
            metrics,
        )

    # --- Rule 2: KV_CACHE_PRESSURE; cache full and/or requests deferred. ----
    if llama_on and (max_kv >= t.kv_pressure_ratio or max_deferred >= 1):
        return _make(
            Verdict.KV_CACHE_PRESSURE,
            "warn",
            min(1.0, 0.5 + max_kv / 2 + (0.2 if max_deferred >= 1 else 0)),
            (
                f"Up to {max_deferred:.0f} request(s) were queued waiting for a free "
                f"slot (KV cache peaked at {max_kv:.0%}). More concurrent requests "
                "than the server has slots, so callers are made to wait."
                if max_deferred >= 1
                else f"KV cache peaked at {max_kv:.0%} usage; close to full."
            ),
            [
                f"Peak KV-cache usage: {max_kv:.0%}",
                f"Max requests deferred: {max_deferred:.0f}",
                f"Mean utilization: {mean_util:.0f}%",
            ],
            [
                "If the VRAM tile shows headroom, give the cache more room: restart llama-server with a larger --ctx-size (e.g. double it).",
                "Fit more context in the same VRAM by quantizing the KV cache: add  --flash-attn --cache-type-k q8_0 --cache-type-v q8_0 .",
                "If you over-committed concurrent slots, lower --parallel so requests stop exhausting the cache.",
                "Avoid recomputing shared prefixes: enable prompt caching ( --prompt-cache <file> ) and reuse it across requests.",
                "If this stays pinned at capacity, scale out: run a second llama-server instance/box behind a load balancer.",
            ],
            metrics,
        )

    # --- Predictive early-warning: catch a problem that is still FORMING -----
    # Runs after the reactive crit/warn rules (an actual throttle/pressure still
    # wins) but before the benign verdicts, so a box that currently looks healthy
    # or merely decoding is flagged when a trend projects danger soon. Remediation
    # can then act with lead time instead of after tokens/sec are already lost.
    predicted = _predict(window, t, mean_util, llama_on, metrics)
    if predicted is not None:
        return predicted

    # --- Rule 3: IDLE_NO_REQUESTS; the GPU is sitting idle. -----------------
    if active_frac < 0.10 and mean_util < t.idle_util_pct:
        idle_pct = (1 - active_frac) * 100
        head = ""
        if mean_mem is not None and mean_mem < t.mem_headroom_ratio:
            head = (
                f" The model is resident ({mean_mem:.0%} VRAM) but doing nothing; "
                "you are paying to keep a warm, empty GPU."
            )
        return _make(
            Verdict.IDLE_NO_REQUESTS,
            "info",
            min(1.0, 0.6 + (t.idle_util_pct - mean_util) / 100 + (1 - active_frac) * 0.3),
            f"GPU idle {idle_pct:.0f}% of this window; no inference requests."
            + head,
            [
                f"Mean utilization: {mean_util:.0f}%",
                f"Active fraction: {active_frac:.0%}",
                *(
                    [f"VRAM resident: {mean_mem:.0%}"]
                    if mean_mem is not None
                    else []
                ),
            ],
            [
                "This is not a misconfiguration: the GPU simply has no requests. The fix is utilization, not a flag.",
                "Reclaim the cost on demand: stop llama-server when idle and start it per request, or run it under a supervisor that unloads the model after N idle minutes.",
                "Fill the gaps: schedule batch/offline work (evals, embeddings, bulk summarization) to run during idle windows.",
                "If the box must stay warm for low latency, keep it: the report quantifies what that readiness costs so it stays a deliberate choice.",
            ],
            metrics,
        )

    # --- Rule 4: DECODE_BANDWIDTH_BOUND; generating but not saturated. ------
    decode_like = (
        active_frac >= 0.25
        and mean_util < t.busy_util_pct
        and (gen_tps > 0 or not llama_on)
    )
    if decode_like:
        conc_note = ""
        if llama_on:
            mean_conc = _mean_or(window, "requests_processing", 0.0) or 0.0
            conc_note = f" Mean concurrency was {mean_conc:.1f} request(s)."
        return _make(
            Verdict.DECODE_BANDWIDTH_BOUND,
            "info",
            min(1.0, 0.5 + (t.busy_util_pct - mean_util) / 100),
            f"Actively serving but GPU only ~{mean_util:.0f}% busy. Low-concurrency "
            "token generation is memory-bandwidth bound, so utilization plateaus "
            "below 100%." + conc_note,
            [
                f"Mean utilization: {mean_util:.0f}%",
                f"Active fraction: {active_frac:.0%}",
                *(
                    [f"Generation: {gen_tps:.0f} tok/s"]
                    if gen_tps > 0
                    else []
                ),
            ],
            [
                "Turn on continuous batching so concurrent requests share each decode step: restart llama-server with  --parallel 4  (raise as load grows) and  --cont-batching .",
                "Throughput climbs with concurrency while utilization barely moves, until you saturate memory bandwidth: that headroom is what you're leaving on the table.",
                "For lower single-stream latency, add speculative decoding with a small draft model:  --model-draft <small.gguf> --draft 16 .",
                "If decode is bandwidth-bound, a faster quant of the same model (e.g. Q4_K_M) raises tok/s.",
            ],
            metrics,
        )

    # --- Rule 5: MEMORY_HEADROOM; under-using VRAM while busy enough. -------
    if mean_mem is not None and mean_mem < t.mem_headroom_ratio and active_frac >= 0.10:
        free_pct = (1 - mean_mem) * 100
        return _make(
            Verdict.MEMORY_HEADROOM,
            "info",
            min(1.0, 0.4 + (t.mem_headroom_ratio - mean_mem)),
            f"{free_pct:.0f}% of VRAM is free while the box is in use; you have "
            "room to do more on this same card.",
            [
                f"VRAM used: {mean_mem:.0%}",
                f"Mean utilization: {mean_util:.0f}%",
            ],
            [
                "Make sure the whole model is on the GPU: increase --n-gpu-layers (use  -ngl 999  to offload all layers; CPU-offloaded layers are slow).",
                "Grow the context window for longer prompts: raise --ctx-size (e.g. 8192 to 16384).",
                "Serve more concurrent users on the free VRAM: raise --parallel.",
                "Or load a larger / higher-precision model now that there's room (less quantization loss, better quality).",
            ],
            metrics,
        )

    # --- Default: HEALTHY ----------------------------------------------------
    return _make(
        Verdict.HEALTHY,
        "ok",
        0.8,
        f"GPU is {mean_util:.0f}% utilized while serving; healthy. No idle "
        "capacity or bottleneck stands out in this window.",
        [
            f"Mean utilization: {mean_util:.0f}%",
            f"Active fraction: {active_frac:.0%}",
            *(
                [f"VRAM used: {mean_mem:.0%}"]
                if mean_mem is not None
                else []
            ),
        ],
        [],
        metrics,
    )
