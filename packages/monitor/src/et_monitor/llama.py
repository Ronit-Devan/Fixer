"""Scrape and parse ``llama-server``'s Prometheus ``/metrics`` endpoint.

``llama.cpp``'s server (``llama-server --metrics``) exposes counters and gauges
that tell us what the GPU is *doing* during inference; request concurrency,
KV-cache occupancy, prompt vs. generation throughput. NVML tells us the GPU is
busy; these metrics tell us *why*, the same role the PyTorch-trace events play
for the training engine.

The scraper is best-effort: if llama-server isn't running, isn't built with
metrics, or the body is malformed, ``read()`` returns ``None`` and the monitor
runs in NVML-only mode. It never raises into the sampling loop.

Reference metric names (llama.cpp examples/server):
  llamacpp:prompt_tokens_total            counter
  llamacpp:tokens_predicted_total         counter
  llamacpp:prompt_tokens_seconds          gauge  (avg prompt throughput)
  llamacpp:predicted_tokens_seconds       gauge  (avg generation throughput)
  llamacpp:kv_cache_usage_ratio           gauge  (0..1)
  llamacpp:kv_cache_tokens                gauge
  llamacpp:requests_processing            gauge  (active slots)
  llamacpp:requests_deferred              gauge  (queued, waiting for a slot)
  llamacpp:n_decode_total                 counter
We parse the whole exposition generically, then read the keys we know.
"""

from __future__ import annotations

import logging
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class LlamaMetrics:
    timestamp_s: float
    reachable: bool
    raw: dict[str, float]
    # Convenience accessors for the keys we act on (None if absent).
    prompt_tokens_total: float | None = None
    predicted_tokens_total: float | None = None
    prompt_tokens_seconds: float | None = None
    predicted_tokens_seconds: float | None = None
    kv_cache_usage_ratio: float | None = None
    kv_cache_tokens: float | None = None
    requests_processing: float | None = None
    requests_deferred: float | None = None
    decode_total: float | None = None

    @property
    def is_active(self) -> bool:
        """Is the server actively serving at least one request right now?"""
        return bool(self.requests_processing and self.requests_processing >= 1)


def parse_prometheus(text: str) -> dict[str, float]:
    """Parse Prometheus text exposition into ``{metric_name: value}``.

    Labels are stripped (we don't need per-label series for a single server);
    when the same base name appears with multiple label-sets the last one wins,
    which is fine for the single-process llama-server case. ``# HELP`` / ``# TYPE``
    comment lines and unparseable values are skipped.
    """
    out: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # ``name{labels} value [timestamp]``; split off the trailing value.
        try:
            left, value_str = line.rsplit(" ", 1)
        except ValueError:
            continue
        name = left.split("{", 1)[0].strip()
        try:
            out[name] = float(value_str)
        except ValueError:
            # Prometheus allows +Inf/-Inf/NaN; ignore those for our gauges.
            continue
    return out


def metrics_from_raw(raw: dict[str, float], *, timestamp_s: float) -> LlamaMetrics:
    g = raw.get
    return LlamaMetrics(
        timestamp_s=timestamp_s,
        reachable=True,
        raw=raw,
        prompt_tokens_total=g("llamacpp:prompt_tokens_total"),
        predicted_tokens_total=g("llamacpp:tokens_predicted_total"),
        prompt_tokens_seconds=g("llamacpp:prompt_tokens_seconds"),
        predicted_tokens_seconds=g("llamacpp:predicted_tokens_seconds"),
        kv_cache_usage_ratio=g("llamacpp:kv_cache_usage_ratio"),
        kv_cache_tokens=g("llamacpp:kv_cache_tokens"),
        requests_processing=g("llamacpp:requests_processing"),
        requests_deferred=g("llamacpp:requests_deferred"),
        decode_total=g("llamacpp:n_decode_total"),
    )


class LlamaScraper:
    """Polls ``{base_url}/metrics``. Tolerates the server being down."""

    def __init__(self, base_url: str, timeout_s: float = 2.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.metrics_url = f"{self.base_url}/metrics"
        self.timeout_s = timeout_s
        self._warned_unreachable = False

    def read(self) -> LlamaMetrics | None:
        now = time.time()
        try:
            req = urllib.request.Request(
                self.metrics_url, headers={"Accept": "text/plain"}
            )
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                body = resp.read().decode("utf-8", "replace")
        except (urllib.error.URLError, OSError, ValueError) as e:
            if not self._warned_unreachable:
                log.info(
                    "llama-server /metrics not reachable at %s (%s); "
                    "running in GPU-only mode. Start it with --metrics to enable "
                    "inference attribution.",
                    self.metrics_url,
                    e,
                )
                self._warned_unreachable = True
            return None
        self._warned_unreachable = False
        raw = parse_prometheus(body)
        if not raw:
            return None
        return metrics_from_raw(raw, timestamp_s=now)
