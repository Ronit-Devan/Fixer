"""Alerting; notify Slack / a generic webhook when the GPU needs attention.

Design goals:
  * **No spam.** Alert once when the box *enters* an alert-worthy state, not
    every second it stays there. Optionally send one "recovered" message when
    it returns to healthy. A cooldown caps how often a fresh episode can re-fire.
  * **Idle is special.** Idle is only "info" severity, but a GPU idle for 20
    minutes is exactly what this product exists to catch; so idle alerts after
    a configurable sustained duration, separate from the severity rule.
  * **Best-effort.** A webhook that's down or slow must never stall or crash the
    sampling loop. Sends run on a daemon thread and swallow errors (logged).

The ``AlertManager`` is a pure state machine driven by ``observe(diagnosis,
now_s)``; the actual HTTP lives in ``Notifier`` implementations so the machine
is trivially testable with a recording notifier.
"""

from __future__ import annotations

import json
import logging
import threading
import urllib.request
from dataclasses import dataclass, field
from typing import Protocol

from et_monitor.types import Diagnosis, Verdict

log = logging.getLogger(__name__)

# Severities that alert on sight (the moment they appear).
_DEFAULT_ALERT_SEVERITIES: frozenset[str] = frozenset({"warn", "crit"})
_SEVERITY_EMOJI = {"ok": "✅", "info": "ℹ️", "warn": "⚠️", "crit": "🚨"}


@dataclass
class AlertConfig:
    alert_severities: frozenset[str] = _DEFAULT_ALERT_SEVERITIES
    idle_alert_after_s: float = 600.0  # 10 min sustained idle -> alert
    cooldown_s: float = 900.0  # min gap between fresh episodes of the same verdict
    notify_recovery: bool = True
    host_label: str = ""  # shown in messages so multi-box users can tell them apart


@dataclass
class AlertPayload:
    kind: str  # "alert" | "recovery"
    diagnosis: Diagnosis
    gpu_name: str
    host_label: str
    context: dict = field(default_factory=dict)


class Notifier(Protocol):
    def send(self, payload: AlertPayload) -> None: ...


# ---------------------------------------------------------------------------
# Message formatting (shared by all notifiers)
# ---------------------------------------------------------------------------


def format_text(payload: AlertPayload) -> str:
    d = payload.diagnosis
    where = f" · {payload.host_label}" if payload.host_label else ""
    if payload.kind == "recovery":
        return f"✅ *Recovered*{where}; GPU back to healthy. {d.summary}"
    emoji = _SEVERITY_EMOJI.get(d.severity, "•")
    lines = [
        f"{emoji} *{d.title}*{where}",
        f"_{payload.gpu_name}_",
        d.summary,
    ]
    if d.recommendations:
        lines.append(f"→ {d.recommendations[0]}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Notifiers
# ---------------------------------------------------------------------------


def _post_json(url: str, body: dict, timeout: float = 5.0) -> None:
    """POST JSON on a daemon thread; never raises into the caller."""

    def _do() -> None:
        try:
            data = json.dumps(body).encode("utf-8")
            req = urllib.request.Request(
                url, data=data, headers={"Content-Type": "application/json"}
            )
            urllib.request.urlopen(req, timeout=timeout).close()
        except Exception as e:  # noqa: BLE001
            log.warning("alert POST to %s failed: %s", _redact(url), e)

    threading.Thread(target=_do, name="et-alert", daemon=True).start()


def _redact(url: str) -> str:
    """Don't log full webhook URLs (they're secrets)."""
    if "://" not in url:
        return url
    scheme, rest = url.split("://", 1)
    host = rest.split("/", 1)[0]
    return f"{scheme}://{host}/…"


class SlackNotifier:
    """Posts to a Slack Incoming Webhook (`{"text": ...}`)."""

    def __init__(self, webhook_url: str) -> None:
        self.url = webhook_url

    def send(self, payload: AlertPayload) -> None:
        _post_json(self.url, {"text": format_text(payload)})


class WebhookNotifier:
    """Posts the full diagnosis as JSON to a generic webhook (PagerDuty, n8n, …)."""

    def __init__(self, webhook_url: str) -> None:
        self.url = webhook_url

    def send(self, payload: AlertPayload) -> None:
        _post_json(
            self.url,
            {
                "kind": payload.kind,
                "host": payload.host_label,
                "gpu": payload.gpu_name,
                "text": format_text(payload),
                "diagnosis": payload.diagnosis.to_dict(),
                "context": payload.context,
            },
        )


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------


class AlertManager:
    """Decides *when* to notify. Pure logic; HTTP is delegated to notifiers."""

    def __init__(
        self, notifiers: list[Notifier], config: AlertConfig | None = None
    ) -> None:
        self.notifiers = notifiers
        self.cfg = config or AlertConfig()
        self._verdict: Verdict | None = None
        self._verdict_since: float = 0.0
        # The verdict of the currently-active (already-notified) episode, or None.
        self._active_alert: Verdict | None = None
        self._last_sent_at: dict[Verdict, float] = {}

    def _is_alerting(self, d: Diagnosis, duration_s: float) -> bool:
        if d.severity in self.cfg.alert_severities:
            return True
        if d.verdict == Verdict.IDLE_NO_REQUESTS and duration_s >= self.cfg.idle_alert_after_s:
            return True
        return False

    def observe(
        self, d: Diagnosis, now_s: float, gpu_name: str = "GPU"
    ) -> AlertPayload | None:
        """Feed one diagnosis. Returns the payload sent (for tests), or None."""
        # Track how long the current verdict has held.
        if d.verdict != self._verdict:
            self._verdict = d.verdict
            self._verdict_since = now_s
        duration = now_s - self._verdict_since

        alerting = self._is_alerting(d, duration)

        if alerting:
            # New episode if we weren't alerting, or the verdict changed to a
            # different alert-worthy verdict. Respect per-verdict cooldown.
            if self._active_alert != d.verdict:
                last = self._last_sent_at.get(d.verdict, float("-inf"))
                if now_s - last >= self.cfg.cooldown_s or self._active_alert is not None:
                    payload = AlertPayload(
                        kind="alert",
                        diagnosis=d,
                        gpu_name=gpu_name,
                        host_label=self.cfg.host_label,
                        context={"verdict_duration_s": round(duration, 1)},
                    )
                    self._dispatch(payload)
                    self._active_alert = d.verdict
                    self._last_sent_at[d.verdict] = now_s
                    return payload
                # within cooldown for this verdict and not switching from another
                # active alert: stay quiet, but mark active so recovery still fires.
                self._active_alert = d.verdict
            return None

        # Not alerting now. If we were, the episode ended -> maybe send recovery.
        if self._active_alert is not None:
            self._active_alert = None
            if self.cfg.notify_recovery and d.severity == "ok":
                payload = AlertPayload(
                    kind="recovery",
                    diagnosis=d,
                    gpu_name=gpu_name,
                    host_label=self.cfg.host_label,
                )
                self._dispatch(payload)
                return payload
        return None

    def _dispatch(self, payload: AlertPayload) -> None:
        for n in self.notifiers:
            try:
                n.send(payload)
            except Exception:  # noqa: BLE001
                log.exception("notifier %s failed", type(n).__name__)
