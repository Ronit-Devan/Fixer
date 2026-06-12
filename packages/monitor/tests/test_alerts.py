"""Alert state machine; the debounce/cooldown logic that prevents spam."""

from __future__ import annotations

from et_monitor.alerts import (
    AlertConfig,
    AlertManager,
    AlertPayload,
    format_text,
)
from et_monitor.types import Diagnosis, Verdict


class Recorder:
    def __init__(self) -> None:
        self.sent: list[AlertPayload] = []

    def send(self, payload: AlertPayload) -> None:
        self.sent.append(payload)


def diag(verdict: Verdict, severity: str, summary="s", recs=None) -> Diagnosis:
    return Diagnosis(
        verdict=verdict,
        title=verdict.value.replace("_", " "),
        severity=severity,
        confidence=0.9,
        summary=summary,
        evidence=[],
        recommendations=recs if recs is not None else ["do the thing"],
        metrics={},
    )


KV = diag(Verdict.KV_CACHE_PRESSURE, "warn")
THROTTLE = diag(Verdict.THERMAL_THROTTLE, "crit")
HEALTHY = diag(Verdict.HEALTHY, "ok")
IDLE = diag(Verdict.IDLE_NO_REQUESTS, "info")


def _mgr(**cfg):
    rec = Recorder()
    return AlertManager([rec], AlertConfig(**cfg)), rec


def test_warn_fires_once_on_entry():
    mgr, rec = _mgr()
    assert mgr.observe(KV, 0) is not None      # entry -> alert
    assert mgr.observe(KV, 1) is None           # still warn -> silent
    assert mgr.observe(KV, 2) is None
    assert len(rec.sent) == 1
    assert rec.sent[0].kind == "alert"


def test_recovery_fires_on_return_to_healthy():
    mgr, rec = _mgr()
    mgr.observe(KV, 0)
    p = mgr.observe(HEALTHY, 5)
    assert p is not None and p.kind == "recovery"
    # back to healthy again -> no duplicate recovery
    assert mgr.observe(HEALTHY, 6) is None


def test_recovery_can_be_disabled():
    mgr, rec = _mgr(notify_recovery=False)
    mgr.observe(KV, 0)
    assert mgr.observe(HEALTHY, 5) is None
    assert len(rec.sent) == 1  # only the original alert


def test_idle_only_alerts_after_sustained_duration():
    mgr, rec = _mgr(idle_alert_after_s=600)
    assert mgr.observe(IDLE, 0) is None      # just became idle
    assert mgr.observe(IDLE, 300) is None    # 5 min; not yet
    assert mgr.observe(IDLE, 600) is not None  # 10 min; alert
    assert len(rec.sent) == 1


def test_cooldown_blocks_rapid_repeat_episodes():
    mgr, rec = _mgr(cooldown_s=900)
    assert mgr.observe(KV, 0) is not None     # episode 1
    mgr.observe(HEALTHY, 10)                   # recovers (ends episode)
    assert mgr.observe(KV, 20) is None         # new KV within cooldown -> silent
    assert mgr.observe(HEALTHY, 30) is not None  # recovery still works
    assert mgr.observe(KV, 1000) is not None   # past cooldown -> alerts again


def test_severity_escalation_fires_again():
    mgr, rec = _mgr()
    assert mgr.observe(KV, 0) is not None        # warn
    p = mgr.observe(THROTTLE, 1)                   # escalates to crit
    assert p is not None and p.diagnosis.verdict == Verdict.THERMAL_THROTTLE


def test_gpu_name_threaded_into_payload():
    mgr, rec = _mgr()
    mgr.observe(KV, 0, gpu_name="Blackwell-01")
    assert rec.sent[0].gpu_name == "Blackwell-01"


def test_format_text_alert_and_recovery():
    alert = AlertPayload("alert", KV, "GPU-X", "sf-box")
    txt = format_text(alert)
    assert "sf-box" in txt and "GPU-X" in txt and KV.summary in txt
    rec = AlertPayload("recovery", HEALTHY, "GPU-X", "sf-box")
    assert "Recovered" in format_text(rec)
