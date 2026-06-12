"""API surface, driven by the deterministic demo timeline."""

from __future__ import annotations

from fastapi.testclient import TestClient

from et_monitor.demo import DemoGpuSampler, DemoLlamaScraper, DemoTimeline
from et_monitor.server import create_app
from et_monitor.state import Monitor, MonitorConfig


def _client() -> tuple[TestClient, Monitor]:
    tl = DemoTimeline()
    mon = Monitor(
        DemoGpuSampler(tl),
        DemoLlamaScraper(tl),  # type: ignore[arg-type]
        MonitorConfig(interval_s=1.0, gpu_hourly_usd=0.5),
    )
    # Don't start the background thread in tests; tick manually.
    for _ in range(6):
        mon.tick()
    app = create_app(mon, start=False)
    return TestClient(app), mon


def test_healthz():
    client, _ = _client()
    assert client.get("/healthz").json() == {"ok": True}


def test_index_served():
    client, _ = _client()
    r = client.get("/")
    assert r.status_code == 200
    assert "Inference Monitor" in r.text


def test_snapshot_endpoint():
    client, _ = _client()
    body = client.get("/api/snapshot").json()
    assert body["backend"] == "demo"
    assert body["latest"]["llama_reachable"] is True
    assert body["session"]["gpu_hourly_usd"] == 0.5


def test_diagnosis_endpoint():
    client, _ = _client()
    d = client.get("/api/diagnosis").json()
    assert d["verdict"]
    assert d["title"]
    assert "severity" in d


def test_state_endpoint_bundles_everything():
    client, _ = _client()
    body = client.get("/api/state").json()
    assert "snapshot" in body and "diagnosis" in body and "history" in body
    assert len(body["history"]) == 6


def test_report_endpoint():
    client, _ = _client()
    r = client.get("/api/report").json()
    assert r["uptime_s"] > 0
    assert "idle_fraction" in r
    assert "projected_monthly_idle_usd" in r
    assert isinstance(r["verdict_breakdown"], list)
    assert r["verdict_breakdown"], "expected at least one verdict tracked"


def test_report_page_served():
    client, _ = _client()
    r = client.get("/report")
    assert r.status_code == 200
    assert "Utilization Report" in r.text


def test_history_carries_inference_fields():
    # The history schema must always expose the inference fields the dashboard
    # charts read (value may be None until rates are derivable; that's tested
    # deterministically in test_state._rate). Here we assert the contract.
    client, _ = _client()
    samples = client.get("/api/history").json()["samples"]
    assert samples
    for s in samples:
        assert "gen_tokens_per_s" in s
        assert "kv_cache_usage_ratio" in s
        assert "requests_processing" in s
