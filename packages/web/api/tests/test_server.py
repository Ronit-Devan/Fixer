import os
from pathlib import Path

from fastapi.testclient import TestClient

import et_api.server as server_mod
from et_api.server import app

FIXTURES = Path(__file__).resolve().parents[4] / "fixtures"

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_analyze_dataloader_starved():
    fixture = FIXTURES / "dataloader_starved.json"
    with fixture.open("rb") as f:
        r = client.post("/analyze", files={"file": (fixture.name, f, "application/json")})
    assert r.status_code == 200
    body = r.json()
    assert body["verdict"] == "dataloader_bound"
    assert body["confidence"] > 0.5
    assert "decisions" in body["stats"]


def test_analyze_rejects_non_json():
    r = client.post("/analyze", files={"file": ("bad.txt", b"hello", "text/plain")})
    assert r.status_code == 400


def test_analyze_invalid_json_body_returns_400():
    """A .json file whose body is not valid JSON is a client error, not a 500.

    Regression test: load_trace raises json.JSONDecodeError, which used to be
    swallowed by the blanket `except Exception` -> HTTP 500 "Engine error".
    """
    r = client.post(
        "/analyze",
        files={"file": ("broken.json", b"not json {{{", "application/json")},
    )
    assert r.status_code == 400
    assert "valid" in r.json()["detail"].lower()


def test_analyze_valid_json_not_a_trace_returns_400():
    """Valid JSON that is not a trace object -> 400.

    A non-dict JSON document (here a JSON array) is well-formed JSON but not a
    PyTorch Profiler trace; load_trace raises while trying to read traceEvents.
    That is a client-input error, so the endpoint returns 400, not 500.

    (Note: a JSON *object* lacking traceEvents — e.g. {"foo": 1} — is handled
    separately: load_trace returns an empty Trace and the request succeeds as
    200 UNKNOWN. See test_analyze_empty_trace_object_is_200_unknown.)
    """
    r = client.post(
        "/analyze",
        files={"file": ("notatrace.json", b"[1, 2, 3]", "application/json")},
    )
    assert r.status_code == 400
    assert "valid" in r.json()["detail"].lower()


def test_analyze_empty_trace_object_is_200_unknown():
    """A structurally valid JSON object with no events stays a 200 success.

    This pins the contract the 400 fix must NOT break: load_trace returns an
    empty Trace for an object without traceEvents (the same path a legitimately
    empty real trace takes), and the engine reports UNKNOWN. It must remain a
    200, never become a 400.
    """
    r = client.post(
        "/analyze",
        files={"file": ("empty.json", b'{"foo": 1}', "application/json")},
    )
    assert r.status_code == 200
    assert r.json()["verdict"] == "unknown"


def test_analyze_400_does_not_leak_temp_file(monkeypatch):
    """The temp file written for the upload is removed even on the 400 path."""
    created: list[str] = []
    real_named_tmp = server_mod.tempfile.NamedTemporaryFile

    def _spy(*args, **kwargs):
        handle = real_named_tmp(*args, **kwargs)
        created.append(handle.name)
        return handle

    monkeypatch.setattr(server_mod.tempfile, "NamedTemporaryFile", _spy)

    r = client.post(
        "/analyze",
        files={"file": ("broken.json", b"not json {{{", "application/json")},
    )
    assert r.status_code == 400
    assert created, "expected the handler to create a temp file"
    for path in created:
        assert not os.path.exists(path), f"temp file leaked on 400 path: {path}"


def test_index_serves_ui():
    r = client.get("/")
    assert r.status_code == 200
    assert "ET" in r.text
    assert "dropzone" in r.text


def test_static_assets():
    r = client.get("/static/app.js")
    assert r.status_code == 200
    assert "analyzeFile" in r.text