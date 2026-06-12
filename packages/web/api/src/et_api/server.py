"""FastAPI wrapper exposing the ET engine over HTTP and a localhost static UI."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from gpu_doctor_engine import __version__, load_trace
from gpu_doctor_engine.diagnose import diagnose_with_stats

# packages/web/static — sibling of packages/web/api
STATIC_DIR = Path(__file__).resolve().parents[3] / "static"

app = FastAPI(title="ET", version=__version__)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8000", "http://localhost:8000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB

if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index() -> FileResponse:
    """Single-page trace upload UI (vanilla JS, no build step)."""
    index_path = STATIC_DIR / "index.html"
    if not index_path.is_file():
        raise HTTPException(status_code=404, detail="UI not found; expected packages/web/static/")
    return FileResponse(index_path)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "engine_version": __version__}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)) -> dict:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    if not (file.filename.endswith(".json") or file.filename.endswith(".json.gz")):
        raise HTTPException(
            status_code=400,
            detail="File must be a .json or .json.gz PyTorch Profiler trace",
        )

    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large (max {MAX_UPLOAD_BYTES // (1024 * 1024)} MB)",
        )

    suffix = ".json.gz" if file.filename.endswith(".json.gz") else ".json"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(raw)
        tmp_path = tmp.name

    try:
        try:
            trace = load_trace(tmp_path)
        except (json.JSONDecodeError, ValueError, AttributeError, TypeError) as e:
            # The uploaded bytes did not load into a usable trace: invalid JSON,
            # or valid JSON that is not a PyTorch Profiler trace object. That is
            # a CLIENT-input error (like a bad file extension, which already
            # returns 400), not an engine failure — so report 400, not 500.
            # NOTE: a structurally valid but empty trace (e.g. {"foo": 1} or
            # {"traceEvents": []}) does NOT raise here; load_trace returns an
            # empty Trace and the request still succeeds as 200 UNKNOWN. That
            # 200 contract is intentionally preserved.
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is not a valid PyTorch profiler trace",
            ) from e
        diag, stats = diagnose_with_stats(trace)
    except HTTPException:
        # A 400 raised by the load step above. Pass it through unchanged — do
        # not let the blanket handler below re-wrap a client error as a 500.
        raise
    except Exception as e:
        # A genuine engine failure (e.g. a bug in diagnose() on a trace that
        # loaded fine) must still surface as 500, never be masked as a 400.
        raise HTTPException(status_code=500, detail=f"Engine error: {e}") from e
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return {
        "verdict": diag.verdict.value,
        "confidence": diag.confidence,
        "summary": diag.summary,
        "evidence": diag.evidence,
        "recommended_actions": diag.recommended_actions,
        "metrics": diag.metrics,
        "stats": stats,
        "trace_info": {
            "event_count": len(trace.events),
            "duration_ms": trace.duration_us / 1000,
            "filename": file.filename,
        },
    }
