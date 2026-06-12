"""FastAPI app that serves the dashboard and the monitor's JSON API.

One process serves both the static dashboard (no build step, no node) and the
polling API the dashboard reads. ``create_app(monitor)`` is split out from the
launcher so tests can drive the API against a demo-backed monitor.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from et_monitor.state import Monitor

_STATIC_DIR = Path(__file__).resolve().parent / "static"


def create_app(monitor: Monitor, *, start: bool = True) -> FastAPI:
    app = FastAPI(title="ET; GPU idleness monitor", docs_url="/api/docs")

    if start:

        @app.on_event("startup")
        def _startup() -> None:
            monitor.start()

        @app.on_event("shutdown")
        def _shutdown() -> None:
            monitor.stop()

    @app.get("/api/snapshot")
    def snapshot() -> JSONResponse:
        return JSONResponse(monitor.snapshot())

    @app.get("/api/history")
    def history() -> JSONResponse:
        return JSONResponse({"samples": monitor.history()})

    @app.get("/api/diagnosis")
    def diagnosis() -> JSONResponse:
        return JSONResponse(monitor.diagnosis().to_dict())

    @app.get("/api/state")
    def state() -> JSONResponse:
        """Everything the dashboard needs in one poll."""
        return JSONResponse(
            {
                "snapshot": monitor.snapshot(),
                "diagnosis": monitor.diagnosis().to_dict(),
                "history": monitor.history(),
            }
        )

    @app.get("/api/report")
    def report() -> JSONResponse:
        return JSONResponse(monitor.report())

    @app.get("/healthz")
    def healthz() -> dict:
        return {"ok": True}

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(_STATIC_DIR / "index.html")

    @app.get("/report")
    def report_page() -> FileResponse:
        return FileResponse(_STATIC_DIR / "report.html")

    if _STATIC_DIR.is_dir():
        app.mount(
            "/static", StaticFiles(directory=str(_STATIC_DIR)), name="static"
        )

    return app
