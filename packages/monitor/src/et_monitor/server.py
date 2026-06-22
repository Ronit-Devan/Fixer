"""FastAPI app that serves the dashboard and the monitor's JSON API.

One process serves both the static dashboard (no build step, no node) and the
polling API the dashboard reads. ``create_app(monitor)`` is split out from the
launcher so tests can drive the API against a demo-backed monitor.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import Body, FastAPI, HTTPException
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

    # -- remediation layer (only mounted when a manager is attached) ---------
    # Works for a single GPU OR a multi-GPU box: every endpoint spans all
    # per-GPU managers (one per card), so the kill-switch, audit, and approvals
    # cover the whole box.

    def _managers() -> dict:
        mgrs = monitor.remediation_managers()
        if not mgrs and getattr(monitor, "remediation_factory", None) is None and \
                getattr(monitor, "remediation_manager", None) is None:
            raise HTTPException(status_code=404, detail="remediation layer not enabled")
        return mgrs

    @app.get("/api/remediation/state")
    def remediation_state() -> JSONResponse:
        enabled = (
            getattr(monitor, "remediation_manager", None) is not None
            or getattr(monitor, "remediation_factory", None) is not None
        )
        if not enabled:
            return JSONResponse({"enabled": False})
        mgrs = monitor.remediation_managers()
        per_gpu = [{"gpu": i, **m.status()} for i, m in sorted(mgrs.items())]
        # Fall back to the persisted box-wide mode when no GPU has been sampled yet.
        mode = monitor.remediation_mode_value()
        return JSONResponse({"enabled": True, "mode": mode, "gpus": per_gpu})

    @app.get("/api/remediation/audit")
    def remediation_audit() -> JSONResponse:
        records: list = []
        for i, m in sorted(_managers().items()):
            for rec in m.audit.as_dicts(limit=200):
                records.append({"gpu": i, **rec})
        records.sort(key=lambda r: r.get("ts", 0))
        return JSONResponse({"records": records[-200:]})

    @app.post("/api/remediation/mode")
    def remediation_mode(payload: dict = Body(...)) -> JSONResponse:
        """The kill-switch over HTTP: set off / advise / dry_run / auto (box-wide)."""
        from et_remediation import RemediationMode

        # Only require the layer to be ENABLED (factory or manager), not that a
        # manager already exists — a flip during startup must still be honored.
        enabled = (
            getattr(monitor, "remediation_manager", None) is not None
            or getattr(monitor, "remediation_factory", None) is not None
        )
        if not enabled:
            raise HTTPException(status_code=404, detail="remediation layer not enabled")
        try:
            mode = RemediationMode(str(payload.get("mode", "")))
        except ValueError:
            raise HTTPException(status_code=400, detail="invalid mode")
        # Persisted on the Monitor so even managers built later inherit it.
        monitor.set_remediation_mode(mode)
        return JSONResponse({"mode": mode.value})

    @app.post("/api/remediation/approvals/{approval_id}/approve")
    def remediation_approve(approval_id: str) -> JSONResponse:
        for m in _managers().values():
            if approval_id in getattr(m, "approvals", {}):
                out = m.approve(approval_id)
                return JSONResponse({"outcome": out.kind.value, "detail": out.detail})
        raise HTTPException(status_code=404, detail="no such approval")

    @app.post("/api/remediation/approvals/{approval_id}/reject")
    def remediation_reject(approval_id: str) -> JSONResponse:
        for m in _managers().values():
            if approval_id in getattr(m, "approvals", {}):
                if m.reject(approval_id):
                    return JSONResponse({"rejected": approval_id})
        raise HTTPException(status_code=404, detail="no such pending approval")

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
