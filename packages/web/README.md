# ET Web UI

Minimal localhost trace diagnosis UI: **FastAPI** backend + **vanilla JS** frontend (no React, no build step).

## Run

```bash
cd packages/web/api
uv run et-api
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000), drag-and-drop a PyTorch Profiler `.json` trace, and view the engine verdict with evidence, metrics, recommended actions, and a collapsible per-rule decision log (`--explain`).

## Layout

- `static/` — `index.html`, `style.css`, `app.js`
- `api/` — FastAPI app (`POST /analyze`, `GET /health`, serves UI at `/`)

The Next.js app under `app/` is optional and not required for this UI.
