"""ET web API — FastAPI + static trace diagnosis UI."""


def main() -> None:
    import uvicorn

    uvicorn.run(
        "et_api.server:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )
