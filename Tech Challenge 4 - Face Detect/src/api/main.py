from __future__ import annotations

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from api.routes_alerts import router as alerts_router
from api.routes_dashboard import router as dashboard_router
from api.routes_health import router as health_router
from api.routes_ingestion import router as ingestion_router
from api.routes_pipelines import router as pipelines_router
from shared.config import load_settings
from shared.state import AppState

load_dotenv()


def create_app() -> FastAPI:
    settings = load_settings()
    state = AppState(settings=settings)

    app = FastAPI(
        title="Tech Challenge 4 - Face Detect",
        version="0.1.0",
        description="Base inicial para monitoramento clinico multimodal e deteccao de anomalias.",
    )
    app.state.container = state
    static_dir = __import__("pathlib").Path(__file__).resolve().parents[1] / "static"
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    app.include_router(dashboard_router)
    app.include_router(health_router)
    app.include_router(ingestion_router)
    app.include_router(alerts_router)
    app.include_router(pipelines_router)
    return app


app = create_app()
