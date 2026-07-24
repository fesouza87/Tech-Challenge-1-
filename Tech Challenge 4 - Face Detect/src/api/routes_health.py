from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter(tags=["health"])


@router.get("/health")
def health(request: Request) -> dict[str, object]:
    container = request.app.state.container
    return {
        "ok": True,
        "app": "tech-challenge-4-face-detect",
        "env": container.settings.app_env,
        "alerts_cached": len(container.alerts),
        "events_cached": len(container.events),
    }
