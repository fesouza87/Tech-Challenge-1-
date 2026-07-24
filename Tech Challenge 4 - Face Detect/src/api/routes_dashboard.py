from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from fusion.risk_scoring import summarize_patient_risk

router = APIRouter(tags=["dashboard"])


@router.get("/")
def dashboard_home() -> FileResponse:
    static_dir = Path(__file__).resolve().parents[1] / "static"
    return FileResponse(static_dir / "index.html")


@router.get("/api/dashboard/demo-audio")
def dashboard_demo_audio() -> FileResponse:
    demo_audio_path = Path(__file__).resolve().parents[2] / "data" / "synthetic" / "media" / "speech_demo_en.wav"
    if not demo_audio_path.exists():
        raise HTTPException(status_code=404, detail="Arquivo demo de audio nao encontrado.")
    return FileResponse(demo_audio_path, media_type="audio/wav", filename=demo_audio_path.name)


@router.get("/api/dashboard/overview")
def dashboard_overview(request: Request) -> dict:
    container = request.app.state.container
    events = container.events
    alerts = container.alerts

    patient_ids: list[str] = []
    seen: set[str] = set()
    for event in reversed(events):
        if event.patient_id not in seen:
            seen.add(event.patient_id)
            patient_ids.append(event.patient_id)

    patients = [
        summarize_patient_risk(patient_id, events, alerts).model_dump(mode="json")
        for patient_id in patient_ids
    ]

    recent_events = [
        event.model_dump(mode="json")
        for event in sorted(events, key=lambda item: item.timestamp, reverse=True)[:12]
    ]
    recent_alerts = [
        alert.model_dump(mode="json")
        for alert in sorted(alerts, key=lambda item: item.created_at, reverse=True)[:12]
    ]

    modality_counts: dict[str, int] = defaultdict(int)
    for event in events:
        modality_counts[event.modality] += 1

    return {
        "stats": {
            "patient_count": len(patient_ids),
            "event_count": len(events),
            "alert_count": len(alerts),
            "high_alert_count": sum(1 for alert in alerts if alert.severity == "high"),
        },
        "modality_counts": dict(modality_counts),
        "patients": patients,
        "recent_events": recent_events,
        "recent_alerts": recent_alerts,
    }


@router.get("/api/dashboard/patient/{patient_id}")
def dashboard_patient(patient_id: str, request: Request) -> dict:
    container = request.app.state.container
    patient_event_objects = sorted(
        [event for event in container.events if event.patient_id == patient_id],
        key=lambda item: item.timestamp,
        reverse=True,
    )
    patient_events = [event.model_dump(mode="json") for event in patient_event_objects]
    patient_alerts = [
        alert.model_dump(mode="json")
        for alert in sorted(
            [alert for alert in container.alerts if alert.patient_id == patient_id],
            key=lambda item: item.created_at,
            reverse=True,
        )
    ]
    patient_risk = summarize_patient_risk(patient_id, container.events, container.alerts).model_dump(mode="json")
    latest_vitals_event = next((event for event in patient_event_objects if event.modality == "vitals"), None)
    latest_vitals = None
    if latest_vitals_event is not None:
        latest_vitals = {
            "event": latest_vitals_event.model_dump(mode="json"),
            "details": container.event_details.get(latest_vitals_event.event_id, {}),
        }
    return {
        "patient_id": patient_id,
        "risk_summary": patient_risk,
        "events": patient_events,
        "alerts": patient_alerts,
        "latest_vitals": latest_vitals,
    }
