from __future__ import annotations

import json
import time

from fastapi import APIRouter, Query, Request
from fastapi.responses import StreamingResponse

from fusion.risk_scoring import summarize_patient_risk
from shared.models import AlertItem, AlertResponse

router = APIRouter(prefix="/api/alerts", tags=["alerts"])


@router.get("", response_model=list[AlertItem])
def list_alerts(
    request: Request,
    patient_id: str | None = Query(default=None),
    severity: str | None = Query(default=None),
) -> list[AlertItem]:
    container = request.app.state.container
    items = container.alerts
    if patient_id:
        items = [item for item in items if item.patient_id == patient_id]
    if severity:
        items = [item for item in items if item.severity == severity]
    return items


@router.get("/patient/{patient_id}", response_model=AlertResponse)
def get_alerts_by_patient(patient_id: str, request: Request) -> AlertResponse:
    container = request.app.state.container
    patient_events = [event for event in container.events if event.patient_id == patient_id]
    patient_alerts = [alert for alert in container.alerts if alert.patient_id == patient_id]
    risk_summary = summarize_patient_risk(patient_id, patient_events, patient_alerts)
    return AlertResponse(patient_id=patient_id, alerts=patient_alerts, risk_summary=risk_summary)


@router.get("/stream")
def stream_alerts(request: Request) -> StreamingResponse:
    container = request.app.state.container

    def event_generator():
        last_index = 0
        while True:
            alerts = container.alerts
            while last_index < len(alerts):
                payload = alerts[last_index].model_dump(mode="json")
                yield f"event: alert\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
                last_index += 1
            yield "event: heartbeat\ndata: {}\n\n"
            time.sleep(1)

    return StreamingResponse(event_generator(), media_type="text/event-stream")
