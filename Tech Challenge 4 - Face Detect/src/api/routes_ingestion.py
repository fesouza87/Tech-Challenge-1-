from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from alerts.engine import build_alert_from_event
from fusion.risk_scoring import summarize_patient_risk
from ingestion.normalizers import normalize_event
from shared.audit import AuditEvent, now_unix, write_audit_event
from shared.models import AlertResponse, IngestEventRequest, IngestEventResponse

router = APIRouter(prefix="/api/events", tags=["events"])


@router.post("", response_model=IngestEventResponse)
def ingest_event(payload: IngestEventRequest, request: Request) -> IngestEventResponse:
    container = request.app.state.container

    try:
        event = normalize_event(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    container.events.append(event)

    generated_alert = build_alert_from_event(event, container.settings)
    if generated_alert is not None:
        container.alerts.append(generated_alert)

    risk_summary = summarize_patient_risk(event.patient_id, container.events, container.alerts)

    write_audit_event(
        container.settings.audit_log_path,
        AuditEvent(
            ts_unix=now_unix(),
            event_type="event_ingested",
            patient_id=event.patient_id,
            modality=event.modality,
            payload={
                "event_id": event.event_id,
                "severity": event.severity,
                "anomaly_score": event.anomaly_score,
                "signal": event.signal,
                "risk_summary": risk_summary.model_dump(),
                "alert_generated": generated_alert.model_dump() if generated_alert else None,
            },
        ),
    )

    return IngestEventResponse(
        accepted=True,
        event=event,
        generated_alert=generated_alert,
        patient_risk=risk_summary,
    )


@router.get("/patient/{patient_id}/summary", response_model=AlertResponse)
def patient_summary(patient_id: str, request: Request) -> AlertResponse:
    container = request.app.state.container
    patient_events = [event for event in container.events if event.patient_id == patient_id]
    if not patient_events:
        raise HTTPException(status_code=404, detail="Paciente sem eventos registrados.")

    patient_alerts = [alert for alert in container.alerts if alert.patient_id == patient_id]
    risk_summary = summarize_patient_risk(patient_id, patient_events, patient_alerts)
    return AlertResponse(patient_id=patient_id, alerts=patient_alerts, risk_summary=risk_summary)
