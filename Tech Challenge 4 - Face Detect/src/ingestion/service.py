from __future__ import annotations

from alerts.engine import build_alert_from_event
from fusion.risk_scoring import summarize_patient_risk
from shared.audit import AuditEvent, now_unix, write_audit_event
from shared.models import MultimodalEvent, PipelineRunResponse


def record_event(container, event: MultimodalEvent, *, source: str, details: dict | None = None) -> PipelineRunResponse:
    container.events.append(event)

    generated_alert = build_alert_from_event(event, container.settings)
    if generated_alert is not None:
        container.alerts.append(generated_alert)

    patient_risk = summarize_patient_risk(event.patient_id, container.events, container.alerts)
    response_details = dict(details or {})
    response_details["source"] = source
    container.event_details[event.event_id] = response_details

    write_audit_event(
        container.settings.audit_log_path,
        AuditEvent(
            ts_unix=now_unix(),
            event_type=f"{source}_processed",
            patient_id=event.patient_id,
            modality=event.modality,
            payload={
                "event": event.model_dump(),
                "generated_alert": generated_alert.model_dump() if generated_alert else None,
                "patient_risk": patient_risk.model_dump(),
                "details": response_details,
            },
        ),
    )

    return PipelineRunResponse(
        accepted=True,
        pipeline=event.modality,
        event=event,
        generated_alert=generated_alert,
        patient_risk=patient_risk,
        details=response_details,
    )
