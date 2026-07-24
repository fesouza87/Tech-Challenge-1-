from __future__ import annotations

from collections import Counter

from shared.models import AlertItem, MultimodalEvent, PatientRiskSummary


def summarize_patient_risk(
    patient_id: str,
    events: list[MultimodalEvent],
    alerts: list[AlertItem],
) -> PatientRiskSummary:
    patient_events = [event for event in events if event.patient_id == patient_id]
    patient_alerts = [alert for alert in alerts if alert.patient_id == patient_id]

    if patient_events:
        average_score = sum(event.anomaly_score for event in patient_events) / len(patient_events)
    else:
        average_score = 0.0

    severity_counter = Counter(alert.severity for alert in patient_alerts)
    active_modalities = sorted({event.modality for event in patient_events})
    latest_signal = patient_events[-1].signal if patient_events else None

    return PatientRiskSummary(
        patient_id=patient_id,
        event_count=len(patient_events),
        alert_count=len(patient_alerts),
        average_anomaly_score=round(average_score, 4),
        highest_severity=_highest_severity(severity_counter),
        active_modalities=active_modalities,
        latest_signal=latest_signal,
    )


def _highest_severity(counter: Counter[str]) -> str:
    if counter.get("high"):
        return "high"
    if counter.get("medium"):
        return "medium"
    if counter.get("low"):
        return "low"
    return "none"
