from __future__ import annotations

from shared.config import Settings
from shared.models import AlertItem, MultimodalEvent


def build_alert_from_event(event: MultimodalEvent, settings: Settings) -> AlertItem | None:
    score = event.anomaly_score
    if score < settings.alert_low_threshold:
        return None

    if score >= settings.alert_high_threshold:
        severity = "high"
    elif score >= settings.alert_medium_threshold:
        severity = "medium"
    else:
        severity = "low"

    evidence = list(event.evidence)
    if event.transcript_excerpt:
        evidence.append(f"transcript_excerpt={event.transcript_excerpt[:120]}")

    return AlertItem(
        alert_id=f"alert-{event.event_id}",
        patient_id=event.patient_id,
        event_id=event.event_id,
        modality=event.modality,
        severity=severity,
        title=f"Anomalia {event.modality} detectada",
        message=(
            f"Evento {event.signal} identificado para o paciente {event.patient_id} "
            f"com score {event.anomaly_score:.2f} e severidade {severity}."
        ),
        evidence=evidence,
        anomaly_score=event.anomaly_score,
        recommended_action=_recommended_action(event.modality, severity),
        created_at=event.timestamp,
    )


def _recommended_action(modality: str, severity: str) -> str:
    base = {
        "audio": "Revisar consulta, transcricao e sinais vocais.",
        "video": "Inspecionar trecho do video e validar o procedimento.",
        "text": "Revisar evolucao clinica e prescricoes relacionadas.",
        "vitals": "Conferir sinais vitais, contexto assistencial e tendencia temporal.",
    }.get(modality, "Validar o evento com a equipe clinica.")
    if severity == "high":
        return base + " Acionar avaliacao humana imediata."
    return base
