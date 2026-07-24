from __future__ import annotations

from uuid import uuid4

from shared.models import IngestEventRequest, MultimodalEvent

_ALLOWED_MODALITIES = {"audio", "video", "text", "vitals"}
_ALLOWED_SEVERITIES = {"info", "low", "medium", "high"}


def normalize_event(payload: IngestEventRequest) -> MultimodalEvent:
    modality = payload.modality.strip().lower()
    severity = payload.severity.strip().lower()

    if modality not in _ALLOWED_MODALITIES:
        raise ValueError("Modalidade invalida. Use audio, video, text ou vitals.")
    if severity not in _ALLOWED_SEVERITIES:
        raise ValueError("Severidade invalida. Use info, low, medium ou high.")

    event_id = payload.event_id or f"evt-{uuid4()}"
    evidence = [item.strip() for item in payload.evidence if item.strip()]

    return MultimodalEvent(
        event_id=event_id,
        patient_id=payload.patient_id.strip(),
        modality=modality,
        timestamp=payload.timestamp,
        signal=payload.signal.strip(),
        severity=severity,
        anomaly_score=payload.anomaly_score,
        evidence=evidence,
        transcript_excerpt=(payload.transcript_excerpt or "").strip() or None,
        metadata=payload.metadata,
    )
