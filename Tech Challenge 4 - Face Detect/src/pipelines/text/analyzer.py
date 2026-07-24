from __future__ import annotations

from uuid import uuid4

from shared.models import MultimodalEvent, TextAnalysisRequest

_CRITICAL_TERMS = (
    "sepse",
    "choque",
    "hemorragia",
    "parada",
    "dessaturacao",
    "hipotensao",
    "hipotensão",
)
_PRESCRIPTION_CHANGE_TERMS = (
    "suspender",
    "aumentar dose",
    "reduzir dose",
    "trocar antibiotico",
    "trocar antibiótico",
    "interromper tratamento",
)


def analyze_text(payload: TextAnalysisRequest) -> tuple[MultimodalEvent, dict]:
    note = payload.clinical_note.lower()
    prescription = (payload.prescription_text or "").lower()

    evidence: list[str] = []
    score = 0.0
    critical_hits = [term for term in _CRITICAL_TERMS if term in note or term in prescription]
    change_hits = [term for term in _PRESCRIPTION_CHANGE_TERMS if term in prescription]

    if critical_hits:
        score += 0.42
        evidence.append(f"termos_criticos={', '.join(critical_hits)}")
    if change_hits:
        score += 0.28
        evidence.append(f"alteracoes_prescricao={', '.join(change_hits)}")
    if "nao responsivo" in note or "não responsivo" in note:
        score += 0.22
        evidence.append("estado_clinico=nao_responsivo")
    if "piora" in note:
        score += 0.14
        evidence.append("evolucao=piora")

    anomaly_score = min(round(score, 4), 1.0)
    signal = "texto_sem_anomalia_relevante"
    if critical_hits:
        signal = "termos_criticos_em_evolucao"
    elif change_hits:
        signal = "alteracao_inesperada_prescricao"

    event = MultimodalEvent(
        event_id=f"text-{uuid4()}",
        patient_id=payload.patient_id,
        modality="text",
        timestamp=payload.timestamp,
        signal=signal,
        severity=_severity_from_score(anomaly_score),
        anomaly_score=anomaly_score,
        evidence=evidence or ["sem_evidencias_relevantes"],
        transcript_excerpt=(payload.prescription_text or payload.clinical_note)[:280],
        metadata={**payload.metadata, "pipeline": "text"},
    )
    details = {
        "critical_terms": critical_hits,
        "prescription_change_terms": change_hits,
        "clinical_note_length": len(payload.clinical_note),
    }
    return event, details


def _severity_from_score(score: float) -> str:
    if score >= 0.8:
        return "high"
    if score >= 0.55:
        return "medium"
    if score >= 0.25:
        return "low"
    return "info"
