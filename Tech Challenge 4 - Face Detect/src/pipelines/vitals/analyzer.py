from __future__ import annotations

from statistics import mean
from uuid import uuid4

from shared.models import MultimodalEvent, VitalsAnalysisRequest


def analyze_vitals(payload: VitalsAnalysisRequest) -> tuple[MultimodalEvent, dict]:
    latest = payload.samples[-1]
    evidence: list[str] = []
    score = 0.0

    if latest.heart_rate is not None and (latest.heart_rate > 120 or latest.heart_rate < 45):
        score += 0.25
        evidence.append(f"heart_rate={latest.heart_rate}")
    if latest.spo2 is not None and latest.spo2 < 92:
        score += 0.35
        evidence.append(f"spo2={latest.spo2}")
    if latest.systolic_bp is not None and latest.systolic_bp < 90:
        score += 0.25
        evidence.append(f"systolic_bp={latest.systolic_bp}")
    if latest.respiratory_rate is not None and latest.respiratory_rate > 24:
        score += 0.20
        evidence.append(f"respiratory_rate={latest.respiratory_rate}")
    if latest.temperature_c is not None and latest.temperature_c >= 38.2:
        score += 0.18
        evidence.append(f"temperature_c={latest.temperature_c}")

    if len(payload.samples) >= 3:
        hr_values = [sample.heart_rate for sample in payload.samples if sample.heart_rate is not None]
        spo2_values = [sample.spo2 for sample in payload.samples if sample.spo2 is not None]
        if len(hr_values) >= 3 and hr_values[-1] - mean(hr_values[:-1]) >= 20:
            score += 0.12
            evidence.append("trend_heart_rate=rise")
        if len(spo2_values) >= 3 and mean(spo2_values[:-1]) - spo2_values[-1] >= 3:
            score += 0.12
            evidence.append("trend_spo2=drop")

    anomaly_score = min(round(score, 4), 1.0)
    signal = "vitals_sem_anomalia_relevante"
    if any(item.startswith("spo2=") for item in evidence):
        signal = "dessaturacao"
    elif any(item.startswith("heart_rate=") for item in evidence):
        signal = "instabilidade_hemodinamica"
    elif evidence:
        signal = "desvio_sinais_vitais"

    event = MultimodalEvent(
        event_id=f"vitals-{uuid4()}",
        patient_id=payload.patient_id,
        modality="vitals",
        timestamp=latest.timestamp,
        signal=signal,
        severity=_severity_from_score(anomaly_score),
        anomaly_score=anomaly_score,
        evidence=evidence or ["sem_evidencias_relevantes"],
        metadata={**payload.metadata, "pipeline": "vitals", "sample_count": len(payload.samples)},
    )
    details = {
        "latest_sample": latest.model_dump(),
        "sample_count": len(payload.samples),
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
