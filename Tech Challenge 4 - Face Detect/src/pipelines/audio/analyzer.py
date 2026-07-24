from __future__ import annotations

import re
from uuid import uuid4

from pipelines.audio.azure_clients import analyze_text_with_azure, get_azure_availability, transcribe_audio_with_azure
from shared.config import Settings
from shared.models import AudioAnalysisRequest, MultimodalEvent

_FATIGUE_TERMS = ("cansaco", "cansaço", "fadiga", "voz fraca", "fraqueza")
_BREATHING_TERMS = ("falta de ar", "dispneia", "dificuldade para respirar", "chiado", "respirar")
_ARTICULATION_TERMS = ("fala arrastada", "fala enrolada", "disartria", "dificuldade para falar")


def analyze_audio(payload: AudioAnalysisRequest, settings: Settings) -> tuple[MultimodalEvent, dict]:
    azure_speech = transcribe_audio_with_azure(
        settings,
        audio_file_path=payload.audio_file_path,
        language=payload.language,
    )
    transcript_source = "request"
    raw_transcript = (payload.transcript or "").strip()
    if azure_speech.success and azure_speech.transcript:
        raw_transcript = azure_speech.transcript
        transcript_source = "azure_speech"

    transcript = _normalize_text(raw_transcript)
    evidence: list[str] = []
    score = 0.0
    tags: list[str] = []

    fatigue_hits = _match_terms(transcript, _FATIGUE_TERMS)
    breathing_hits = _match_terms(transcript, _BREATHING_TERMS)
    articulation_hits = _match_terms(transcript, _ARTICULATION_TERMS)

    if fatigue_hits:
        score += 0.28
        tags.append("fadiga_vocal")
        evidence.append(f"termos_fadiga={', '.join(fatigue_hits)}")
    if breathing_hits:
        score += 0.35
        tags.append("sintoma_respiratorio")
        evidence.append(f"termos_respiratorios={', '.join(breathing_hits)}")
    if articulation_hits:
        score += 0.32
        tags.append("alteracao_articulacao")
        evidence.append(f"termos_articulacao={', '.join(articulation_hits)}")

    metrics = payload.metrics
    if metrics.pause_ratio is not None and metrics.pause_ratio >= 0.30:
        score += 0.14
        evidence.append(f"pause_ratio={metrics.pause_ratio:.2f}")
    if metrics.speech_rate_wpm is not None and metrics.speech_rate_wpm < 95:
        score += 0.10
        evidence.append(f"speech_rate_wpm={metrics.speech_rate_wpm:.1f}")
    if metrics.vocal_energy is not None and metrics.vocal_energy < 0.35:
        score += 0.10
        evidence.append(f"vocal_energy={metrics.vocal_energy:.2f}")
    if metrics.articulation_clarity is not None and metrics.articulation_clarity < 0.45:
        score += 0.14
        evidence.append(f"articulation_clarity={metrics.articulation_clarity:.2f}")
    if metrics.breathing_irregularity is not None and metrics.breathing_irregularity >= 0.55:
        score += 0.16
        evidence.append(f"breathing_irregularity={metrics.breathing_irregularity:.2f}")

    azure_text = analyze_text_with_azure(settings, text=raw_transcript)
    if azure_text.success:
        if azure_text.sentiment == "negative":
            score += 0.08
            evidence.append("azure_sentiment=negative")
        if azure_text.key_phrases:
            evidence.append(f"azure_key_phrases={', '.join(azure_text.key_phrases[:4])}")
        symptom_entities = [
            entity["text"]
            for entity in azure_text.entities
            if str(entity.get("category") or "").lower() in {"symptomorsign", "healthcare", "medicalcondition"}
        ]
        if symptom_entities:
            score += 0.08
            evidence.append(f"azure_entities={', '.join(symptom_entities[:4])}")

    anomaly_score = min(round(score, 4), 1.0)
    signal = _select_signal(tags)
    severity = _severity_from_score(anomaly_score)

    azure = get_azure_availability(settings)
    details = {
        "signal_tags": tags,
        "azure_speech_configured": azure.speech_enabled,
        "azure_text_configured": azure.text_enabled,
        "azure_speech_used": azure_speech.provider_used,
        "azure_speech_success": azure_speech.success,
        "azure_speech_error": azure_speech.error,
        "azure_text_used": azure_text.provider_used,
        "azure_text_success": azure_text.success,
        "azure_text_error": azure_text.error,
        "azure_text_sentiment": azure_text.sentiment,
        "azure_text_key_phrases": azure_text.key_phrases[:8],
        "azure_text_entities": azure_text.entities[:8],
        "transcript_source": transcript_source,
        "transcript_length": len(raw_transcript),
    }

    event = MultimodalEvent(
        event_id=f"audio-{uuid4()}",
        patient_id=payload.patient_id,
        modality="audio",
        timestamp=payload.timestamp,
        signal=signal,
        severity=severity,
        anomaly_score=anomaly_score,
        evidence=evidence or ["sem_evidencias_relevantes"],
        transcript_excerpt=raw_transcript[:280] if raw_transcript else None,
        metadata={
            **payload.metadata,
            "pipeline": "audio",
            "audio_file_path": payload.audio_file_path,
            "language": payload.language,
            "transcript_source": transcript_source,
            "azure_speech_configured": azure.speech_enabled,
            "azure_text_configured": azure.text_enabled,
            "azure_speech_success": azure_speech.success,
            "azure_text_success": azure_text.success,
        },
    )
    return event, details


def _normalize_text(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text.strip().lower())
    return cleaned


def _match_terms(text: str, terms: tuple[str, ...]) -> list[str]:
    return [term for term in terms if term in text]


def _select_signal(tags: list[str]) -> str:
    if not tags:
        return "audio_sem_anomalia_relevante"
    return tags[0]


def _severity_from_score(score: float) -> str:
    if score >= 0.8:
        return "high"
    if score >= 0.55:
        return "medium"
    if score >= 0.25:
        return "low"
    return "info"
