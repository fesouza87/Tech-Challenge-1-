from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from shared.config import Settings


@dataclass(frozen=True)
class AzureAvailability:
    speech_enabled: bool
    text_enabled: bool


@dataclass(frozen=True)
class AzureSpeechResult:
    transcript: str | None
    provider_used: bool
    success: bool
    error: str | None


@dataclass(frozen=True)
class AzureTextResult:
    provider_used: bool
    success: bool
    sentiment: str | None
    confidence_scores: dict[str, float]
    key_phrases: list[str]
    entities: list[dict[str, Any]]
    error: str | None


def get_azure_availability(settings: Settings) -> AzureAvailability:
    return AzureAvailability(
        speech_enabled=bool(settings.azure_speech_endpoint and settings.azure_speech_key),
        text_enabled=bool(settings.azure_text_endpoint and settings.azure_text_key),
    )


def transcribe_audio_with_azure(
    settings: Settings,
    *,
    audio_file_path: str | None,
    language: str,
) -> AzureSpeechResult:
    availability = get_azure_availability(settings)
    if not availability.speech_enabled:
        return AzureSpeechResult(transcript=None, provider_used=False, success=False, error="azure_speech_not_configured")
    if not audio_file_path:
        return AzureSpeechResult(transcript=None, provider_used=False, success=False, error="audio_file_path_not_provided")

    audio_path = Path(audio_file_path)
    if not audio_path.is_file():
        return AzureSpeechResult(transcript=None, provider_used=False, success=False, error="audio_file_not_found")

    try:
        import azure.cognitiveservices.speech as speechsdk
    except Exception:
        return AzureSpeechResult(transcript=None, provider_used=False, success=False, error="azure_speech_sdk_not_installed")

    try:
        region = _resolve_speech_region(settings)
        if region:
            speech_config = speechsdk.SpeechConfig(
                subscription=settings.azure_speech_key,
                region=region,
            )
        else:
            speech_config = speechsdk.SpeechConfig(
                subscription=settings.azure_speech_key,
                endpoint=settings.azure_speech_endpoint,
            )
        speech_config.speech_recognition_language = language
        audio_config = speechsdk.audio.AudioConfig(filename=str(audio_path))
        recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
        result = recognizer.recognize_once()

        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            transcript = (result.text or "").strip()
            if transcript:
                return AzureSpeechResult(transcript=transcript, provider_used=True, success=True, error=None)
            return AzureSpeechResult(transcript=None, provider_used=True, success=False, error="azure_speech_empty_transcript")

        if result.reason == speechsdk.ResultReason.NoMatch:
            return AzureSpeechResult(transcript=None, provider_used=True, success=False, error="azure_speech_no_match")

        cancellation = result.cancellation_details
        message = getattr(cancellation, "error_details", None) or str(getattr(cancellation, "reason", "azure_speech_canceled"))
        return AzureSpeechResult(transcript=None, provider_used=True, success=False, error=message)
    except Exception as exc:
        return AzureSpeechResult(transcript=None, provider_used=True, success=False, error=str(exc))


def _resolve_speech_region(settings: Settings) -> str | None:
    if settings.azure_speech_region:
        return settings.azure_speech_region
    if not settings.azure_speech_endpoint:
        return None

    try:
        host = (urlparse(settings.azure_speech_endpoint).hostname or "").lower()
    except Exception:
        return None

    suffix = ".api.cognitive.microsoft.com"
    if host.endswith(suffix):
        region = host[: -len(suffix)].strip(".")
        return region or None
    return None


def analyze_text_with_azure(settings: Settings, *, text: str) -> AzureTextResult:
    availability = get_azure_availability(settings)
    if not availability.text_enabled:
        return AzureTextResult(
            provider_used=False,
            success=False,
            sentiment=None,
            confidence_scores={},
            key_phrases=[],
            entities=[],
            error="azure_text_not_configured",
        )
    if not text.strip():
        return AzureTextResult(
            provider_used=False,
            success=False,
            sentiment=None,
            confidence_scores={},
            key_phrases=[],
            entities=[],
            error="empty_text",
        )

    try:
        from azure.ai.textanalytics import TextAnalyticsClient
        from azure.core.credentials import AzureKeyCredential
    except Exception:
        return AzureTextResult(
            provider_used=False,
            success=False,
            sentiment=None,
            confidence_scores={},
            key_phrases=[],
            entities=[],
            error="azure_text_sdk_not_installed",
        )

    try:
        client = TextAnalyticsClient(
            endpoint=settings.azure_text_endpoint,
            credential=AzureKeyCredential(settings.azure_text_key),
        )

        sentiment_doc = client.analyze_sentiment([text])[0]
        key_phrase_doc = client.extract_key_phrases([text])[0]
        entities_doc = client.recognize_entities([text])[0]

        if getattr(sentiment_doc, "is_error", False):
            raise RuntimeError(f"sentiment_error:{sentiment_doc.error.code}")
        if getattr(key_phrase_doc, "is_error", False):
            raise RuntimeError(f"key_phrase_error:{key_phrase_doc.error.code}")
        if getattr(entities_doc, "is_error", False):
            raise RuntimeError(f"entities_error:{entities_doc.error.code}")

        confidence_scores = {
            "positive": float(sentiment_doc.confidence_scores.positive),
            "neutral": float(sentiment_doc.confidence_scores.neutral),
            "negative": float(sentiment_doc.confidence_scores.negative),
        }
        entities = [
            {
                "text": entity.text,
                "category": entity.category,
                "subcategory": entity.subcategory,
                "confidence_score": float(entity.confidence_score),
            }
            for entity in entities_doc.entities
        ]

        return AzureTextResult(
            provider_used=True,
            success=True,
            sentiment=str(sentiment_doc.sentiment),
            confidence_scores=confidence_scores,
            key_phrases=list(key_phrase_doc.key_phrases),
            entities=entities,
            error=None,
        )
    except Exception as exc:
        return AzureTextResult(
            provider_used=True,
            success=False,
            sentiment=None,
            confidence_scores={},
            key_phrases=[],
            entities=[],
            error=str(exc),
        )
