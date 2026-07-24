from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

Modality = Literal["audio", "video", "text", "vitals"]
Severity = Literal["info", "low", "medium", "high"]
AlertSeverity = Literal["low", "medium", "high"]


class IngestEventRequest(BaseModel):
    event_id: str | None = None
    patient_id: str = Field(min_length=1)
    modality: str = Field(min_length=1)
    timestamp: datetime
    signal: str = Field(min_length=1)
    severity: str = Field(default="medium", min_length=1)
    anomaly_score: float = Field(ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list)
    transcript_excerpt: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class MultimodalEvent(BaseModel):
    event_id: str
    patient_id: str
    modality: Modality
    timestamp: datetime
    signal: str
    severity: Severity
    anomaly_score: float
    evidence: list[str] = Field(default_factory=list)
    transcript_excerpt: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AlertItem(BaseModel):
    alert_id: str
    patient_id: str
    event_id: str
    modality: Modality
    severity: AlertSeverity
    title: str
    message: str
    evidence: list[str] = Field(default_factory=list)
    anomaly_score: float
    recommended_action: str
    created_at: datetime


class PatientRiskSummary(BaseModel):
    patient_id: str
    event_count: int
    alert_count: int
    average_anomaly_score: float
    highest_severity: str
    active_modalities: list[str] = Field(default_factory=list)
    latest_signal: str | None = None


class IngestEventResponse(BaseModel):
    accepted: bool
    event: MultimodalEvent
    generated_alert: AlertItem | None = None
    patient_risk: PatientRiskSummary


class AlertResponse(BaseModel):
    patient_id: str
    alerts: list[AlertItem] = Field(default_factory=list)
    risk_summary: PatientRiskSummary


class AudioMetrics(BaseModel):
    speech_rate_wpm: float | None = Field(default=None, ge=0.0)
    pause_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    vocal_energy: float | None = Field(default=None, ge=0.0, le=1.0)
    articulation_clarity: float | None = Field(default=None, ge=0.0, le=1.0)
    breathing_irregularity: float | None = Field(default=None, ge=0.0, le=1.0)


class AudioAnalysisRequest(BaseModel):
    patient_id: str = Field(min_length=1)
    timestamp: datetime
    transcript: str | None = None
    audio_file_path: str | None = None
    language: str = Field(default="pt-BR", min_length=2)
    metrics: AudioMetrics = Field(default_factory=AudioMetrics)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_audio_input(self) -> "AudioAnalysisRequest":
        if not (self.transcript and self.transcript.strip()) and not (self.audio_file_path and self.audio_file_path.strip()):
            raise ValueError("Informe `transcript` ou `audio_file_path` para o pipeline de audio.")
        return self


class TextAnalysisRequest(BaseModel):
    patient_id: str = Field(min_length=1)
    timestamp: datetime
    clinical_note: str = Field(min_length=1)
    prescription_text: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class VitalsSample(BaseModel):
    timestamp: datetime
    heart_rate: float | None = Field(default=None, ge=0.0)
    spo2: float | None = Field(default=None, ge=0.0, le=100.0)
    systolic_bp: float | None = Field(default=None, ge=0.0)
    diastolic_bp: float | None = Field(default=None, ge=0.0)
    respiratory_rate: float | None = Field(default=None, ge=0.0)
    temperature_c: float | None = Field(default=None, ge=0.0)


class VitalsAnalysisRequest(BaseModel):
    patient_id: str = Field(min_length=1)
    samples: list[VitalsSample] = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class VitalsVitalDbImportRequest(BaseModel):
    patient_id: str = Field(min_length=1)
    vital_file_path: str | None = None
    interval_seconds: int = Field(default=60, ge=1)
    max_samples: int = Field(default=24, ge=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class VideoAnalysisRequest(BaseModel):
    patient_id: str = Field(min_length=1)
    timestamp: datetime
    procedure_type: str = Field(min_length=1)
    video_file_path: str | None = None
    expected_objects: list[str] = Field(default_factory=list)
    expected_people: int = Field(default=1, ge=1)
    frame_stride: int = Field(default=10, ge=1)
    max_frames: int = Field(default=24, ge=1)
    pose_deviation_score: float = Field(default=0.0, ge=0.0, le=1.0)
    motion_anomaly_score: float = Field(default=0.0, ge=0.0, le=1.0)
    critical_area_intrusions: int = Field(default=0, ge=0)
    unexpected_objects: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class PipelineExecutionResult(BaseModel):
    pipeline: Modality
    event: MultimodalEvent
    details: dict[str, Any] = Field(default_factory=dict)


class PipelineRunResponse(BaseModel):
    accepted: bool
    pipeline: Modality
    event: MultimodalEvent
    generated_alert: AlertItem | None = None
    patient_risk: PatientRiskSummary
    details: dict[str, Any] = Field(default_factory=dict)
