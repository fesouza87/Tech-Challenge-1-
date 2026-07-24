from __future__ import annotations

import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from pydantic import ValidationError

from ingestion.service import record_event
from pipelines.audio.analyzer import analyze_audio
from pipelines.text.analyzer import analyze_text
from pipelines.vitals.analyzer import analyze_vitals
from pipelines.vitals.vitaldb_import import import_vitaldb_as_vitals
from pipelines.video.analyzer import analyze_video
from pipelines.video.reporting import write_video_report
from shared.models import (
    AudioMetrics,
    AudioAnalysisRequest,
    PipelineRunResponse,
    TextAnalysisRequest,
    VideoAnalysisRequest,
    VitalsAnalysisRequest,
    VitalsVitalDbImportRequest,
)

router = APIRouter(prefix="/api/pipelines", tags=["pipelines"])


@router.post("/audio", response_model=PipelineRunResponse)
def run_audio_pipeline(payload: AudioAnalysisRequest, request: Request) -> PipelineRunResponse:
    container = request.app.state.container
    event, details = analyze_audio(payload, container.settings)
    return record_event(container, event, source="pipeline_audio", details=details)


@router.post("/audio/upload", response_model=PipelineRunResponse)
async def run_audio_upload_pipeline(
    request: Request,
    patient_id: str = Form(...),
    language: str = Form("pt-BR"),
    timestamp: str | None = Form(default=None),
    transcript: str | None = Form(default=None),
    pause_ratio: float | None = Form(default=None),
    speech_rate_wpm: float | None = Form(default=None),
    vocal_energy: float | None = Form(default=None),
    articulation_clarity: float | None = Form(default=None),
    breathing_irregularity: float | None = Form(default=None),
    audio_file: UploadFile | None = File(default=None),
) -> PipelineRunResponse:
    container = request.app.state.container
    temp_file_path: str | None = None

    if audio_file is not None and audio_file.filename:
        suffix = Path(audio_file.filename).suffix or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file_path = temp_file.name
            while chunk := await audio_file.read(1024 * 1024):
                temp_file.write(chunk)

    try:
        payload = AudioAnalysisRequest(
            patient_id=patient_id,
            timestamp=timestamp or datetime.now(timezone.utc).isoformat(),
            transcript=transcript,
            audio_file_path=temp_file_path,
            language=language,
            metrics=AudioMetrics(
                pause_ratio=pause_ratio,
                speech_rate_wpm=speech_rate_wpm,
                vocal_energy=vocal_energy,
                articulation_clarity=articulation_clarity,
                breathing_irregularity=breathing_irregularity,
            ),
            metadata={"ingest_mode": "multipart_upload"},
        )
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc

    try:
        event, details = analyze_audio(payload, container.settings)
        details["uploaded_filename"] = audio_file.filename if audio_file else None
        return record_event(container, event, source="pipeline_audio_upload", details=details)
    finally:
        if audio_file is not None:
            await audio_file.close()
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)


@router.post("/text", response_model=PipelineRunResponse)
def run_text_pipeline(payload: TextAnalysisRequest, request: Request) -> PipelineRunResponse:
    container = request.app.state.container
    event, details = analyze_text(payload)
    return record_event(container, event, source="pipeline_text", details=details)


@router.post("/vitals", response_model=PipelineRunResponse)
def run_vitals_pipeline(payload: VitalsAnalysisRequest, request: Request) -> PipelineRunResponse:
    container = request.app.state.container
    event, details = analyze_vitals(payload)
    return record_event(container, event, source="pipeline_vitals", details=details)


@router.post("/vitals/vitaldb", response_model=PipelineRunResponse)
def run_vitals_vitaldb_pipeline(payload: VitalsVitalDbImportRequest, request: Request) -> PipelineRunResponse:
    container = request.app.state.container
    try:
        imported = import_vitaldb_as_vitals(payload)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    event, details = analyze_vitals(imported.payload)
    details.update(imported.details)
    return record_event(container, event, source="pipeline_vitals_vitaldb", details=details)


@router.post("/video", response_model=PipelineRunResponse)
def run_video_pipeline(payload: VideoAnalysisRequest, request: Request) -> PipelineRunResponse:
    container = request.app.state.container
    event, details = analyze_video(payload, container.settings)
    details.update(write_video_report(container.settings, event, details))
    return record_event(container, event, source="pipeline_video", details=details)
