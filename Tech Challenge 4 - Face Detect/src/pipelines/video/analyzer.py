from __future__ import annotations

from uuid import uuid4

from pipelines.video.inference import analyze_video_file
from shared.config import Settings
from shared.models import MultimodalEvent, VideoAnalysisRequest


def analyze_video(payload: VideoAnalysisRequest, settings: Settings) -> tuple[MultimodalEvent, dict]:
    runtime = analyze_video_file(payload, settings)
    evidence: list[str] = []
    score = 0.0

    pose_score = max(payload.pose_deviation_score, runtime.pose_deviation_score)
    motion_score = max(payload.motion_anomaly_score, runtime.motion_anomaly_score)
    critical_intrusions = max(payload.critical_area_intrusions, runtime.critical_area_intrusions)
    unexpected_objects = sorted(set(payload.unexpected_objects) | set(runtime.unexpected_objects))

    if pose_score >= 0.55:
        score += min(pose_score * 0.35, 0.35)
        evidence.append(f"pose_deviation_score={pose_score:.2f}")
    if motion_score >= 0.50:
        score += min(motion_score * 0.30, 0.30)
        evidence.append(f"motion_anomaly_score={motion_score:.2f}")
    if critical_intrusions > 0:
        score += min(critical_intrusions * 0.15, 0.30)
        evidence.append(f"critical_area_intrusions={critical_intrusions}")
    if unexpected_objects:
        score += min(len(unexpected_objects) * 0.08, 0.24)
        evidence.append(f"unexpected_objects={', '.join(unexpected_objects)}")

    anomaly_score = min(round(score, 4), 1.0)
    signal = "video_sem_anomalia_relevante"
    if critical_intrusions > 0:
        signal = "intrusao_area_critica"
    elif pose_score >= 0.55:
        signal = "desvio_postural"
    elif motion_score >= 0.50:
        signal = "movimento_fora_do_padrao"

    event = MultimodalEvent(
        event_id=f"video-{uuid4()}",
        patient_id=payload.patient_id,
        modality="video",
        timestamp=payload.timestamp,
        signal=signal,
        severity=_severity_from_score(anomaly_score),
        anomaly_score=anomaly_score,
        evidence=evidence or ["sem_evidencias_relevantes"],
        metadata={
            **payload.metadata,
            "pipeline": "video",
            "procedure_type": payload.procedure_type,
            "video_file_path": payload.video_file_path,
            "yolo_enabled": runtime.yolo_used,
            "pose_enabled": bool(runtime.pose_provider and not runtime.pose_error),
            "pose_provider": runtime.pose_provider,
        },
    )
    details = {
        "procedure_type": payload.procedure_type,
        "frames_processed": runtime.frames_processed,
        "yolo_used": runtime.yolo_used,
        "yolo_error": runtime.yolo_error,
        "pose_provider": runtime.pose_provider,
        "pose_error": runtime.pose_error,
        "object_counts": runtime.object_counts,
        "unexpected_objects": unexpected_objects,
        "unexpected_objects_count": len(unexpected_objects),
        "critical_area_intrusions": critical_intrusions,
        "runtime_pose_deviation_score": runtime.pose_deviation_score,
        "runtime_motion_anomaly_score": runtime.motion_anomaly_score,
        "report_notes": runtime.report_notes,
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
