from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import fabs
from pathlib import Path

from shared.config import Settings
from shared.models import VideoAnalysisRequest


@dataclass(frozen=True)
class VideoInferenceResult:
    frames_processed: int
    yolo_used: bool
    yolo_error: str | None
    pose_provider: str | None
    pose_error: str | None
    object_counts: dict[str, int]
    unexpected_objects: list[str]
    pose_deviation_score: float
    motion_anomaly_score: float
    critical_area_intrusions: int
    report_notes: list[str]


def analyze_video_file(payload: VideoAnalysisRequest, settings: Settings) -> VideoInferenceResult:
    if not payload.video_file_path:
        return VideoInferenceResult(
            frames_processed=0,
            yolo_used=False,
            yolo_error="video_file_path_not_provided",
            pose_provider=None,
            pose_error="video_file_path_not_provided",
            object_counts={},
            unexpected_objects=[],
            pose_deviation_score=0.0,
            motion_anomaly_score=0.0,
            critical_area_intrusions=0,
            report_notes=["Analise visual real nao executada: caminho do video nao informado."],
        )

    video_path = Path(payload.video_file_path)
    if not video_path.is_file():
        return VideoInferenceResult(
            frames_processed=0,
            yolo_used=False,
            yolo_error="video_file_not_found",
            pose_provider=None,
            pose_error="video_file_not_found",
            object_counts={},
            unexpected_objects=[],
            pose_deviation_score=0.0,
            motion_anomaly_score=0.0,
            critical_area_intrusions=0,
            report_notes=[f"Arquivo de video nao encontrado: {video_path}"],
        )

    try:
        import cv2
        import numpy as np
    except Exception:
        return VideoInferenceResult(
            frames_processed=0,
            yolo_used=False,
            yolo_error="opencv_not_installed",
            pose_provider=None,
            pose_error="opencv_not_installed",
            object_counts={},
            unexpected_objects=[],
            pose_deviation_score=0.0,
            motion_anomaly_score=0.0,
            critical_area_intrusions=0,
            report_notes=["OpenCV nao instalado; analise visual real indisponivel."],
        )

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return VideoInferenceResult(
            frames_processed=0,
            yolo_used=False,
            yolo_error="video_open_failed",
            pose_provider=None,
            pose_error="video_open_failed",
            object_counts={},
            unexpected_objects=[],
            pose_deviation_score=0.0,
            motion_anomaly_score=0.0,
            critical_area_intrusions=0,
            report_notes=["Falha ao abrir o arquivo de video."],
        )

    yolo_model, yolo_error = _build_yolo_model(settings)
    pose_backend, pose_provider, pose_error = _build_pose_backend(settings)

    object_counter: Counter[str] = Counter()
    unexpected: set[str] = set()
    people_counts: list[int] = []
    pose_scores: list[float] = []
    motion_scores: list[float] = []
    previous_gray = None
    frames_processed = 0
    frame_index = 0

    while frames_processed < payload.max_frames:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index % payload.frame_stride != 0:
            frame_index += 1
            continue

        frames_processed += 1
        frame_index += 1

        yolo_labels: list[str] = []
        if yolo_model is not None:
            try:
                results = yolo_model.predict(frame, verbose=False)
                if results:
                    names = getattr(results[0], "names", {})
                    boxes = getattr(results[0], "boxes", None)
                    if boxes is not None and getattr(boxes, "cls", None) is not None:
                        for cls_idx in boxes.cls.tolist():
                            label = names.get(int(cls_idx), str(int(cls_idx)))
                            yolo_labels.append(str(label))
            except Exception as exc:
                yolo_error = str(exc)
                yolo_model = None

        for label in yolo_labels:
            object_counter[label] += 1
        if yolo_labels:
            people_counts.append(sum(1 for label in yolo_labels if label == "person"))
            if payload.expected_objects:
                for label in yolo_labels:
                    if label not in payload.expected_objects:
                        unexpected.add(label)

        pose_score = 0.0
        if pose_backend is not None:
            try:
                pose_score = _estimate_pose_score(frame, pose_backend, pose_provider)
            except Exception as exc:
                pose_error = str(exc)
                pose_backend = None
        if pose_score > 0:
            pose_scores.append(pose_score)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if previous_gray is not None:
            diff = cv2.absdiff(previous_gray, gray)
            motion_scores.append(float(np.mean(diff) / 255.0))
        previous_gray = gray

    capture.release()

    pose_deviation = min(_avg(pose_scores), 1.0)
    motion_anomaly = min(_avg(motion_scores) * 3.0, 1.0)
    critical_intrusions = max(0, (max(people_counts) if people_counts else 0) - payload.expected_people)

    notes = [
        f"Frames processados: {frames_processed}",
        f"YOLO utilizado: {bool(yolo_model is not None or not yolo_error)}",
        f"Provedor de pose: {pose_provider or 'nenhum'}",
    ]
    if yolo_error:
        notes.append(f"YOLO erro: {yolo_error}")
    if pose_error:
        notes.append(f"Pose erro: {pose_error}")

    return VideoInferenceResult(
        frames_processed=frames_processed,
        yolo_used=bool(yolo_model is not None or (yolo_error is None and frames_processed > 0)),
        yolo_error=yolo_error,
        pose_provider=pose_provider,
        pose_error=pose_error,
        object_counts=dict(object_counter),
        unexpected_objects=sorted(unexpected),
        pose_deviation_score=round(pose_deviation, 4),
        motion_anomaly_score=round(motion_anomaly, 4),
        critical_area_intrusions=critical_intrusions,
        report_notes=notes,
    )


def _build_yolo_model(settings: Settings):
    try:
        from ultralytics import YOLO
    except Exception:
        return None, "ultralytics_not_installed"

    model_ref = settings.yolo_model_path or "yolov8n.pt"
    try:
        return YOLO(model_ref), None
    except Exception as exc:
        return None, str(exc)


def _build_pose_backend(settings: Settings):
    provider = settings.video_pose_provider
    if provider in {"auto", "openpose"}:
        backend, error = _try_openpose(settings)
        if backend is not None:
            return backend, "openpose", None
        if provider == "openpose":
            return None, "openpose", error

    if provider in {"auto", "mediapipe"}:
        backend, error = _try_mediapipe()
        if backend is not None:
            return backend, "mediapipe", None
        return None, "mediapipe", error

    return None, provider, "unsupported_pose_provider"


def _try_openpose(settings: Settings):
    if not settings.openpose_dir:
        return None, "openpose_dir_not_configured"
    try:
        import sys

        openpose_python = Path(settings.openpose_dir) / "build" / "python"
        if str(openpose_python) not in sys.path:
            sys.path.append(str(openpose_python))
        import pyopenpose as op

        params = {"model_folder": str(Path(settings.openpose_dir) / "models")}
        wrapper = op.WrapperPython()
        wrapper.configure(params)
        wrapper.start()
        return {"op": op, "wrapper": wrapper}, None
    except Exception as exc:
        return None, str(exc)


def _try_mediapipe():
    try:
        import mediapipe as mp

        pose = mp.solutions.pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        return {"mp": mp, "pose": pose}, None
    except Exception as exc:
        return None, str(exc)


def _estimate_pose_score(frame, backend, provider: str | None) -> float:
    if provider == "openpose":
        op = backend["op"]
        wrapper = backend["wrapper"]
        datum = op.Datum()
        datum.cvInputData = frame
        wrapper.emplaceAndPop(op.VectorDatum([datum]))
        keypoints = datum.poseKeypoints
        if keypoints is None or len(keypoints) == 0:
            return 0.0
        shoulder_diff = fabs(float(keypoints[0][2][1]) - float(keypoints[0][5][1]))
        hip_diff = fabs(float(keypoints[0][9][1]) - float(keypoints[0][12][1]))
        frame_h = max(frame.shape[0], 1)
        return min((shoulder_diff + hip_diff) / (2.0 * frame_h), 1.0)

    if provider == "mediapipe":
        cv2 = __import__("cv2")
        pose = backend["pose"]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)
        if not result.pose_landmarks:
            return 0.0
        landmarks = result.pose_landmarks.landmark
        shoulder_diff = fabs(landmarks[11].y - landmarks[12].y)
        hip_diff = fabs(landmarks[23].y - landmarks[24].y)
        return min((shoulder_diff + hip_diff) * 2.5, 1.0)

    return 0.0


def _avg(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)
