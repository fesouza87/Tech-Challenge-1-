from __future__ import annotations

import json
import subprocess
import tempfile
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
    precomputed_pose_scores: list[float] | None = None
    if _pose_backend_mode(pose_backend) == "cli":
        precomputed_pose_scores, pose_error = _run_openpose_cli(video_path, payload, pose_backend)
        if precomputed_pose_scores is None:
            pose_backend = None
            if settings.video_pose_provider == "auto":
                fallback_backend, fallback_error = _try_mediapipe()
                if fallback_backend is not None:
                    pose_backend = fallback_backend
                    pose_provider = "mediapipe"
                    pose_error = None
                else:
                    pose_provider = "mediapipe"
                    pose_error = fallback_error

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
        if precomputed_pose_scores is not None:
            pose_score_index = frames_processed - 1
            if pose_score_index < len(precomputed_pose_scores):
                pose_score = precomputed_pose_scores[pose_score_index]
        elif pose_backend is not None:
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
    if pose_provider == "openpose" and _pose_backend_mode(pose_backend):
        notes.append(f"Modo OpenPose: {_pose_backend_mode(pose_backend)}")
        if pose_backend is not None and pose_backend.get("json_frames") is not None:
            notes.append(f"Frames JSON OpenPose: {pose_backend['json_frames']}")
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
    openpose_root = Path(settings.openpose_dir)
    errors: list[str] = []

    try:
        import sys

        openpose_python = openpose_root / "build" / "python"
        if str(openpose_python) not in sys.path:
            sys.path.append(str(openpose_python))
        import pyopenpose as op

        params = {"model_folder": str(openpose_root / "models")}
        wrapper = op.WrapperPython()
        wrapper.configure(params)
        wrapper.start()
        return {"mode": "python", "op": op, "wrapper": wrapper}, None
    except Exception as exc:
        errors.append(f"python={exc}")

    openpose_demo, model_folder = _find_openpose_cli_paths(openpose_root)
    if openpose_demo is not None and model_folder is not None:
        return {
            "mode": "cli",
            "demo_exe": openpose_demo,
            "model_folder": model_folder,
        }, None

    if openpose_demo is None:
        errors.append("cli=openpose_demo_not_found")
    elif model_folder is None:
        errors.append("cli=openpose_models_not_found")
    return None, "; ".join(errors)


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


def _pose_backend_mode(backend) -> str | None:
    if backend is None:
        return None
    return backend.get("mode")


def _find_openpose_cli_paths(openpose_root: Path) -> tuple[Path | None, Path | None]:
    demo_candidates = [
        openpose_root / "bin" / "OpenPoseDemo.exe",
        openpose_root / "build" / "x64" / "Release" / "OpenPoseDemo.exe",
    ]
    model_candidates = [
        openpose_root / "models",
        openpose_root / "build" / "models",
    ]

    demo_exe = next((path for path in demo_candidates if path.is_file()), None)
    model_folder = next((path for path in model_candidates if path.is_dir()), None)
    return demo_exe, model_folder


def _run_openpose_cli(video_path: Path, payload: VideoAnalysisRequest, backend) -> tuple[list[float] | None, str | None]:
    demo_exe = backend["demo_exe"]
    model_folder = backend["model_folder"]
    last_frame = max((payload.max_frames - 1) * payload.frame_stride, 0)
    command = [
        str(demo_exe),
        "--video",
        str(video_path),
        "--frame_first",
        "0",
        "--frame_last",
        str(last_frame),
        "--frame_step",
        str(payload.frame_stride),
        "--model_pose",
        "BODY_25",
        "--model_folder",
        str(model_folder),
        "--display",
        "0",
        "--render_pose",
        "0",
    ]

    try:
        with tempfile.TemporaryDirectory(prefix="tc4_openpose_") as temp_dir:
            output_dir = Path(temp_dir)
            command.extend(["--write_json", str(output_dir)])
            completed = subprocess.run(command, capture_output=True, text=True, timeout=600, check=False)
            if completed.returncode != 0:
                stderr = (completed.stderr or completed.stdout or "").strip()
                return None, stderr or f"openpose_cli_failed_with_exit_code_{completed.returncode}"

            json_files = sorted(output_dir.glob("*_keypoints.json"))
            if not json_files:
                return None, "openpose_cli_no_json_output"

            sampled_files = json_files[: payload.max_frames]
            backend["json_frames"] = len(json_files)
            backend["sampled_json_frames"] = len(sampled_files)
            return [_pose_score_from_json(path) for path in sampled_files], None
    except subprocess.TimeoutExpired:
        return None, "openpose_cli_timeout"
    except Exception as exc:
        return None, str(exc)


def _pose_score_from_json(json_path: Path) -> float:
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return 0.0

    best_score = 0.0
    for person in payload.get("people", []):
        keypoints = person.get("pose_keypoints_2d") or []
        score = _body25_alignment_score(keypoints)
        if score > best_score:
            best_score = score
    return min(best_score, 1.0)


def _body25_alignment_score(keypoints: list[float]) -> float:
    left_shoulder = _body25_point(keypoints, 5)
    right_shoulder = _body25_point(keypoints, 2)
    left_hip = _body25_point(keypoints, 12)
    right_hip = _body25_point(keypoints, 9)
    if not all([left_shoulder, right_shoulder, left_hip, right_hip]):
        return 0.0

    valid_y = [point[1] for point in _iter_body25_points(keypoints)]
    body_height = max(valid_y, default=0.0) - min(valid_y, default=0.0)
    if body_height <= 0:
        return 0.0

    shoulder_diff = fabs(right_shoulder[1] - left_shoulder[1])
    hip_diff = fabs(right_hip[1] - left_hip[1])
    return min((shoulder_diff + hip_diff) / (2.0 * body_height), 1.0)


def _body25_point(keypoints: list[float], index: int) -> tuple[float, float, float] | None:
    offset = index * 3
    if len(keypoints) <= offset + 2:
        return None
    x_coord = float(keypoints[offset])
    y_coord = float(keypoints[offset + 1])
    confidence = float(keypoints[offset + 2])
    if confidence <= 0 or (x_coord == 0 and y_coord == 0):
        return None
    return x_coord, y_coord, confidence


def _iter_body25_points(keypoints: list[float]):
    for index in range(len(keypoints) // 3):
        point = _body25_point(keypoints, index)
        if point is not None:
            yield point
