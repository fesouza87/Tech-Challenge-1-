from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    app_env: str
    api_host: str
    api_port: int
    alert_high_threshold: float
    alert_medium_threshold: float
    alert_low_threshold: float
    audit_log_path: str
    azure_speech_endpoint: str | None
    azure_speech_key: str | None
    azure_speech_region: str | None
    azure_text_endpoint: str | None
    azure_text_key: str | None
    video_report_dir: str
    yolo_model_path: str | None
    openpose_dir: str | None
    video_pose_provider: str


def load_settings() -> Settings:
    base_dir = Path(__file__).resolve().parents[2]
    audit_log_raw = os.environ.get("TC4_AUDIT_LOG", "logs/audit.jsonl")
    audit_log_path = Path(audit_log_raw)
    if not audit_log_path.is_absolute():
        audit_log_path = base_dir / audit_log_path
    video_report_raw = os.environ.get("TC4_VIDEO_REPORT_DIR", "reports/video")
    video_report_dir = Path(video_report_raw)
    if not video_report_dir.is_absolute():
        video_report_dir = base_dir / video_report_dir

    return Settings(
        app_env=os.environ.get("TC4_ENV", "dev").strip().lower(),
        api_host=os.environ.get("TC4_API_HOST", "127.0.0.1").strip(),
        api_port=int(os.environ.get("TC4_API_PORT", "8010")),
        alert_high_threshold=float(os.environ.get("TC4_ALERT_HIGH_THRESHOLD", "0.8")),
        alert_medium_threshold=float(os.environ.get("TC4_ALERT_MEDIUM_THRESHOLD", "0.6")),
        alert_low_threshold=float(os.environ.get("TC4_ALERT_LOW_THRESHOLD", "0.4")),
        audit_log_path=str(audit_log_path),
        azure_speech_endpoint=_optional_env("AZURE_SPEECH_ENDPOINT"),
        azure_speech_key=_optional_env("AZURE_SPEECH_KEY"),
        azure_speech_region=_optional_env("AZURE_SPEECH_REGION"),
        azure_text_endpoint=_optional_env("AZURE_TEXT_ENDPOINT"),
        azure_text_key=_optional_env("AZURE_TEXT_KEY"),
        video_report_dir=str(video_report_dir),
        yolo_model_path=_optional_env("TC4_YOLO_MODEL_PATH"),
        openpose_dir=_optional_env("TC4_OPENPOSE_DIR"),
        video_pose_provider=os.environ.get("TC4_VIDEO_POSE_PROVIDER", "auto").strip().lower(),
    )


def _optional_env(name: str) -> str | None:
    value = os.environ.get(name, "").strip()
    return value or None
