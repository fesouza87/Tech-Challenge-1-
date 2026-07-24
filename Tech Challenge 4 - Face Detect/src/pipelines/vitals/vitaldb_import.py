from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from shared.models import VitalsAnalysisRequest, VitalsSample, VitalsVitalDbImportRequest


@dataclass(frozen=True)
class VitalDbImportResult:
    payload: VitalsAnalysisRequest
    details: dict[str, Any]


DEFAULT_VITAL_DEMO_PATH = Path(__file__).resolve().parents[3] / "vital" / "0001.vital"

TRACK_CANDIDATES: dict[str, tuple[str, ...]] = {
    "heart_rate": ("Solar8000/HR", "Solar8000/PLETH_HR"),
    "spo2": ("Solar8000/PLETH_SPO2",),
    "systolic_bp": ("Solar8000/NIBP_SBP", "Solar8000/ART_SBP"),
    "diastolic_bp": ("Solar8000/NIBP_DBP", "Solar8000/ART_DBP"),
    "respiratory_rate": ("Solar8000/VENT_RR", "Solar8000/RR_CO2", "Primus/RR_CO2"),
    "temperature_c": ("Solar8000/BT",),
}


def import_vitaldb_as_vitals(payload: VitalsVitalDbImportRequest) -> VitalDbImportResult:
    try:
        import vitaldb
    except Exception as exc:
        raise RuntimeError("vitaldb_not_installed") from exc

    vital_path = Path(payload.vital_file_path) if payload.vital_file_path else DEFAULT_VITAL_DEMO_PATH
    if not vital_path.is_file():
        raise FileNotFoundError(f"Arquivo .vital nao encontrado: {vital_path}")

    track_names = [track for candidates in TRACK_CANDIDATES.values() for track in candidates]
    vital_file = vitaldb.VitalFile(str(vital_path))
    frame = vital_file.to_pandas(track_names, interval=payload.interval_seconds, return_datetime=True)

    samples: list[VitalsSample] = []
    for row in frame.to_dict(orient="records"):
        timestamp = _normalize_timestamp(row.get("Time"))
        if timestamp is None:
            continue

        sample = VitalsSample(
            timestamp=timestamp,
            heart_rate=_pick_value(row, TRACK_CANDIDATES["heart_rate"], minimum=20.0, maximum=250.0),
            spo2=_pick_value(row, TRACK_CANDIDATES["spo2"], minimum=50.0, maximum=100.0),
            systolic_bp=_pick_value(row, TRACK_CANDIDATES["systolic_bp"], minimum=40.0, maximum=300.0),
            diastolic_bp=_pick_value(row, TRACK_CANDIDATES["diastolic_bp"], minimum=20.0, maximum=200.0),
            respiratory_rate=_pick_value(row, TRACK_CANDIDATES["respiratory_rate"], minimum=1.0, maximum=80.0),
            temperature_c=_pick_value(row, TRACK_CANDIDATES["temperature_c"], minimum=30.0, maximum=45.0),
        )
        if _sample_has_data(sample):
            samples.append(sample)

    if not samples:
        raise ValueError("Nenhuma amostra valida foi extraida do arquivo .vital.")

    selected_samples = samples[-payload.max_samples :]
    track_map = {
        field_name: _first_existing_track(frame.columns, candidates)
        for field_name, candidates in TRACK_CANDIDATES.items()
    }
    request_payload = VitalsAnalysisRequest(
        patient_id=payload.patient_id,
        samples=selected_samples,
        metadata={
            **payload.metadata,
            "source": "vitaldb_import",
            "vital_file_path": str(vital_path),
            "interval_seconds": payload.interval_seconds,
            "track_map": track_map,
            "available_track_count": len(vital_file.get_track_names()),
            "raw_sample_count": len(samples),
        },
    )
    details = {
        "vital_file_path": str(vital_path),
        "interval_seconds": payload.interval_seconds,
        "raw_sample_count": len(samples),
        "selected_sample_count": len(selected_samples),
        "track_map": track_map,
        "latest_sample": selected_samples[-1].model_dump(),
        "sample_series": [sample.model_dump(mode="json") for sample in selected_samples],
        "signal_ranges": _build_signal_ranges(selected_samples),
    }
    return VitalDbImportResult(payload=request_payload, details=details)


def _pick_value(row: dict[str, Any], candidates: tuple[str, ...], *, minimum: float, maximum: float) -> float | None:
    for track_name in candidates:
        value = _coerce_float(row.get(track_name))
        if value is None:
            continue
        if minimum <= value <= maximum:
            return value
    return None


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:
        return None
    return number


def _normalize_timestamp(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if hasattr(value, "to_pydatetime"):
        try:
            return value.to_pydatetime()
        except Exception:
            return None
    return None


def _sample_has_data(sample: VitalsSample) -> bool:
    return any(
        value is not None
        for value in (
            sample.heart_rate,
            sample.spo2,
            sample.systolic_bp,
            sample.diastolic_bp,
            sample.respiratory_rate,
            sample.temperature_c,
        )
    )


def _first_existing_track(columns: Any, candidates: tuple[str, ...]) -> str | None:
    for track_name in candidates:
        if track_name in columns:
            return track_name
    return None


def _build_signal_ranges(samples: list[VitalsSample]) -> dict[str, dict[str, float]]:
    fields = (
        "heart_rate",
        "spo2",
        "systolic_bp",
        "diastolic_bp",
        "respiratory_rate",
        "temperature_c",
    )
    ranges: dict[str, dict[str, float]] = {}
    for field_name in fields:
        values = [float(value) for sample in samples if (value := getattr(sample, field_name)) is not None]
        if not values:
            continue
        ranges[field_name] = {
            "min": min(values),
            "max": max(values),
        }
    return ranges
