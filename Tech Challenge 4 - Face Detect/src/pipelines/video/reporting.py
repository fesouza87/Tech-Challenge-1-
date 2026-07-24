from __future__ import annotations

import json
from pathlib import Path

from shared.config import Settings
from shared.models import MultimodalEvent


def write_video_report(settings: Settings, event: MultimodalEvent, details: dict) -> dict[str, str]:
    report_dir = Path(settings.video_report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    base_name = f"{event.patient_id}_{event.event_id}"
    json_path = report_dir / f"{base_name}.json"
    txt_path = report_dir / f"{base_name}.txt"

    payload = {
        "event": event.model_dump(mode="json"),
        "details": details,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "Relatorio de Analise de Video",
        f"Paciente: {event.patient_id}",
        f"Evento: {event.event_id}",
        f"Procedimento: {event.metadata.get('procedure_type', 'nao informado')}",
        f"Sinal detectado: {event.signal}",
        f"Severidade: {event.severity}",
        f"Score de anomalia: {event.anomaly_score:.4f}",
        "",
        "Evidencias:",
    ]
    for item in event.evidence:
        lines.append(f"- {item}")

    lines.extend(
        [
            "",
            "Resumo tecnico:",
            f"- Frames processados: {details.get('frames_processed', 0)}",
            f"- YOLO utilizado: {details.get('yolo_used')}",
            f"- Provedor de pose: {details.get('pose_provider')}",
            f"- Objetos detectados: {details.get('object_counts')}",
            f"- Objetos inesperados: {details.get('unexpected_objects')}",
            f"- Intrusoes em area critica: {details.get('critical_area_intrusions')}",
        ]
    )
    txt_path.write_text("\n".join(lines), encoding="utf-8")

    return {
        "report_json_path": str(json_path),
        "report_txt_path": str(txt_path),
    }
