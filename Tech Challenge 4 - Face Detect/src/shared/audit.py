from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AuditEvent:
    ts_unix: float
    event_type: str
    patient_id: str
    modality: str
    payload: dict[str, Any]


_lock = threading.Lock()


def now_unix() -> float:
    return time.time()


def write_audit_event(path: str, event: AuditEvent) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    line = json.dumps(
        {
            "ts_unix": event.ts_unix,
            "event_type": event.event_type,
            "patient_id": event.patient_id,
            "modality": event.modality,
            "payload": event.payload,
        },
        ensure_ascii=False,
        default=str,
    )
    with _lock:
        with open(path, "a", encoding="utf-8") as file_handle:
            file_handle.write(line + "\n")
