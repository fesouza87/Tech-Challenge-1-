from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AuditEvent:
    request_id: str
    ts_unix: float
    clinician_id: str | None
    patient_id: str | None
    input_message: str
    decision_flow: str
    model: str
    retrieval: list[dict[str, Any]]
    output_text: str
    policy: dict[str, Any]


_lock = threading.Lock()


def new_request_id() -> str:
    return str(uuid.uuid4())


def write_audit_event(path: str, event: AuditEvent) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "request_id": event.request_id,
        "ts_unix": event.ts_unix,
        "clinician_id": event.clinician_id,
        "patient_id": event.patient_id,
        "input_message": event.input_message,
        "decision_flow": event.decision_flow,
        "model": event.model,
        "retrieval": event.retrieval,
        "output_text": event.output_text,
        "policy": event.policy,
    }
    line = json.dumps(payload, ensure_ascii=False)
    with _lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def now_unix() -> float:
    return time.time()
