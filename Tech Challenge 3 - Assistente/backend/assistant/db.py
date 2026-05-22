from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PatientSnapshot:
    patient_id: str
    name_masked: str
    age: int
    sex: str
    allergies: str
    comorbidities: str
    last_visit_summary: str
    pending_exams: list[dict[str, Any]]
    recent_labs: list[dict[str, Any]]


def connect(db_path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_schema(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS patients (
            patient_id TEXT PRIMARY KEY,
            name_masked TEXT NOT NULL,
            age INTEGER NOT NULL,
            sex TEXT NOT NULL,
            allergies TEXT NOT NULL,
            comorbidities TEXT NOT NULL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS visits (
            visit_id TEXT PRIMARY KEY,
            patient_id TEXT NOT NULL,
            ts TEXT NOT NULL,
            summary TEXT NOT NULL,
            FOREIGN KEY(patient_id) REFERENCES patients(patient_id)
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS pending_exams (
            exam_id TEXT PRIMARY KEY,
            patient_id TEXT NOT NULL,
            name TEXT NOT NULL,
            status TEXT NOT NULL,
            requested_ts TEXT NOT NULL,
            FOREIGN KEY(patient_id) REFERENCES patients(patient_id)
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS labs (
            lab_id TEXT PRIMARY KEY,
            patient_id TEXT NOT NULL,
            name TEXT NOT NULL,
            value TEXT NOT NULL,
            unit TEXT NOT NULL,
            ts TEXT NOT NULL,
            flag TEXT NOT NULL,
            FOREIGN KEY(patient_id) REFERENCES patients(patient_id)
        )
        """
    )
    conn.commit()


def seed_synthetic(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) AS n FROM patients")
    if int(cur.fetchone()["n"]) > 0:
        return

    patients = [
        ("P-0001", "PACIENTE A*** S***", 54, "F", "penicilina", "HAS; DM2"),
        ("P-0002", "PACIENTE B*** M***", 38, "M", "sem alergias conhecidas", "asma"),
    ]
    cur.executemany("INSERT INTO patients(patient_id, name_masked, age, sex, allergies, comorbidities) VALUES(?,?,?,?,?,?)", patients)

    visits = [
        ("V-1001", "P-0001", "2026-05-10T10:12:00", "Dor torácica atípica. ECG sem isquemia. Orientado retorno se piora."),
        ("V-1002", "P-0002", "2026-05-18T14:20:00", "Dispneia leve em contexto de crise asmática. Respondeu a broncodilatador."),
    ]
    cur.executemany("INSERT INTO visits(visit_id, patient_id, ts, summary) VALUES(?,?,?,?)", visits)

    pending = [
        ("E-9001", "P-0001", "Troponina", "pendente", "2026-05-21T08:05:00"),
        ("E-9002", "P-0001", "Raio-X Tórax", "pendente", "2026-05-21T08:05:00"),
        ("E-9003", "P-0002", "Peak Flow", "pendente", "2026-05-21T09:40:00"),
    ]
    cur.executemany("INSERT INTO pending_exams(exam_id, patient_id, name, status, requested_ts) VALUES(?,?,?,?,?)", pending)

    labs = [
        ("L-7001", "P-0001", "Glicemia", "168", "mg/dL", "2026-05-10T10:30:00", "alto"),
        ("L-7002", "P-0001", "Creatinina", "0.9", "mg/dL", "2026-05-10T10:30:00", "normal"),
        ("L-7003", "P-0002", "Saturação O2", "96", "%", "2026-05-18T14:25:00", "normal"),
    ]
    cur.executemany("INSERT INTO labs(lab_id, patient_id, name, value, unit, ts, flag) VALUES(?,?,?,?,?,?,?)", labs)
    conn.commit()


def get_patient_snapshot(conn: sqlite3.Connection, patient_id: str) -> PatientSnapshot | None:
    cur = conn.cursor()
    cur.execute("SELECT * FROM patients WHERE patient_id = ?", (patient_id,))
    p = cur.fetchone()
    if p is None:
        return None

    cur.execute(
        "SELECT ts, summary FROM visits WHERE patient_id = ? ORDER BY ts DESC LIMIT 1",
        (patient_id,),
    )
    v = cur.fetchone()
    last_summary = str(v["summary"]) if v is not None else "Sem visitas registradas."

    cur.execute(
        "SELECT exam_id, name, status, requested_ts FROM pending_exams WHERE patient_id = ? ORDER BY requested_ts DESC",
        (patient_id,),
    )
    pending_exams = [dict(r) for r in cur.fetchall()]

    cur.execute(
        "SELECT name, value, unit, ts, flag FROM labs WHERE patient_id = ? ORDER BY ts DESC LIMIT 8",
        (patient_id,),
    )
    recent_labs = [dict(r) for r in cur.fetchall()]

    return PatientSnapshot(
        patient_id=str(p["patient_id"]),
        name_masked=str(p["name_masked"]),
        age=int(p["age"]),
        sex=str(p["sex"]),
        allergies=str(p["allergies"]),
        comorbidities=str(p["comorbidities"]),
        last_visit_summary=last_summary,
        pending_exams=pending_exams,
        recent_labs=recent_labs,
    )
