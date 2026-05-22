from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    app_env: str
    data_dir: str
    audit_log_path: str
    patient_db_path: str
    protocol_dir: str
    protocol_external_dir: str
    vectorstore_dir: str
    embeddings_model: str
    llm_provider: str
    llm_model: str
    ollama_host: str
    anthropic_model: str
    hf_model_id: str
    hf_adapter_path: str | None


def load_settings() -> Settings:
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    data_dir = os.environ.get("TC3_DATA_DIR", os.path.join(base_dir, "data"))
    audit_log_path = os.environ.get("TC3_AUDIT_LOG", os.path.join(base_dir, "logs", "audit.jsonl"))
    patient_db_path = os.environ.get("TC3_PATIENT_DB", os.path.join(data_dir, "patients.db"))
    protocol_dir = os.environ.get("TC3_PROTOCOL_DIR", os.path.join(data_dir, "protocols"))
    protocol_external_dir = os.environ.get("TC3_PROTOCOL_EXTERNAL_DIR", os.path.join(data_dir, "protocols_external"))
    vectorstore_dir = os.environ.get("TC3_VECTORSTORE_DIR", os.path.join(data_dir, "vectorstore"))
    embeddings_model = os.environ.get("TC3_EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    llm_provider = os.environ.get("TC3_LLM_PROVIDER", "ollama").strip().lower()
    llm_model = os.environ.get("TC3_LLM_MODEL", "llama3.1").strip()
    ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    anthropic_model = os.environ.get("TC3_ANTHROPIC_MODEL", "claude-sonnet-4-6").strip()
    hf_model_id = os.environ.get("TC3_HF_MODEL_ID", "TinyLlama/TinyLlama-1.1B-Chat-v1.0").strip()
    hf_adapter_path = os.environ.get("TC3_HF_ADAPTER_PATH", "").strip() or None
    app_env = os.environ.get("TC3_ENV", "dev").strip().lower()

    return Settings(
        app_env=app_env,
        data_dir=data_dir,
        audit_log_path=audit_log_path,
        patient_db_path=patient_db_path,
        protocol_dir=protocol_dir,
        protocol_external_dir=protocol_external_dir,
        vectorstore_dir=vectorstore_dir,
        embeddings_model=embeddings_model,
        llm_provider=llm_provider,
        llm_model=llm_model,
        ollama_host=ollama_host,
        anthropic_model=anthropic_model,
        hf_model_id=hf_model_id,
        hf_adapter_path=hf_adapter_path,
    )
