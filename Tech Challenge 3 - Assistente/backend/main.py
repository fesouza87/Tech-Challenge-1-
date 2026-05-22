from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from assistant.audit import AuditEvent, new_request_id, now_unix, write_audit_event
from assistant.config import load_settings
from assistant.db import connect, ensure_schema, seed_synthetic
from assistant.graph import build_graph
from assistant.llm import build_llm_client
from assistant.rag import build_vectorstore


load_dotenv()


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    patient_id: str | None = None
    clinician_id: str | None = None


class ChatResponse(BaseModel):
    request_id: str
    answer: str
    alerts: list[dict[str, Any]]
    sources: list[dict[str, Any]]


def create_app() -> FastAPI:
    settings = load_settings()
    app = FastAPI(title="Tech Challenge 3 - Assistente Virtual Médico", version="0.1.0")

    base_dir = os.path.abspath(os.path.dirname(__file__))
    static_dir = os.path.join(base_dir, "static")
    if os.path.isdir(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

    conn = connect(settings.patient_db_path)
    ensure_schema(conn)
    seed_synthetic(conn)

    vectorstore = build_vectorstore(settings.protocol_dir, settings.vectorstore_dir, settings.embeddings_model, settings.protocol_external_dir)
    llm_client = build_llm_client(
        provider=settings.llm_provider,
        ollama_host=settings.ollama_host,
        ollama_model=settings.llm_model,
        anthropic_model=settings.anthropic_model,
        hf_model_id=settings.hf_model_id,
        hf_adapter_path=settings.hf_adapter_path,
    )
    compiled = build_graph(conn=conn, vectorstore=vectorstore, llm_client=llm_client)

    app.state.settings = settings
    app.state.conn = conn
    app.state.vectorstore = vectorstore
    app.state.llm = llm_client
    app.state.graph = compiled

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "llm": {"provider": llm_client.info.provider, "model": llm_client.info.model}}

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        index_path = os.path.join(static_dir, "index.html")
        if os.path.isfile(index_path):
            with open(index_path, "r", encoding="utf-8") as f:
                return f.read()
        return "<html><body><h1>UI não encontrada</h1><p>Crie backend/static/index.html</p></body></html>"

    @app.post("/api/chat", response_model=ChatResponse)
    def chat(req: ChatRequest) -> ChatResponse:
        request_id = new_request_id()
        state_in = {"clinician_id": req.clinician_id, "patient_id": req.patient_id, "message": req.message}
        state_out = app.state.graph.invoke(state_in)
        answer = str(state_out.get("answer") or "")
        alerts = list(state_out.get("alerts") or [])
        sources = list(state_out.get("retrieved") or [])

        write_audit_event(
            settings.audit_log_path,
            AuditEvent(
                request_id=request_id,
                ts_unix=now_unix(),
                clinician_id=req.clinician_id,
                patient_id=req.patient_id,
                input_message=req.message,
                decision_flow="langgraph:policy->patient->pending_exams->retrieval->answer",
                model=f"{llm_client.info.provider}:{llm_client.info.model}",
                retrieval=sources,
                output_text=answer,
                policy={
                    "allowed": bool(state_out.get("policy_allowed", True)),
                    "reason": str(state_out.get("policy_reason") or ""),
                    "flags": dict(state_out.get("policy_flags") or {}),
                },
            ),
        )

        return ChatResponse(request_id=request_id, answer=answer, alerts=alerts, sources=sources)

    return app


app = create_app()
