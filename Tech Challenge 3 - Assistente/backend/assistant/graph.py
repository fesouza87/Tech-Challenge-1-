from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from assistant.db import PatientSnapshot, get_patient_snapshot
from assistant.policy import check_policy, safe_refusal_text
from assistant.rag import chunks_to_citations, retrieve_chunks


class AssistantState(TypedDict, total=False):
    clinician_id: str | None
    patient_id: str | None
    message: str
    policy_allowed: bool
    policy_reason: str
    policy_flags: dict[str, bool]
    patient: PatientSnapshot | None
    alerts: list[dict[str, Any]]
    retrieved: list[dict[str, Any]]
    answer: str


def build_graph(*, conn, vectorstore, llm_client):
    def format_llm_error_text(exc: Exception) -> str:
        provider = getattr(getattr(llm_client, "info", None), "provider", "llm")
        model = getattr(getattr(llm_client, "info", None), "model", "")
        header = f"Falha ao consultar o LLM configurado ({provider}{':' + model if model else ''})."

        name = exc.__class__.__name__
        module = getattr(exc.__class__, "__module__", "")
        details = ""
        if module.startswith("anthropic"):
            if name == "NotFoundError":
                details = "O modelo Anthropic configurado não foi encontrado. Ajuste TC3_ANTHROPIC_MODEL no .env para um modelo válido/habilitado na sua conta (ex.: claude-sonnet-4-6)."
            elif name in {"AuthenticationError", "PermissionDeniedError"}:
                details = "Falha de autenticação/permissão. Verifique ANTHROPIC_API_KEY e o acesso ao modelo na sua conta."
            elif name == "RateLimitError":
                details = "Limite de requisições atingido na Anthropic. Tente novamente ou ajuste o plano/limites."
            else:
                details = f"Erro Anthropic ({name})."
        else:
            details = f"Erro ({name})."

        return "\n".join([header, details]).strip()

    def node_policy(state: AssistantState) -> AssistantState:
        r = check_policy(state["message"])
        state["policy_allowed"] = r.allowed
        state["policy_reason"] = r.reason
        state["policy_flags"] = r.flags
        if not r.allowed:
            state["answer"] = safe_refusal_text(r.reason)
        return state

    def node_patient(state: AssistantState) -> AssistantState:
        pid = state.get("patient_id")
        if pid:
            state["patient"] = get_patient_snapshot(conn, pid)
        else:
            state["patient"] = None
        return state

    def node_pending_exams(state: AssistantState) -> AssistantState:
        alerts: list[dict[str, Any]] = []
        patient = state.get("patient")
        if patient and patient.pending_exams:
            for e in patient.pending_exams:
                if str(e.get("status", "")).lower() == "pendente":
                    alerts.append(
                        {
                            "type": "exame_pendente",
                            "message": f"Exame pendente: {e.get('name')} (solicitado em {e.get('requested_ts')}).",
                        }
                    )
        state["alerts"] = alerts
        return state

    def node_retrieval(state: AssistantState) -> AssistantState:
        query = state["message"]
        patient = state.get("patient")
        if patient:
            query = (
                f"{query}\n\nContexto do paciente: idade {patient.age}, sexo {patient.sex}, "
                f"alergias: {patient.allergies}, comorbidades: {patient.comorbidities}, "
                f"última visita: {patient.last_visit_summary}."
            )
        chunks = retrieve_chunks(vectorstore, query=query, k=4)
        state["retrieved"] = chunks_to_citations(chunks)
        return state

    def node_answer(state: AssistantState) -> AssistantState:
        if not state.get("policy_allowed", True):
            return state

        patient = state.get("patient")
        retrieved = state.get("retrieved") or []
        alerts = state.get("alerts") or []

        patient_block = "Paciente: não informado."
        if patient:
            patient_block = (
                "Paciente:\n"
                f"- ID: {patient.patient_id}\n"
                f"- Nome (mascarado): {patient.name_masked}\n"
                f"- Idade/sexo: {patient.age}/{patient.sex}\n"
                f"- Alergias: {patient.allergies}\n"
                f"- Comorbidades: {patient.comorbidities}\n"
                f"- Última visita: {patient.last_visit_summary}\n"
                f"- Exames pendentes: {len(patient.pending_exams)}\n"
            )

        citations_text = "\n".join(
            [f"[{i+1}] {c.get('title')} | {c.get('source')} | {c.get('excerpt')}" for i, c in enumerate(retrieved)]
        ).strip()
        alerts_text = "\n".join([f"- {a.get('message')}" for a in alerts]).strip()

        prompt = "\n".join(
            [
                "Você é um assistente virtual médico do hospital.",
                "Objetivo: ajudar médicos a organizar condutas e dúvidas com base em protocolos internos e no contexto do paciente.",
                "Limites obrigatórios:",
                "- Nunca prescreva (sem doses/posologia).",
                "- Não substitui avaliação médica. Sempre peça validação humana antes de qualquer conduta.",
                "- Não invente exames, diagnósticos, protocolos ou dados do paciente.",
                "",
                "Formato de resposta:",
                "1) Resumo do caso (1-3 linhas)",
                "2) Hipóteses/possibilidades (bullet points, se aplicável)",
                "3) Próximos passos sugeridos (sem prescrição, com checagens e segurança)",
                "4) Alertas (se houver)",
                "5) Fontes citadas (referencie [1], [2]...)",
                "",
                patient_block,
                "",
                f"Pergunta do médico: {state['message']}",
                "",
                "Trechos recuperados (protocolos internos e evidência externa, para citar):",
                citations_text if citations_text else "(nenhum trecho recuperado)",
                "",
                "Alertas automáticos:",
                alerts_text if alerts_text else "(nenhum alerta)",
            ]
        )

        try:
            text = llm_client.generate(prompt).strip()
        except Exception as exc:
            text = format_llm_error_text(exc)
        disclaimer = "Este conteúdo é apoio à decisão e requer validação médica; não constitui prescrição."
        if disclaimer.lower() not in text.lower():
            text = text + "\n\n" + disclaimer
        state["answer"] = text
        return state

    graph = StateGraph(AssistantState)
    graph.add_node("policy", node_policy)
    graph.add_node("load_patient", node_patient)
    graph.add_node("pending_exams", node_pending_exams)
    graph.add_node("retrieval", node_retrieval)
    graph.add_node("generate_answer", node_answer)

    graph.set_entry_point("policy")
    graph.add_conditional_edges(
        "policy",
        lambda s: "generate_answer" if not s.get("policy_allowed", True) else "load_patient",
        {"generate_answer": "generate_answer", "load_patient": "load_patient"},
    )
    graph.add_edge("load_patient", "pending_exams")
    graph.add_edge("pending_exams", "retrieval")
    graph.add_edge("retrieval", "generate_answer")
    graph.add_edge("generate_answer", END)
    return graph.compile()
