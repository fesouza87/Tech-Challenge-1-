from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import asdict
from typing import Any

from routes import Instance, RoutePlan, compute_route_distance, compute_route_load

"""
Integração de LLMs e geração textual para o Tech Challenge 2 - Rotas.

Funções principais:
- generate_llm_report: gera relatório final usando LLM (Ollama/OpenAI) com contexto JSON e comparativo.
- answer_question: responde perguntas sobre o plano usando regras simples e, se habilitado, LLM.
- format_summary_report / format_driver_instructions: geração clássica (sem LLM) do relatório.

Variáveis de ambiente:
- LLM_ENABLE: '1'/'true' para habilitar uso de LLM
- LLM_PROVIDER: 'ollama' (default)
- LLM_MODEL: nome do modelo (ex.: 'deepseek-v3:671b-cloud' no Ollama)
- OLLAMA_HOST: URL do servidor Ollama (default http://localhost:11434)
- LLM_TEMPERATURE / LLM_NUM_PREDICT: parâmetros do Ollama para controlar geração
"""

def _http_post_json(url: str, payload: dict, headers: dict[str, str]) -> dict:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    timeout_s = float(os.environ.get("LLM_HTTP_TIMEOUT", "1200"))
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        body = response.read().decode("utf-8")
        return json.loads(body)


def _ollama_generate(prompt: str) -> str:
    """Chama Ollama /api/generate com prompt e opções (temperature, num_predict)."""
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    model = os.environ.get("LLM_MODEL", "llama3.1")
    temperature = float(os.environ.get("LLM_TEMPERATURE", "0.2"))
    num_predict = int(os.environ.get("LLM_NUM_PREDICT", "900"))
    url = f"{host}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
        },
    }
    out = _http_post_json(url, payload, headers={"Content-Type": "application/json"})
    if out.get("error"):
        raise ValueError(str(out["error"]))
    text = out.get("response")
    if (text is None or str(text).strip() == "") and out.get("thinking"):
        raise ValueError("Modelo retornou apenas 'thinking' sem resposta final. Aumente LLM_NUM_PREDICT ou use outro modelo (ex.: qwen3:30b).")
    return str(text) if text is not None else ""


def llm_enabled() -> bool:
    value = os.environ.get("LLM_ENABLE", "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def generate_text_with_llm(prompt: str) -> str:
    provider = os.environ.get("LLM_PROVIDER", "ollama").strip().lower()
    return _ollama_generate(prompt)


def format_driver_instructions(instance: Instance, plan: RoutePlan) -> str:
    """Gera instruções operacionais por veículo (sem LLM)."""
    lines: list[str] = []
    lines.append("Instrucoes de Entrega")
    lines.append("")
    for vehicle in instance.vehicles:
        route = plan.routes_by_vehicle.get(vehicle.id, [])
        if not route:
            lines.append(f"- Veiculo {vehicle.id}: sem entregas")
            continue
        load = compute_route_load(instance, route)
        dist = compute_route_distance(instance, route)
        lines.append(f"- Veiculo {vehicle.id}: carga {load:.1f}/{vehicle.capacity:.1f}, distancia {dist:.1f}/{vehicle.max_distance:.1f}")
        for i, delivery_id in enumerate(route, start=1):
            delivery = instance.deliveries[delivery_id]
            loc = instance.locations[delivery.location_id]
            lines.append(f"  {i}. {loc.name} ({loc.id}) | entrega {delivery.id} | prioridade {delivery.priority} | demanda {delivery.demand:.1f}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def format_summary_report(instance: Instance, plan: RoutePlan) -> str:
    """Resumo do plano: viabilidade, custo total e penalidades (sem LLM)."""
    lines: list[str] = []
    lines.append("Relatorio de Rotas")
    lines.append("")
    lines.append(f"Feasible: {plan.feasible}")
    lines.append(f"Custo total (distancia + penalidades): {plan.total_distance:.2f}")
    if plan.penalties:
        lines.append("Penalidades:")
        for k in sorted(plan.penalties.keys()):
            lines.append(f"- {k}: {plan.penalties[k]:.2f}")
    total_assigned = sum(len(plan.routes_by_vehicle.get(v.id, [])) for v in instance.vehicles)
    total_deliveries = len(instance.deliveries)
    lines.append(f"Entregas alocadas: {total_assigned}/{total_deliveries}")
    return "\n".join(lines).strip() + "\n"


def answer_question_classic(instance: Instance, plan: RoutePlan, question: str) -> str:
    """Responde perguntas comuns sobre o plano usando regras simples (sem LLM)."""
    q = question.strip().lower()
    if not q:
        return "Pergunta vazia.\n"
    if "distancia" in q and ("total" in q or "hoje" in q):
        return f"A distancia total (com penalidades) e {plan.total_distance:.2f}.\n"
    if "viavel" in q or "feasible" in q:
        return f"O plano e viavel: {plan.feasible}.\n"
    if "critico" in q or "critical" in q:
        critical = {d.id for d in instance.deliveries.values() if d.priority.lower() == 'critical'}
        by_vehicle: dict[str, list[str]] = {}
        for vehicle in instance.vehicles:
            route = plan.routes_by_vehicle.get(vehicle.id, [])
            hits = [d for d in route if d in critical]
            if hits:
                by_vehicle[vehicle.id] = hits
        if not by_vehicle:
            return "Nao ha entregas criticas alocadas no plano.\n"
        return json.dumps(by_vehicle, ensure_ascii=False, indent=2) + "\n"
    return "Nao encontrei uma resposta direta. Pergunte sobre distancia total, viabilidade ou entregas criticas.\n"


def answer_question_llm(instance: Instance, plan: RoutePlan, question: str) -> str:
    """Responde usando LLM com contexto JSON. Retorna string vazia em falha/indisponibilidade."""
    if not question.strip():
        return ""
    context = export_context(instance, plan)
    prompt = "\n".join(
        [
            "Você responde perguntas sobre rotas e entregas usando SOMENTE o Contexto JSON fornecido.",
            "Regras:",
            "- Se a resposta não estiver no contexto, diga exatamente: 'Não tenho dados suficientes no contexto para responder.'",
            "- Responda em português (Brasil), em no máximo 6 linhas.",
            "- Use números e listas curtas quando ajudar.",
            "",
            "Contexto JSON:",
            json.dumps(context, ensure_ascii=False),
            "",
            f"Pergunta: {question}",
        ]
    )
    try:
        text = generate_text_with_llm(prompt).strip()
    except (urllib.error.URLError, urllib.error.HTTPError, ValueError):
        text = ""
    return (text + "\n") if text else ""


def answer_question(instance: Instance, plan: RoutePlan, question: str) -> str:
    """
    Respostas rápidas sobre o plano por regras simples.
    Se LLM estiver habilitado, tenta responder usando o contexto JSON.
    """
    classic = answer_question_classic(instance, plan, question)
    if classic.startswith("Nao encontrei") and llm_enabled():
        llm_text = answer_question_llm(instance, plan, question)
        if llm_text:
            return llm_text
    return classic


def export_context(instance: Instance, plan: RoutePlan) -> dict[str, Any]:
    return {"instance": asdict(instance), "plan": {"routes_by_vehicle": plan.routes_by_vehicle, "total_distance": plan.total_distance, "feasible": plan.feasible}}


def generate_llm_report(attribution: str, comparison: str, instance: Instance, plan: RoutePlan) -> str:
    """
    Gera relatório final via LLM usando:
    - Seção fixa de atribuição
    - Bloco de comparativo calculado
    - Contexto JSON (instância e plano)
    O texto é retornado em português e com estrutura de seções definidas.
    """
    context = export_context(instance, plan)
    prompt = "\n".join(
        [
            "Gere um relatório em português (Brasil) para equipes hospitalares e coordenação logística.",
            "Regras de formato:",
            "- Texto simples (sem Markdown).",
            "- Use estas seções nesta ordem: BASE, RESUMO, COMPARATIVO, INSTRUCOES, ALERTAS, SUGESTOES.",
            "- Em INSTRUCOES, gere uma subseção por veículo no formato: 'VEICULO <id>:' seguido de passos numerados.",
            "- Não invente dados; use somente o Contexto JSON e o comparativo fornecido.",
            "",
            "BASE (copie exatamente):",
            attribution.strip().rstrip(),
            "",
            "COMPARATIVO (use exatamente estes números):",
            comparison.strip().rstrip(),
            "",
            "CONTEXTO JSON (instância e solução):",
            json.dumps(context, ensure_ascii=False),
        ]
    )
    text = generate_text_with_llm(prompt).strip()
    return text + "\n"
