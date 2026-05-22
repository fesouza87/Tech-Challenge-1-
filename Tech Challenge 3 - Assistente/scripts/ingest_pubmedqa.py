from __future__ import annotations

import argparse
import json
import os
from typing import Any


def write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def normalize_decision(x: Any) -> str:
    v = str(x or "").strip().lower()
    if v in {"yes", "no", "maybe"}:
        return v
    if v in {"y", "true"}:
        return "yes"
    if v in {"n", "false"}:
        return "no"
    return "maybe" if v else "maybe"


def build_protocol_text(item: dict[str, Any]) -> str:
    q = str(item.get("QUESTION") or "").strip()
    contexts = item.get("CONTEXTS") or []
    contexts = [str(c).strip() for c in contexts if str(c).strip()]
    long_answer = str(item.get("LONG_ANSWER") or "").strip()
    decision = normalize_decision(item.get("final_decision"))
    year = str(item.get("YEAR") or "").strip()
    meshes = item.get("MESHES") or []
    meshes = [str(m).strip() for m in meshes if str(m).strip()]

    lines: list[str] = []
    if q:
        lines.append(f"Pergunta (PubMedQA): {q}")
    if contexts:
        lines.append("")
        lines.append("Contextos (trechos de resumo/artigo):")
        for i, c in enumerate(contexts, start=1):
            lines.append(f"- ({i}) {c}")
    if decision:
        lines.append("")
        lines.append(f"Decisão (rótulo): {decision}")
    if long_answer:
        lines.append("")
        lines.append(f"Resposta longa: {long_answer}")
    if year or meshes:
        lines.append("")
        if year:
            lines.append(f"Ano: {year}")
        if meshes:
            lines.append("MeSH: " + ", ".join(meshes))
    return "\n".join(lines).strip()


def build_sft_messages(item: dict[str, Any]) -> list[dict[str, str]]:
    q = str(item.get("QUESTION") or "").strip()
    long_answer = str(item.get("LONG_ANSWER") or "").strip()
    decision = normalize_decision(item.get("final_decision"))
    contexts = item.get("CONTEXTS") or []
    contexts = [str(c).strip() for c in contexts if str(c).strip()]

    system = (
        "Você é um assistente clínico do hospital. Você pode resumir evidências e organizar raciocínio, "
        "mas não prescreve dose/posologia. Você sempre pede validação médica. "
        "Se o conteúdo for externo (PubMed), deixe isso explícito."
    )
    user = q if q else "Responda com base no contexto."
    if contexts:
        user = user + "\n\n" + "Contexto (PubMedQA):\n" + "\n".join([f"- {c}" for c in contexts[:4]])

    answer_parts: list[str] = []
    answer_parts.append("Fonte: PubMedQA (evidência externa; não substitui protocolos internos).")
    answer_parts.append(f"Resposta curta (rótulo): {decision}.")
    if long_answer:
        answer_parts.append(f"Resumo/explicação: {long_answer}")
    answer_parts.append("Requer validação médica; não constitui prescrição.")
    assistant = "\n".join(answer_parts).strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}, {"role": "assistant", "content": assistant}]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--src_json", required=True, help="Caminho do ori_pqal.json (ou similar).")
    p.add_argument("--out_protocol_jsonl", default=os.path.join(os.path.dirname(__file__), "..", "data", "protocols_external", "pubmedqa_pqal.jsonl"))
    p.add_argument("--out_sft_jsonl", default=os.path.join(os.path.dirname(__file__), "..", "data", "synthetic", "pubmedqa_train.jsonl"))
    p.add_argument("--limit", type=int, default=500)
    p.add_argument("--include_non_humans", action="store_true", help="Inclui entradas não-humanas (plantas, in vitro etc.).")
    args = p.parse_args()

    only_humans = not bool(args.include_non_humans)

    with open(os.path.abspath(args.src_json), "r", encoding="utf-8") as f:
        data = json.load(f)

    protocol_rows: list[dict[str, Any]] = []
    sft_rows: list[dict[str, Any]] = []

    for pmid, item in data.items():
        if not isinstance(item, dict):
            continue
        meshes = item.get("MESHES") or []
        meshes = {str(m).strip() for m in meshes if str(m).strip()}
        if only_humans and "Humans" not in meshes:
            continue

        text = build_protocol_text(item)
        if not text:
            continue
        protocol_rows.append(
            {
                "id": f"PMID:{pmid}",
                "title": str(item.get("QUESTION") or "PubMedQA"),
                "source": "pubmedqa_pqal",
                "text": text,
            }
        )
        sft_rows.append({"messages": build_sft_messages(item)})

        if args.limit and len(protocol_rows) >= int(args.limit):
            break

    write_jsonl(args.out_protocol_jsonl, protocol_rows)
    write_jsonl(args.out_sft_jsonl, sft_rows)
    print(f"Gerado RAG (externo): {os.path.abspath(args.out_protocol_jsonl)}")
    print(f"Gerado SFT (chat): {os.path.abspath(args.out_sft_jsonl)}")
    print("Para ativar RAG externo no backend, use TC3_PROTOCOL_EXTERNAL_DIR apontando para a pasta do JSONL gerado.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
