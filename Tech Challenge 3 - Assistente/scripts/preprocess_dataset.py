from __future__ import annotations

import argparse
import json
import os
import re


_CPF = re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b")
_PHONE = re.compile(r"\b(?:\+?55\s*)?(?:\(?\d{2}\)?\s*)?\d{4,5}-?\d{4}\b")
_EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_DATE = re.compile(r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})\b")


def anonymize_text(text: str) -> str:
    t = text
    t = _CPF.sub("[CPF_REMOVIDO]", t)
    t = _PHONE.sub("[TEL_REMOVIDO]", t)
    t = _EMAIL.sub("[EMAIL_REMOVIDO]", t)
    t = _DATE.sub("[DATA]", t)
    return t


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--in_jsonl", required=True)
    p.add_argument("--out_jsonl", required=True)
    p.add_argument("--field", default="text", help="Campo de texto a anonimizar (ex.: text).")
    args = p.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out_jsonl)), exist_ok=True)

    written = 0
    with open(args.in_jsonl, "r", encoding="utf-8") as fin, open(args.out_jsonl, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if args.field in obj and isinstance(obj[args.field], str):
                obj[args.field] = anonymize_text(obj[args.field])
            if "messages" in obj and isinstance(obj["messages"], list):
                for m in obj["messages"]:
                    if isinstance(m, dict) and isinstance(m.get("content"), str):
                        m["content"] = anonymize_text(m["content"])
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            written += 1

    print(f"Linhas processadas: {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

