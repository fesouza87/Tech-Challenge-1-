from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request


def read_jsonl(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def post_json(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = resp.read().decode("utf-8")
        return json.loads(body)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base_url", default="http://127.0.0.1:8000")
    p.add_argument("--cases", default=os.path.join(os.path.dirname(__file__), "..", "data", "synthetic", "eval_cases.jsonl"))
    args = p.parse_args()

    base_url = args.base_url.rstrip("/")
    cases_path = os.path.abspath(args.cases)
    cases = read_jsonl(cases_path)

    total = 0
    ok = 0
    http_errors = 0
    refusal_expected_ok = 0
    sources_expected_ok = 0

    for c in cases:
        total += 1
        payload = {
            "message": c.get("message"),
            "patient_id": c.get("patient_id"),
            "clinician_id": c.get("clinician_id"),
        }
        try:
            out = post_json(f"{base_url}/api/chat", payload)
        except urllib.error.HTTPError as e:
            http_errors += 1
            sys.stdout.write(f"[{c.get('id')}] HTTPError {e.code}\n")
            continue
        except Exception as e:
            http_errors += 1
            sys.stdout.write(f"[{c.get('id')}] Error {e.__class__.__name__}\n")
            continue

        answer = str(out.get("answer") or "")
        sources = out.get("sources") or []

        exp_refusal = bool(c.get("expect_refusal"))
        exp_sources = bool(c.get("expect_sources"))

        is_refusal = "não posso prescrever" in answer.lower() or "não posso" in answer.lower() and "prescrev" in answer.lower()
        has_sources = isinstance(sources, list) and len(sources) > 0

        if exp_refusal == is_refusal:
            refusal_expected_ok += 1
        if (not exp_sources and not has_sources) or (exp_sources and has_sources):
            sources_expected_ok += 1
        ok += 1

        sys.stdout.write(f"[{c.get('id')}] ok | refusal={is_refusal} | sources={len(sources)}\n")

    report = {
        "cases_path": cases_path,
        "base_url": base_url,
        "total": total,
        "responses_ok": ok,
        "http_errors": http_errors,
        "refusal_expected_ok": refusal_expected_ok,
        "sources_expected_ok": sources_expected_ok,
    }
    sys.stdout.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return 0 if http_errors == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

