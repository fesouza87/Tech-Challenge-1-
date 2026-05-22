from __future__ import annotations

import re
from dataclasses import dataclass


_RX_PRESCRIPTION = re.compile(
    r"\b(prescrev(a|er)|receit(a|ar)|dose|posologi(a|ia)|mg\b|ml\b|q\d+h|a cada \d+ horas)\b",
    re.IGNORECASE,
)
_RX_CONTROLLED = re.compile(r"\b(opioide|morfina|fentanil|benzodiazepin|midazolam)\b", re.IGNORECASE)
_RX_EMERGENCY = re.compile(r"\b(parada|reanima|ressuscita|choque|sepse grave)\b", re.IGNORECASE)


@dataclass(frozen=True)
class PolicyResult:
    allowed: bool
    reason: str
    flags: dict[str, bool]


def check_policy(user_message: str) -> PolicyResult:
    msg = user_message.strip()
    flags = {
        "asks_prescription": bool(_RX_PRESCRIPTION.search(msg)),
        "mentions_controlled": bool(_RX_CONTROLLED.search(msg)),
        "mentions_emergency": bool(_RX_EMERGENCY.search(msg)),
    }

    if flags["asks_prescription"] or flags["mentions_controlled"]:
        return PolicyResult(
            allowed=False,
            reason="O assistente não pode prescrever, sugerir dose/posologia ou orientar uso de medicamentos controlados sem validação humana.",
            flags=flags,
        )

    return PolicyResult(allowed=True, reason="ok", flags=flags)


def safe_refusal_text(reason: str) -> str:
    return (
        "Posso ajudar com informações gerais e com os protocolos internos, mas não posso prescrever ou indicar dose/posologia.\n"
        f"Motivo: {reason}\n"
        "Se você quiser, descreva o quadro clínico, exames, alergias e comorbidades, que eu organizo hipóteses e condutas possíveis para revisão médica."
    )
