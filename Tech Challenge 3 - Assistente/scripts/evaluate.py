from __future__ import annotations

import argparse
import os

from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    p.add_argument("--adapter", default="")
    p.add_argument("--prompt", default="Como devo organizar a triagem de dor torácica segundo o protocolo interno?")
    p.add_argument("--max_new_tokens", type=int, default=250)
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.base_model, device_map="auto")
    if args.adapter.strip():
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, os.path.abspath(args.adapter))

    messages = [{"role": "system", "content": "Você é um assistente médico do hospital, baseado em protocolos internos. Você nunca prescreve."}, {"role": "user", "content": args.prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = f"SYSTEM: {messages[0]['content']}\nUSER: {messages[1]['content']}\nASSISTANT:"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=int(args.max_new_tokens), do_sample=True, temperature=0.2)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

