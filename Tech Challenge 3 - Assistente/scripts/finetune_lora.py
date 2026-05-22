from __future__ import annotations

import argparse
import os
from typing import Any

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer, TrainingArguments
from trl import SFTTrainer


def format_chat_example(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    parts: list[str] = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        parts.append(f"{role.upper()}: {content}")
    return "\n".join(parts)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    p.add_argument("--train_jsonl", required=True)
    p.add_argument("--out_dir", default=os.path.join(os.path.dirname(__file__), "..", "artifacts", "lora_adapter"))
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=8)
    args = p.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = load_dataset("json", data_files={"train": os.path.abspath(args.train_jsonl)})["train"]

    def _map(ex: dict[str, Any]) -> dict[str, str]:
        msgs = ex.get("messages") or []
        return {"text": format_chat_example(tokenizer, msgs)}

    ds = ds.map(_map, remove_columns=ds.column_names)

    lora = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    )

    use_fp16 = torch.cuda.is_available()
    training_args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=float(args.epochs),
        learning_rate=float(args.lr),
        per_device_train_batch_size=int(args.batch_size),
        gradient_accumulation_steps=int(args.grad_accum),
        logging_steps=10,
        save_strategy="epoch",
        fp16=bool(use_fp16),
        bf16=False,
        report_to=[],
    )

    trainer = SFTTrainer(
        model=args.model_id,
        train_dataset=ds,
        dataset_text_field="text",
        max_seq_length=1024,
        peft_config=lora,
        args=training_args,
    )
    trainer.train()
    trainer.model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"Adapter salvo em: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
