from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LLMInfo:
    provider: str
    model: str


class LLMClient:
    def __init__(self, impl, info: LLMInfo) -> None:
        self._impl = impl
        self.info = info

    def generate(self, prompt: str) -> str:
        out = self._impl.invoke(prompt)
        if isinstance(out, str):
            return out
        content = getattr(out, "content", None)
        return str(content) if content is not None else str(out)


def build_llm_client(
    *,
    provider: str,
    ollama_host: str,
    ollama_model: str,
    anthropic_model: str,
    hf_model_id: str,
    hf_adapter_path: str | None,
):
    provider = provider.strip().lower()
    if provider == "ollama":
        from langchain_community.llms.ollama import Ollama

        impl = Ollama(base_url=ollama_host, model=ollama_model, temperature=0.2)
        return LLMClient(impl, LLMInfo(provider="ollama", model=ollama_model))

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        impl = ChatAnthropic(model=anthropic_model, temperature=0.2)
        return LLMClient(impl, LLMInfo(provider="anthropic", model=anthropic_model))

    if provider == "hf":
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

        tokenizer = AutoTokenizer.from_pretrained(hf_model_id, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(hf_model_id, device_map="auto")
        if hf_adapter_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, hf_adapter_path)
        gen = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.2,
            return_full_text=False,
        )
        from langchain_community.llms import HuggingFacePipeline

        impl = HuggingFacePipeline(pipeline=gen)
        return LLMClient(impl, LLMInfo(provider="hf", model=hf_model_id))

    raise ValueError(f"TC3_LLM_PROVIDER inválido: {provider}. Use 'anthropic', 'ollama' ou 'hf'.")
