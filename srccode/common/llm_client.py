"""Provider-agnostic LLM dispatcher.

Routes `chat()` and `health_check()` to the active backend selected by
`config.PROVIDER` ('local' | 'cloud' | 'bedrock').

All backends return the same tuple shape:
    (assistant_text: str, usage: {"input_tokens", "output_tokens", "total_tokens"})
"""
from __future__ import annotations

from typing import Optional

from . import bedrock_client, config, ollama_client


def chat(
    model: str,
    system: str,
    user: str,
    *,
    temperature: float = 0.0,
    seed: int = 42,
    num_ctx: Optional[int] = 8192,
    num_predict: Optional[int] = 4096,
) -> tuple[str, dict]:
    if config.is_bedrock():
        return bedrock_client.chat(
            model, system, user,
            temperature=temperature, seed=seed,
            num_ctx=num_ctx, num_predict=num_predict,
        )
    return ollama_client.chat(
        model, system, user,
        temperature=temperature, seed=seed,
        num_ctx=num_ctx, num_predict=num_predict,
    )


def health_check() -> str:
    if config.is_bedrock():
        return bedrock_client.health_check()
    return ollama_client.health_check()


def provider_name() -> str:
    return config.PROVIDER
