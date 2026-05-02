"""Thin wrapper around the `ollama` SDK that works for both local and cloud.

Usage:
    from common.ollama_client import chat
    text = chat(model="qwen2.5-coder:7b",
                system="You are ...",
                user="Detect smells in ...")
"""
from __future__ import annotations

import os
from typing import Optional

import ollama

from . import config


def _client() -> ollama.Client:
    headers = {}
    if config.OLLAMA_API_KEY:
        headers["Authorization"] = f"Bearer {config.OLLAMA_API_KEY}"
    return ollama.Client(host=config.OLLAMA_HOST, headers=headers)


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
    """Send a single chat turn.

    Returns:
        (assistant_text, usage_dict) where usage_dict has keys
        ``input_tokens``, ``output_tokens``, ``total_tokens`` (ints; may be
        0 if the backend did not report counts).
    """
    options = {"temperature": temperature, "seed": seed}
    if num_ctx is not None:
        options["num_ctx"] = num_ctx
    if num_predict is not None:
        options["num_predict"] = num_predict

    resp = _client().chat(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        options=options,
    )
    text = resp["message"]["content"]

    # Ollama returns prompt_eval_count / eval_count on the response object.
    in_tok  = int(getattr(resp, "prompt_eval_count", 0) or resp.get("prompt_eval_count", 0) or 0) \
        if hasattr(resp, "get") else int(getattr(resp, "prompt_eval_count", 0) or 0)
    out_tok = int(getattr(resp, "eval_count", 0) or resp.get("eval_count", 0) or 0) \
        if hasattr(resp, "get") else int(getattr(resp, "eval_count", 0) or 0)
    usage = {
        "input_tokens":  in_tok,
        "output_tokens": out_tok,
        "total_tokens":  in_tok + out_tok,
    }
    return text, usage


def health_check() -> str:
    """Quick sanity check; raises on failure."""
    try:
        _client().list()
    except Exception as e:  # noqa: BLE001
        host = config.OLLAMA_HOST
        cloud = " (cloud)" if config.is_cloud() else " (local)"
        raise RuntimeError(f"Cannot reach Ollama at {host}{cloud}: {e}") from e
    return f"OK: {config.OLLAMA_HOST}{' (cloud)' if config.is_cloud() else ' (local)'}"
