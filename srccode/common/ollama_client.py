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
) -> str:
    """Send a single chat turn and return the assistant text.

    Deterministic by default (temperature=0, fixed seed) — required for
    reproducible research runs.
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
    return resp["message"]["content"]


def health_check() -> str:
    """Quick sanity check; raises on failure."""
    try:
        _client().list()
    except Exception as e:  # noqa: BLE001
        host = config.OLLAMA_HOST
        cloud = " (cloud)" if config.is_cloud() else " (local)"
        raise RuntimeError(f"Cannot reach Ollama at {host}{cloud}: {e}") from e
    return f"OK: {config.OLLAMA_HOST}{' (cloud)' if config.is_cloud() else ' (local)'}"
