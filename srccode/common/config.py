"""Configuration: paths and Ollama provider settings.

Two modes are supported:

  * **Local Ollama desktop** (default):
        OLLAMA_HOST=http://localhost:11434      (default — no env var needed)

  * **Ollama Cloud**:
        export OLLAMA_HOST=https://ollama.com
        export OLLAMA_API_KEY=...

Both use the same `ollama` Python SDK; only the host (and optional API key)
differ. Models suitable for cloud are e.g. `gpt-oss:120b`, `qwen3-coder:480b`;
local desktop typically runs `qwen2.5-coder:7b`, `llama3.1:8b`, etc.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# --- dataset paths -----------------------------------------------------------
DATA_ROOT       = ROOT / "data" / "datasets" / "SmellyCodeDataset"
PREPARED        = ROOT / "prepared_data" / "datasets"
TEST_FILE_UNANN = PREPARED / "unannotated" / "test.json"
TEST_FILE_ANN   = PREPARED / "annotated"   / "test.json"
TRAIN_FILE_ANN  = PREPARED / "annotated"   / "train.json"

# --- prompt paths ------------------------------------------------------------
PROMPTS_ROOT  = ROOT / "prompts"
TAXONOMY_FILE = PROMPTS_ROOT / "_taxonomy.md"
SCHEMA_FILE   = PROMPTS_ROOT / "_output_schema.md"
SYSTEM_FILE   = PROMPTS_ROOT / "_system.md"

# --- output paths ------------------------------------------------------------
RESULTS_ROOT = ROOT / "results" / "llm_runs"
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

# --- Ollama provider ---------------------------------------------------------
OLLAMA_HOST    = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY", )  # only set for cloud
# "f8a369b1171545caaee783be071718f2.IHUkyMMvAdr3vaoECMoW8fAd"
# Default model names. Override per-run with --model.
DEFAULT_MODEL_LOCAL = "qwen3:0.6b"
DEFAULT_MODEL_CLOUD = "deepseek-v4-flash:cloud"

def is_cloud() -> bool:
    # Cloud mode is determined by host, not by presence of an API key
    # (some local setups may still set a key for proxy auth).
    return "ollama.com" in OLLAMA_HOST

def default_model() -> str:
    return DEFAULT_MODEL_CLOUD if is_cloud() else DEFAULT_MODEL_LOCAL
