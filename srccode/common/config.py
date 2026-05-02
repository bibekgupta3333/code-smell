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
OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY", "")  # only set for cloud

# Default model names. Override per-run with --model.
DEFAULT_MODEL_LOCAL = "qwen3:0.6b"
DEFAULT_MODEL_CLOUD = "deepseek-v4-flash:cloud"

# --- AWS Bedrock provider ----------------------------------------------------
# Uses the boto3 `bedrock-runtime` `converse` API (unified across model families,
# returns input/output token usage).
#
# Authentication — pick ONE:
#   1. Standard AWS credential chain (aws configure / SSO / env vars / role)
#   2. Short-term **Bedrock API key** (bearer token) — set BEDROCK_API_KEY or
#      AWS_BEARER_TOKEN_BEDROCK. boto3 picks this up automatically and uses it
#      as `Authorization: Bearer <key>` for bedrock-runtime calls.
#
# Region defaults to ap-southeast-2 (Sydney). Override with AWS_REGION env.
AWS_REGION         = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "ap-southeast-2"
BEDROCK_PROFILE    = os.environ.get("AWS_PROFILE")  # optional named profile
BEDROCK_API_KEY    = os.environ.get("BEDROCK_API_KEY") or os.environ.get("AWS_BEARER_TOKEN_BEDROCK","")
print(BEDROCK_API_KEY)
# Propagate to the env var boto3 reads natively, so the bearer flow is
# transparent regardless of which name the user set.
if BEDROCK_API_KEY and not os.environ.get("AWS_BEARER_TOKEN_BEDROCK"):
    os.environ["AWS_BEARER_TOKEN_BEDROCK"] = BEDROCK_API_KEY
# Default Bedrock model id (Anthropic Claude 3.5 Sonnet on-demand).
# Override with --model on the CLI.
DEFAULT_MODEL_BEDROCK = os.environ.get(
    "BEDROCK_MODEL_ID",
    "mistral.devstral-2-123b",
)

# --- Active provider ---------------------------------------------------------
# Set by runner.py based on --provider CLI flag. One of:
#   'local'   → Ollama at http://localhost:11434
#   'cloud'   → Ollama Cloud at https://ollama.com
#   'bedrock' → AWS Bedrock
PROVIDER: str = "bedrock"


def is_cloud() -> bool:
    return PROVIDER == "cloud" or "ollama.com" in OLLAMA_HOST


def is_bedrock() -> bool:
    return PROVIDER == "bedrock"


def default_model() -> str:
    if is_bedrock():
        return DEFAULT_MODEL_BEDROCK
    return DEFAULT_MODEL_CLOUD if is_cloud() else DEFAULT_MODEL_LOCAL
