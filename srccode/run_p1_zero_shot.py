"""P1 — Zero-shot code-smell detection.

Usage:
    # local Ollama desktop (default)
    python -m srccode.run_p1_zero_shot --model qwen2.5-coder:7b

    # Ollama Cloud
    export OLLAMA_HOST=https://ollama.com
    export OLLAMA_API_KEY=...
    python -m srccode.run_p1_zero_shot --model gpt-oss:120b

    # one language only / quick smoke test
    python -m srccode.run_p1_zero_shot --language java --limit 2
"""
from srccode.common.runner import run

if __name__ == "__main__":
    raise SystemExit(run("p1_zero_shot"))
