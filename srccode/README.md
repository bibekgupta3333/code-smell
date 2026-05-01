# `srccode/` — LLM Code-Smell Detection Runners

Five thin scripts (one per prompt strategy) that read
`prepared_data/datasets/unannotated/test.json`, send each file to an LLM via
**Ollama** (local desktop **or** Ollama Cloud), parse the JSON response,
and grade it against the ground truth.

```
srccode/
├── run_p1_zero_shot.py
├── run_p2_few_shot.py
├── run_p3_taxonomy_tree.py
├── run_p4_self_verify.py
├── run_p5_rag.py
└── common/
    ├── config.py          # paths + provider env vars
    ├── ollama_client.py   # one chat() function for local + cloud
    ├── prompt_loader.py   # renders prompts/<lang>/<prompt>.md per record
    ├── dataset.py         # loads test.json + ground-truth keys
    ├── evaluator.py       # JSON extraction + P/R/F1 (overall + per-lang + per-smell)
    └── runner.py          # the shared run loop used by all 5 scripts
```

## Setup

```bash
pip install ollama
```

That's it — the rest is stdlib + the `ollama` SDK.

## Choosing the provider

| Mode | What to set |
|---|---|
| **Local Ollama desktop** (default) | nothing — runs against `http://localhost:11434`. Make sure the model is pulled: `ollama pull qwen2.5-coder:7b` |
| **Ollama Cloud** | `export OLLAMA_HOST=https://ollama.com` and `export OLLAMA_API_KEY=<your-key>` |

The same scripts work with both — only the env vars differ.

## Running

```bash
# Smoke test: 2 Java files, local model
python -m srccode.run_p1_zero_shot --language java --limit 2

# Full P3 run on all 4 languages, local
python -m srccode.run_p3_taxonomy_tree --model qwen2.5-coder:7b

# Full P4 run on cloud
export OLLAMA_HOST=https://ollama.com
export OLLAMA_API_KEY=...
python -m srccode.run_p4_self_verify --model gpt-oss:120b
```

### Common flags (all 5 scripts)

| Flag | Default | Purpose |
|---|---|---|
| `--model` | `qwen2.5-coder:7b` (local) / `gpt-oss:120b` (cloud) | Ollama model tag |
| `--language` | all four | Restrict to `java` / `python` / `javascript` / `cpp` |
| `--limit N` | none | Only process the first N records (debug) |
| `--temperature` | `0.0` | Determinism |
| `--seed` | `42` | Determinism |
| `--output-dir` | `results/llm_runs/` | Where to write metrics |

## Output

Each run writes two files under
`results/llm_runs/<prompt>/<prompt>__<model>__<lang>__<timestamp>.{metrics,predictions}.json`:

- **`*.metrics.json`** — overall, per-language and per-smell precision /
  recall / F1, plus run metadata (model, host, seed, elapsed seconds).
- **`*.predictions.json`** — for each record: parsed findings, gold keys,
  predicted keys, parse_error flag, and the raw response (for error
  analysis).

Console output is a per-record `tp / fp / fn` line plus a final summary.

## Grading rule (matches `prepared_data` exactly)

A *match key* is the 4-tuple

```
(file_path, class_name, method, smell_type)
```

with the canonical 23-smell vocabulary. `method = "Entire Class"` for
class-level smells. The grader uses set equality on these keys, so
- True positives = `gold ∩ pred`
- False positives = `pred − gold`
- False negatives = `gold − pred`

This is method-level granularity (line numbers are not used in matching
because the unannotated ground truth doesn't have them). If you switch the
input to `prepared_data/datasets/annotated/test.json` later, you can extend
the keys with `(line_start, line_end)` for line-level grading.

## How the prompts are assembled

For each record, `prompt_loader.render()`:

1. Reads `prompts/<lang>/<prompt>.md` (the per-language template).
2. Substitutes `{SYSTEM_BLOCK}` ← `prompts/_system.md`,
   `{TAXONOMY}` ← `prompts/_taxonomy.md`,
   `{OUTPUT_SCHEMA}` ← `prompts/_output_schema.md`.
3. Substitutes `{FILE_PATH}`, `{CLASS_NAME}`, `{SOURCE_CODE}` (with
   line-number prefixes `   N| `).
4. For P2: builds `{FEW_SHOT_EXAMPLES}` from the annotated train split
   (2 positive + 1 negative, deterministic via `seed=42`).
5. For P5: builds `{RETRIEVED_SNIPPETS}` from the same train split (2
   random language-matched exemplars; replace with a real embedding
   retriever if/when needed).
6. Splits the result on `<!-- SYSTEM -->` / `<!-- USER -->` markers and
   sends them as `role=system` and `role=user` messages.

## Reproducing one cell of the experimental matrix

```bash
# P3 × qwen2.5-coder:7b × Java
python -m srccode.run_p3_taxonomy_tree --model qwen2.5-coder:7b --language java
```

To sweep all 60 cells, write a tiny shell loop over
`{p1,p2,p3,p4,p5} × {model_low, model_med, model_high} × {java,python,javascript,cpp}`
and call the matching script.
