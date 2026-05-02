"""Load and render prompt templates with placeholder substitution.

Templates live at:
    prompts/<lang>/<prompt_name>.md   — has <!-- SYSTEM --> and <!-- USER --> markers
    prompts/_system.md                — global system role
    prompts/_taxonomy.md              — 23-smell taxonomy
    prompts/_output_schema.md         — JSON output contract

Returned by `render(...)`:
    (system_text, user_text)
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional

from . import config


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def _split_roles(template: str) -> tuple[str, str]:
    """Split a prompt template on the SYSTEM / USER delimiters."""
    if "<!-- SYSTEM -->" not in template or "<!-- USER -->" not in template:
        raise ValueError("Template missing <!-- SYSTEM --> / <!-- USER --> markers")
    after_sys = template.split("<!-- SYSTEM -->", 1)[1]
    sys_text, user_text = after_sys.split("<!-- USER -->", 1)
    return sys_text.strip(), user_text.strip()


def _number_lines(source: str) -> str:
    return "\n".join(f"{i:4d}| {line}" for i, line in enumerate(source.splitlines(), 1))


def render(
    language: str,
    prompt_name: str,
    record: dict,
    *,
    few_shot_examples: Optional[str] = None,
    retrieved_snippets: Optional[str] = None,
    rag_mode: str = "dense",
    rag_k: int = 2,
) -> tuple[str, str]:
    """Render a prompt for one record.

    Args:
        language: 'java' | 'python' | 'javascript' | 'cpp'
        prompt_name: 'p1_zero_shot' | 'p2_few_shot' | 'p3_taxonomy_tree' |
                     'p4_self_verify' | 'p5_rag'
        record: a single entry from prepared_data/datasets/unannotated/test.json

    Returns:
        (system_text, user_text)
    """
    template = _read(config.PROMPTS_ROOT / language / f"{prompt_name}.md")
    system_block = _read(config.SYSTEM_FILE)
    taxonomy     = _read(config.TAXONOMY_FILE)
    schema       = _read(config.SCHEMA_FILE)

    template = (template
                .replace("{SYSTEM_BLOCK}", system_block)
                .replace("{TAXONOMY}", taxonomy)
                .replace("{OUTPUT_SCHEMA}", schema)
                .replace("{FILE_PATH}", record["file_path"])
                .replace("{CLASS_NAME}", record["class_name"])
                .replace("{SOURCE_CODE}", _number_lines(record["source_code"])))

    if "{FEW_SHOT_EXAMPLES}" in template:
        template = template.replace(
            "{FEW_SHOT_EXAMPLES}", few_shot_examples or build_few_shot(language)
        )
    if "{RETRIEVED_SNIPPETS}" in template:
        template = template.replace(
            "{RETRIEVED_SNIPPETS}",
            retrieved_snippets
            or build_rag_context(language, record, k=rag_k, mode=rag_mode),
        )

    return _split_roles(template)


# ---------------------------------------------------------------------------
# Few-shot example builder (P2). Picks 2 positive + 1 negative from train set.
# ---------------------------------------------------------------------------

def build_few_shot(language: str, n_pos: int = 2) -> str:
    """Build few-shot examples from the annotated training set."""
    train = json.loads(_read(config.TRAIN_FILE_ANN))
    pos = [r for r in train if r["language"] == language and r["annotations"]]
    neg = [r for r in train if r["language"] == language and not r["annotations"]]

    rng = random.Random(42)
    chosen: list[dict] = []
    if pos:
        chosen.extend(rng.sample(pos, min(n_pos, len(pos))))
    if neg:
        chosen.append(rng.choice(neg))
    if not chosen and pos:
        chosen = pos[:n_pos]

    blocks: list[str] = []
    for i, r in enumerate(chosen, 1):
        findings = [
            {
                "smell_type": a["smell_type"],
                "category":   a["category"],
                "method":     a.get("method") or "Entire Class",
                "line_start": a.get("line_start"),
                "line_end":   a.get("line_end"),
                "evidence":   a.get("evidence", ""),
            }
            for a in r["annotations"][:8]  # cap per example
        ]
        out = {
            "language":   r["language"],
            "file_path":  r["file_path"],
            "class_name": r["class_name"],
            "findings":   findings,
        }
        blocks.append(
            f"### Example {i} — file_path = `{r['file_path']}`\n"
            f"```{r['language']}\n{_number_lines(r['source_code'])}\n```\n\n"
            f"Expected output:\n```json\n{json.dumps(out, indent=2)}\n```\n"
        )
    return "\n".join(blocks)


# ---------------------------------------------------------------------------
# RAG context builder (P5). Real dense retriever:
#   * Index   = annotated training split (built once, cached on disk).
#   * Query   = the test record's source code.
#   * Encoder = sentence-transformers/all-MiniLM-L6-v2 (frozen, public).
#   * Score   = cosine similarity (L2-normalised dot product).
#   * Filter  = same language, exclude query's own sample_id.
# See `srccode/common/rag_index.py` for the indexing/retrieval implementation.
# A `--rag-mode random` fallback is exposed for the ablation baseline.
# ---------------------------------------------------------------------------

def build_rag_context(
    language: str,
    target: dict,
    k: int = 2,
    *,
    mode: str = "dense",
) -> str:
    """Render the retrieved-exemplar block for prompt P5.

    mode:
      'dense'  — sentence-transformers similarity over training corpus (default)
      'random' — uniform random sample from training corpus (ablation lower bound)
    """
    chosen: list[dict] = []
    if mode == "dense":
        from . import rag_index
        chosen = rag_index.retrieve(language, target, k=k)

    if not chosen:  # mode == 'random' or dense returned empty (e.g. cold cache fallback)
        train = json.loads(_read(config.TRAIN_FILE_ANN))
        pool = [
            r for r in train
            if r["language"] == language
            and r["sample_id"] != target.get("sample_id")
            and r["annotations"]
        ]
        rng = random.Random(hash(target.get("sample_id", "x")) & 0xFFFFFFFF)
        chosen = rng.sample(pool, min(k, len(pool))) if pool else []

    blocks: list[str] = []
    for i, r in enumerate(chosen, 1):
        findings = [
            {
                "smell_type": a["smell_type"],
                "category":   a["category"],
                "method":     a.get("method") or "Entire Class",
                "line_start": a.get("line_start"),
                "line_end":   a.get("line_end"),
                "evidence":   a.get("evidence", ""),
            }
            for a in r["annotations"][:6]
        ]
        score = r.get("_rag_score")
        score_str = f" — cos={score:.3f}" if isinstance(score, float) else ""
        blocks.append(
            f"### Exemplar {i} — file_path={r['file_path']}{score_str}\n"
            f"<source>\n{_number_lines(r['source_code'])}\n</source>\n"
            f"<findings>\n{json.dumps(findings, indent=2)}\n</findings>\n"
        )
    return "\n".join(blocks) if blocks else "(no exemplars retrieved)"
