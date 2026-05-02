"""Real retrieval-augmented generation (RAG) index for P5.

Index source : annotated training split  (data/processed/train.json)
Query source : test record at runtime
Encoder      : sentence-transformers/all-MiniLM-L6-v2 (384-d, deterministic)
Similarity   : cosine (vectors are L2-normalised, so dot-product == cosine)
Cache        : cache/embeddings/<lang>.npz   (vectors + sample_ids)

Methodology notes for the paper
-------------------------------
* Index is built from the **training split only**. The query record never
  appears in the candidate pool (defensive `sample_id` self-exclusion is
  also applied — splits are disjoint by construction, this is belt-and-
  braces).
* Encoder is a frozen public checkpoint; no fine-tuning, no leakage of
  test labels through the embedder.
* Retrieval is per-language: we never cross-pollinate exemplars across
  programming languages (different surface syntax → different similarity
  geometry).
* k is set by the caller (default 2), matching the few-shot exemplar
  count for fair ablation against P2.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Optional

import numpy as np

from . import config

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384
CACHE_DIR = config.ROOT / "cache" / "embeddings"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Lazy singletons (loading the model is the expensive part)
# ---------------------------------------------------------------------------

_model = None
_model_lock = threading.Lock()
_index_cache: dict[str, dict] = {}  # language -> {"vecs": np.ndarray, "records": list[dict]}


def _get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from sentence_transformers import SentenceTransformer
                _model = SentenceTransformer(EMBED_MODEL_NAME)
    return _model


def _truncate_for_encoder(text: str, max_chars: int = 4000) -> str:
    """MiniLM caps at 256 tokens; ~4k chars is a safe upper bound that
    captures class signatures, fields, and the start of method bodies —
    where smell signals are most concentrated."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def _embed_texts(texts: list[str]) -> np.ndarray:
    model = _get_model()
    vecs = model.encode(
        [_truncate_for_encoder(t) for t in texts],
        batch_size=8,
        convert_to_numpy=True,
        normalize_embeddings=True,   # L2-normalise → dot product == cosine
        show_progress_bar=False,
    )
    return vecs.astype(np.float32)


# ---------------------------------------------------------------------------
# Index build / load
# ---------------------------------------------------------------------------

def _cache_path(language: str) -> Path:
    return CACHE_DIR / f"rag_train_{language}.npz"


def build_index(language: str, *, force: bool = False) -> dict:
    """Build (or load cached) the RAG index for one language.

    Returns dict with keys:
        vecs    : np.ndarray  shape (N, EMBED_DIM), L2-normalised
        records : list[dict]  the training records, aligned with vecs rows
    """
    if not force and language in _index_cache:
        return _index_cache[language]

    train = json.loads(config.TRAIN_FILE_ANN.read_text(encoding="utf-8"))
    pool = [r for r in train if r["language"] == language and r.get("annotations")]

    cache = _cache_path(language)
    if cache.exists() and not force:
        npz = np.load(cache, allow_pickle=False)
        cached_ids = list(npz["sample_ids"])
        live_ids = [r["sample_id"] for r in pool]
        if cached_ids == live_ids:
            entry = {"vecs": npz["vecs"], "records": pool}
            _index_cache[language] = entry
            return entry
        # mismatch → rebuild

    if not pool:
        entry = {"vecs": np.zeros((0, EMBED_DIM), dtype=np.float32), "records": []}
        _index_cache[language] = entry
        return entry

    vecs = _embed_texts([r["source_code"] for r in pool])
    np.savez(
        cache,
        vecs=vecs,
        sample_ids=np.array([r["sample_id"] for r in pool], dtype="U128"),
    )
    entry = {"vecs": vecs, "records": pool}
    _index_cache[language] = entry
    return entry


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve(language: str, query_record: dict, k: int = 2) -> list[dict]:
    """Return the top-k training records most similar to `query_record`,
    excluding the query itself by `sample_id`.

    Each returned record is augmented with a `_rag_score` (cosine sim).
    """
    idx = build_index(language)
    if idx["vecs"].shape[0] == 0:
        return []

    q_vec = _embed_texts([query_record["source_code"]])  # (1, D)
    sims = (idx["vecs"] @ q_vec[0])                       # (N,)

    query_id = query_record.get("sample_id")
    order = np.argsort(-sims)
    out: list[dict] = []
    for i in order:
        rec = idx["records"][int(i)]
        if rec["sample_id"] == query_id:
            continue
        rec_with_score = dict(rec)
        rec_with_score["_rag_score"] = float(sims[int(i)])
        out.append(rec_with_score)
        if len(out) >= k:
            break
    return out


# ---------------------------------------------------------------------------
# CLI: pre-build all language indices  (`python -m srccode.build_rag_index`)
# ---------------------------------------------------------------------------

def build_all(verbose: bool = True) -> None:
    train = json.loads(config.TRAIN_FILE_ANN.read_text(encoding="utf-8"))
    languages = sorted({r["language"] for r in train})
    for lang in languages:
        idx = build_index(lang, force=True)
        if verbose:
            print(f"[rag-index] {lang:12s}  n={idx['vecs'].shape[0]:4d}  "
                  f"dim={idx['vecs'].shape[1] if idx['vecs'].size else 0}  "
                  f"cache={_cache_path(lang).name}")


def get_encoder_metadata() -> dict:
    """Reportable provenance for the paper's methodology section."""
    return {
        "encoder": EMBED_MODEL_NAME,
        "dim": EMBED_DIM,
        "similarity": "cosine (L2-normalised dot product)",
        "truncation_chars": 4000,
    }
