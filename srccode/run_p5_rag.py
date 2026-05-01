"""P5 — Retrieval-Augmented Detection (real dense RAG).

Index   : training split (annotated), per language.
Encoder : sentence-transformers/all-MiniLM-L6-v2.
Score   : cosine similarity (L2-normalised dot product).
Query   : test record source code (truncated to ~4k chars for the encoder).

Use `--rag-mode random` for the null-retriever ablation baseline.
Use `--rag-k N` to vary the number of retrieved exemplars (default 2).

The index is built lazily on first call and cached at
`cache/embeddings/rag_train_<lang>.npz`. To pre-build all caches run:

    python -m srccode.build_rag_index
"""
from srccode.common.runner import run

if __name__ == "__main__":
    raise SystemExit(run("p5_rag"))
