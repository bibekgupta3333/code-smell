"""Pre-build the P5 RAG index for every language found in the training split.

Run once after data preparation, or whenever train.json changes:

    python -m srccode.build_rag_index
"""
from __future__ import annotations

from .common import rag_index


def main() -> None:
    meta = rag_index.get_encoder_metadata()
    print(f"[rag-index] encoder={meta['encoder']}  dim={meta['dim']}  "
          f"sim={meta['similarity']}")
    rag_index.build_all(verbose=True)
    print("[rag-index] done")


if __name__ == "__main__":
    main()
