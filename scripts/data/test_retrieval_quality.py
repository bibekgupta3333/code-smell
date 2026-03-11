#!/usr/bin/env python3
"""
Retrieval Quality Testing Script

Tests RAG retrieval quality after ChromaDB indexing by running sample queries
for each code smell type and evaluating top-k accuracy.

Usage:
    python scripts/test_retrieval_quality.py
    python scripts/test_retrieval_quality.py --top-k 3 --threshold 0.5
    python scripts/test_retrieval_quality.py --use-mmr

Architecture: Phase 3.2 (Test retrieval quality)
"""

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

# Add project root to Python path (adjusted for scripts subdirectory)
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    CHROMADB_COLLECTION_NAME,
    CHROMADB_DIR,
    CODE_SMELL_TYPES,
    DATA_DIR,
    METRICS_DIR,
    RAG_CONFIG,
)
from src.data.data_loader import CodeSample
from src.data.data_preprocessor import DataPreprocessor
from src.rag.vector_store import VectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

PROCESSED_DIR = DATA_DIR / "processed"


def evaluate_retrieval(
    store: VectorStore,
    test_samples: List[CodeSample],
    top_k: int,
    threshold: float,
    use_mmr: bool,
) -> Dict:
    """Evaluate retrieval quality using validation/test samples.

    For each test sample that has annotations, query the vector store with
    the sample's source code and check whether retrieved results contain
    matching smell types (Precision@k / Hit@k).

    Args:
        store: Initialized VectorStore with indexed training data.
        test_samples: Held-out samples with ground truth annotations.
        top_k: Number of results to retrieve.
        threshold: Similarity threshold.
        use_mmr: Whether to use MMR reranking.

    Returns:
        Aggregate metrics dict.
    """
    annotated = [s for s in test_samples if s.has_smells]
    if not annotated:
        logger.warning("No annotated test samples found — skipping evaluation")
        return {"error": "no annotated samples"}

    per_smell: Dict[str, Dict] = defaultdict(lambda: {"hits": 0, "total": 0, "latencies": []})
    overall_hits = 0
    overall_total = 0
    latencies: List[float] = []

    for sample in tqdm(annotated, desc="Evaluating retrieval"):
        query = sample.source_code
        expected_smells = set(sample.smell_types)

        start = time.perf_counter()
        if use_mmr:
            results = store.search_with_mmr(
                query=query, top_k=top_k, threshold=threshold,
            )
        else:
            results = store.search(
                query=query, top_k=top_k, threshold=threshold,
            )
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.append(elapsed_ms)

        # Collect smell types from retrieved results
        retrieved_smells = set()
        for r in results:
            meta = r.get("metadata", {})
            # Single smell type (annotation docs)
            if meta.get("smell_type"):
                retrieved_smells.add(meta["smell_type"])
            # Comma-separated (full sample docs)
            if meta.get("smell_types") and meta["smell_types"] != "none":
                for s in meta["smell_types"].split(", "):
                    retrieved_smells.add(s.strip())

        # Hit if any expected smell is in retrieved
        hit = bool(expected_smells & retrieved_smells)
        if hit:
            overall_hits += 1
        overall_total += 1

        # Per-smell tracking
        for smell in expected_smells:
            per_smell[smell]["total"] += 1
            if smell in retrieved_smells:
                per_smell[smell]["hits"] += 1
            per_smell[smell]["latencies"].append(elapsed_ms)

    # Aggregate
    hit_rate = overall_hits / overall_total if overall_total else 0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0

    per_smell_summary = {}
    for smell, data in sorted(per_smell.items()):
        rate = data["hits"] / data["total"] if data["total"] else 0
        avg_lat = sum(data["latencies"]) / len(data["latencies"]) if data["latencies"] else 0
        per_smell_summary[smell] = {
            "hit_rate": round(rate, 4),
            "hits": data["hits"],
            "total": data["total"],
            "avg_latency_ms": round(avg_lat, 2),
        }

    return {
        "top_k": top_k,
        "threshold": threshold,
        "use_mmr": use_mmr,
        "annotated_samples": len(annotated),
        "overall_hit_rate": round(hit_rate, 4),
        "overall_hits": overall_hits,
        "overall_total": overall_total,
        "avg_latency_ms": round(avg_latency, 2),
        "min_latency_ms": round(min(latencies), 2) if latencies else 0,
        "max_latency_ms": round(max(latencies), 2) if latencies else 0,
        "per_smell": per_smell_summary,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Test RAG retrieval quality after ChromaDB indexing",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=RAG_CONFIG["top_k"],
        help=f"Number of results to retrieve (default: {RAG_CONFIG['top_k']})",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=RAG_CONFIG["similarity_threshold"],
        help=f"Similarity threshold (default: {RAG_CONFIG['similarity_threshold']})",
    )
    parser.add_argument(
        "--use-mmr",
        action="store_true",
        help="Use Max Marginal Relevance reranking",
    )
    parser.add_argument(
        "--split",
        choices=["validation", "test"],
        default="validation",
        help="Which split to evaluate on (default: validation)",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=CHROMADB_COLLECTION_NAME,
        help=f"ChromaDB collection (default: {CHROMADB_COLLECTION_NAME})",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=PROCESSED_DIR,
        help="Directory with processed JSON splits",
    )
    args = parser.parse_args()

    # Load split
    logger.info(f"Loading {args.split} split from {args.input_dir}")
    split = DataPreprocessor.load_split(args.input_dir)
    samples = split.validation if args.split == "validation" else split.test

    if not samples:
        logger.error(f"No {args.split} samples found.")
        return 1

    logger.info(f"Loaded {len(samples)} {args.split} samples")

    # Initialize vector store
    store = VectorStore(
        persist_dir=CHROMADB_DIR,
        collection_name=args.collection,
    )
    stats = store.get_stats()
    doc_count = stats.get("document_count", 0)
    if doc_count == 0:
        logger.error("Vector store is empty. Run index_datasets.py first.")
        return 1
    logger.info(f"Vector store has {doc_count} documents")

    # Evaluate
    result = evaluate_retrieval(
        store=store,
        test_samples=samples,
        top_k=args.top_k,
        threshold=args.threshold,
        use_mmr=args.use_mmr,
    )

    # Save results
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = METRICS_DIR / "rag_retrieval_quality.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    # Print summary
    print("\n" + "=" * 55)
    print("RETRIEVAL QUALITY EVALUATION")
    print("=" * 55)
    print(f"Split:              {args.split}")
    print(f"Samples evaluated:  {result['annotated_samples']}")
    print(f"Top-k:              {result['top_k']}")
    print(f"Threshold:          {result['threshold']}")
    print(f"MMR:                {result['use_mmr']}")
    print(f"Overall Hit Rate:   {result['overall_hit_rate']:.2%}")
    print(f"Avg Latency:        {result['avg_latency_ms']:.1f}ms")

    if result.get("per_smell"):
        print(f"\nPer-Smell Hit Rate:")
        for smell, data in sorted(
            result["per_smell"].items(),
            key=lambda x: x[1]["hit_rate"],
            reverse=True,
        ):
            print(f"  {smell:35s} {data['hit_rate']:.2%}  ({data['hits']}/{data['total']})")

    print(f"\nResults saved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
