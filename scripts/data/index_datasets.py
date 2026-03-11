#!/usr/bin/env python3
"""
Vector Store Indexing Script

Loads preprocessed training data and indexes it into ChromaDB for RAG retrieval.
Only the training set (60%) is indexed — validation and test sets are held out.

Usage:
    python scripts/index_datasets.py
    python scripts/index_datasets.py --batch-size 16 --collection code_smell_examples
    python scripts/index_datasets.py --clear   # Clear existing index first
    python scripts/index_datasets.py --stats    # Show index statistics only

Architecture: Phase 3.2 (Vector Store Indexing)
"""

import argparse
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

from tqdm import tqdm

# Add project root to Python path (adjusted for scripts subdirectory)
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import CHROMADB_DIR, CHROMADB_COLLECTION_NAME, DATA_DIR
from src.data.data_loader import CodeSample
from src.data.data_preprocessor import DataPreprocessor
from src.rag.vector_store import VectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

PROCESSED_DIR = DATA_DIR / "processed"


def prepare_documents(
    samples: List[CodeSample],
) -> Tuple[List[str], List[Dict], List[str]]:
    """Convert CodeSamples to texts, metadata, and IDs for ChromaDB.

    Each sample becomes one document. Metadata stores smell types,
    language, dataset source, and annotation details for filtered retrieval.

    Args:
        samples: Preprocessed training samples.

    Returns:
        (texts, metadatas, ids) ready for VectorStore.add_documents().
    """
    texts: List[str] = []
    metadatas: List[Dict] = []
    ids: List[str] = []

    for sample in samples:
        if not sample.source_code.strip():
            continue

        smell_types = sample.smell_types
        # ChromaDB metadata values must be str, int, float, or bool
        meta = {
            "sample_id": sample.sample_id,
            "language": sample.language,
            "dataset": sample.dataset,
            "class_name": sample.class_name,
            "has_smells": sample.has_smells,
            "num_smells": len(sample.annotations),
            "smell_types": ", ".join(smell_types) if smell_types else "none",
        }

        texts.append(sample.source_code)
        metadatas.append(meta)
        ids.append(f"train_{sample.sample_id}")

        # Also index individual annotations as separate docs for per-smell retrieval
        for idx, ann in enumerate(sample.annotations):
            ann_text = (
                f"Code smell: {ann.smell_type} ({ann.category})\n"
                f"Method: {ann.method}\n"
                f"Description: {ann.description}\n\n"
                f"Source code:\n{sample.source_code}"
            )
            ann_meta = {
                "sample_id": sample.sample_id,
                "language": sample.language,
                "dataset": sample.dataset,
                "smell_type": ann.smell_type,
                "category": ann.category,
                "method": ann.method,
                "is_annotation": True,
            }
            texts.append(ann_text)
            metadatas.append(ann_meta)
            ids.append(f"train_{sample.sample_id}_ann_{idx}")

    return texts, metadatas, ids


def index_into_chromadb(
    texts: List[str],
    metadatas: List[Dict],
    ids: List[str],
    collection_name: str,
    batch_size: int,
    clear_existing: bool,
) -> Dict:
    """Index documents into ChromaDB in batches.

    Args:
        texts: Document texts.
        metadatas: Document metadata.
        ids: Document IDs.
        collection_name: ChromaDB collection name.
        batch_size: Number of docs per batch.
        clear_existing: Whether to clear the collection first.

    Returns:
        Statistics dict with counts and timings.
    """
    store = VectorStore(
        persist_dir=CHROMADB_DIR,
        collection_name=collection_name,
    )

    if clear_existing:
        logger.info("Clearing existing collection...")
        store.clear_collection()

    total = len(texts)
    indexed = 0
    failed = 0
    start_time = time.time()

    for i in tqdm(range(0, total, batch_size), desc="Indexing batches"):
        batch_texts = texts[i : i + batch_size]
        batch_meta = metadatas[i : i + batch_size]
        batch_ids = ids[i : i + batch_size]

        count = store.add_documents(
            texts=batch_texts,
            metadatas=batch_meta,
            ids=batch_ids,
        )
        if count > 0:
            indexed += count
        else:
            failed += len(batch_texts)

    elapsed = time.time() - start_time
    stats = store.get_stats()

    return {
        "total_prepared": total,
        "indexed": indexed,
        "failed": failed,
        "elapsed_seconds": round(elapsed, 2),
        "docs_per_second": round(indexed / elapsed, 1) if elapsed > 0 else 0,
        "collection_stats": stats,
    }


def show_stats(collection_name: str) -> None:
    """Print current collection statistics."""
    store = VectorStore(
        persist_dir=CHROMADB_DIR,
        collection_name=collection_name,
    )
    stats = store.get_stats()
    print(f"\nCollection: {stats.get('collection', collection_name)}")
    print(f"Documents:  {stats.get('document_count', 0)}")
    print(f"Persist:    {stats.get('persist_dir', CHROMADB_DIR)}")


def main():
    parser = argparse.ArgumentParser(
        description="Index preprocessed training data into ChromaDB for RAG",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Documents per indexing batch (default: 32)",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=CHROMADB_COLLECTION_NAME,
        help=f"ChromaDB collection name (default: {CHROMADB_COLLECTION_NAME})",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing collection before indexing",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show collection statistics and exit",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=PROCESSED_DIR,
        help=f"Directory with processed JSON splits (default: {PROCESSED_DIR})",
    )
    args = parser.parse_args()

    if args.stats:
        show_stats(args.collection)
        return 0

    # Load preprocessed training split
    logger.info(f"Loading training data from {args.input_dir}")
    split = DataPreprocessor.load_split(args.input_dir)

    if not split.train:
        logger.error("No training samples found. Run data preprocessing first.")
        return 1

    logger.info(f"Loaded {len(split.train)} training samples")

    # Show smell distribution
    smell_counts = Counter()
    for sample in split.train:
        for st in sample.smell_types:
            smell_counts[st] += 1
    if smell_counts:
        logger.info("Smell type distribution in training set:")
        for smell, count in smell_counts.most_common():
            logger.info(f"  {smell}: {count}")

    # Prepare documents
    logger.info("Preparing documents for indexing...")
    texts, metadatas, ids = prepare_documents(split.train)
    logger.info(
        f"Prepared {len(texts)} documents "
        f"({len(split.train)} samples + {len(texts) - len(split.train)} annotations)"
    )

    # Index into ChromaDB
    logger.info(f"Indexing into ChromaDB collection: {args.collection}")
    result = index_into_chromadb(
        texts=texts,
        metadatas=metadatas,
        ids=ids,
        collection_name=args.collection,
        batch_size=args.batch_size,
        clear_existing=args.clear,
    )

    # Save statistics
    stats_path = args.input_dir / "indexing_stats.json"
    stats_data = {
        "training_samples": len(split.train),
        "smell_distribution": dict(smell_counts),
        **result,
    }
    with open(stats_path, "w") as f:
        json.dump(stats_data, f, indent=2)

    # Print summary
    print("\n" + "=" * 50)
    print("INDEXING COMPLETE")
    print("=" * 50)
    print(f"Training samples:    {len(split.train)}")
    print(f"Documents prepared:  {result['total_prepared']}")
    print(f"Documents indexed:   {result['indexed']}")
    print(f"Failed:              {result['failed']}")
    print(f"Time:                {result['elapsed_seconds']}s")
    print(f"Speed:               {result['docs_per_second']} docs/s")
    print(f"Collection docs:     {result['collection_stats'].get('document_count', 'N/A')}")
    print(f"Stats saved:         {stats_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
