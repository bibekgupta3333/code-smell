"""Dataset loaders and ground-truth helpers."""
from __future__ import annotations

import json
from pathlib import Path

from . import config


_VALID_DATASETS = ("annotated", "unannotated")
_VALID_SPLITS   = ("all", "train", "val", "test", "java", "python", "javascript", "cpp")


def load_test(
    language: str | None = None,
    dataset: str = "unannotated",
    split: str = "test",
) -> list[dict]:
    """Load records for evaluation.

    Args:
        language: optional filter ('java' | 'python' | 'javascript' | 'cpp').
                  If `split` is itself a language name, this is ignored.
        dataset:  'unannotated' (class-level GT) or 'annotated' (line-level GT).
        split:    'test' (default, 8 records), 'train' (20), 'val' (6),
                  'all' (34), or a language name to load that language's full
                  set ('java' = 7 records, 'cpp' = 13, etc.).
    """
    if dataset not in _VALID_DATASETS:
        raise ValueError(f"Unknown dataset: {dataset!r}")
    if split not in _VALID_SPLITS:
        raise ValueError(f"Unknown split: {split!r}")
    path = config.PREPARED / dataset / f"{split}.json"
    records = json.loads(path.read_text(encoding="utf-8"))
    if language:
        records = [r for r in records if r["language"] == language]
    return records


def ground_truth_keys(record: dict) -> set[tuple[str, str, str, str]]:
    """Convert a record's ground truth into hashable keys for grading.

    Key = (file_path, class_name, method, smell_type).
    Method-level granularity (line numbers not used in matching).
    Reads `ground_truth` (unannotated) or `annotations` (annotated).
    """
    items = record.get("ground_truth") or record.get("annotations") or []
    keys: set[tuple[str, str, str, str]] = set()
    for g in items:
        keys.add((
            record["file_path"],
            record["class_name"],
            (g.get("method") or "Entire Class").strip(),
            g["smell_type"].strip(),
        ))
    return keys
