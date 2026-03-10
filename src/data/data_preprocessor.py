"""
Data Preprocessor for Code Smell Detection System.
Cleans code samples, extracts metrics, validates labels,
and creates train/validation/test splits.

Architecture: Implements Benchmarking Section 1 (60/20/20 split)
Focus: Production code smells only (Gap #12)
"""

import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from sklearn.model_selection import train_test_split

from config import (
    DATA_DIR,
    DATA_SPLIT,
    RANDOM_SEED,
    CODE_SMELL_TYPES,
    SUPPORTED_LANGUAGES,
)
from src.data.data_loader import CodeSample, SmellAnnotation

logger = logging.getLogger(__name__)

PROCESSED_DIR = DATA_DIR / "processed"


@dataclass
class DatasetSplit:
    """Container for train/validation/test split."""
    train: List[CodeSample]
    validation: List[CodeSample]
    test: List[CodeSample]

    @property
    def sizes(self) -> Dict[str, int]:
        return {"train": len(self.train), "validation": len(self.validation), "test": len(self.test)}

    def summary(self) -> str:
        total = sum(self.sizes.values())
        parts = [f"{k}: {v} ({v/total*100:.1f}%)" for k, v in self.sizes.items()] if total else []
        return f"Total: {total} | " + " | ".join(parts)


class DataPreprocessor:
    """
    Preprocesses code samples for the code smell detection pipeline.

    Features:
    - Code cleaning (normalize whitespace, optionally strip comments)
    - Label validation against known code smell types
    - Metric extraction (LOC, method count)
    - Stratified train/validation/test split (60/20/20)
    - Production code filtering (Gap #12)

    Example:
        >>> from src.data_loader import DatasetLoader
        >>> loader = DatasetLoader()
        >>> samples = loader.load_all()
        >>> preprocessor = DataPreprocessor()
        >>> split = preprocessor.preprocess_and_split(samples)
        >>> print(split.summary())
    """

    def __init__(
        self,
        strip_comments: bool = False,
        normalize_whitespace: bool = True,
        min_lines: int = 3,
        validate_labels: bool = True,
    ):
        self.strip_comments = strip_comments
        self.normalize_whitespace = normalize_whitespace
        self.min_lines = min_lines
        self.validate_labels = validate_labels
        self._known_smells = set(CODE_SMELL_TYPES)

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def preprocess_and_split(self, samples: List[CodeSample]) -> DatasetSplit:
        """Full preprocessing pipeline: clean, validate, filter, split."""
        logger.info(f"Starting preprocessing of {len(samples)} samples")

        # Step 1: Clean code
        samples = [self._clean_sample(s) for s in samples]

        # Step 2: Filter empty / too-short samples
        samples = [s for s in samples if self._is_valid_sample(s)]
        logger.info(f"After filtering: {len(samples)} valid samples")

        # Step 3: Validate labels
        if self.validate_labels:
            samples = self._validate_annotations(samples)

        # Step 4: Filter production code only
        samples = [s for s in samples if self._is_production_code(s)]
        logger.info(f"After production filter: {len(samples)} samples")

        # Step 5: Extract metrics
        for s in samples:
            s.metadata.update(self._extract_metrics(s))

        # Step 6: Split
        split = self._create_split(samples)
        logger.info(f"Split: {split.summary()}")
        return split

    # ------------------------------------------------------------------
    # Code cleaning
    # ------------------------------------------------------------------

    def _clean_sample(self, sample: CodeSample) -> CodeSample:
        """Clean a code sample's source code."""
        code = sample.source_code
        if self.strip_comments:
            code = self._remove_comments(code, sample.language)
        if self.normalize_whitespace:
            # Normalize line endings and trailing whitespace
            lines = [line.rstrip() for line in code.splitlines()]
            # Remove excessive blank lines (max 2 consecutive)
            cleaned = []
            blank_count = 0
            for line in lines:
                if not line.strip():
                    blank_count += 1
                    if blank_count <= 2:
                        cleaned.append(line)
                else:
                    blank_count = 0
                    cleaned.append(line)
            code = "\n".join(cleaned).strip() + "\n"
        sample.source_code = code
        return sample

    @staticmethod
    def _remove_comments(code: str, language: str) -> str:
        """Remove comments from code (best-effort, regex-based)."""
        if language in ("python",):
            # Remove # comments (not inside strings - simplified)
            code = re.sub(r'#[^\n]*', '', code)
            # Remove docstrings
            code = re.sub(r'"""[\s\S]*?"""', '', code)
            code = re.sub(r"'''[\s\S]*?'''", '', code)
        elif language in ("java", "javascript", "cpp"):
            # Remove // and /* */ comments
            code = re.sub(r'//[^\n]*', '', code)
            code = re.sub(r'/\*[\s\S]*?\*/', '', code)
        return code

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _is_valid_sample(self, sample: CodeSample) -> bool:
        """Check if a sample meets minimum quality criteria."""
        if not sample.source_code.strip():
            return False
        loc = sample.source_code.count("\n") + 1
        return loc >= self.min_lines

    def _validate_annotations(self, samples: List[CodeSample]) -> List[CodeSample]:
        """Validate and normalize smell type labels."""
        unknown_types: Counter = Counter()
        for sample in samples:
            valid_anns = []
            for ann in sample.annotations:
                normalized = self._normalize_smell_type(ann.smell_type)
                if normalized:
                    ann.smell_type = normalized
                    valid_anns.append(ann)
                else:
                    unknown_types[ann.smell_type] += 1
            sample.annotations = valid_anns

        if unknown_types:
            logger.warning(f"Dropped {sum(unknown_types.values())} annotations "
                          f"with unknown types: {dict(unknown_types.most_common(10))}")
        return samples

    def _normalize_smell_type(self, smell_type: str) -> Optional[str]:
        """Normalize a smell type label to match CODE_SMELL_TYPES."""
        if not smell_type:
            return None
        # Exact match
        if smell_type in self._known_smells:
            return smell_type
        # Case-insensitive match
        lower_map = {s.lower(): s for s in self._known_smells}
        if smell_type.lower() in lower_map:
            return lower_map[smell_type.lower()]
        # Partial match (e.g., "God Class" matches "God Class")
        for known in self._known_smells:
            if smell_type.lower() in known.lower() or known.lower() in smell_type.lower():
                return known
        return None

    @staticmethod
    def _is_production_code(sample: CodeSample) -> bool:
        """Ensure sample is production code (not test code)."""
        indicators = ["test", "Test", "mock", "Mock", "spec", "Spec"]
        if any(ind in sample.class_name for ind in indicators):
            return False
        path = sample.file_path.lower()
        if any(p in path for p in ["/test/", "/tests/", "test_", "_test."]):
            return False
        return True

    # ------------------------------------------------------------------
    # Metric extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_metrics(sample: CodeSample) -> Dict:
        """Extract basic code metrics from a sample."""
        lines = sample.source_code.splitlines()
        non_blank = [l for l in lines if l.strip()]
        return {
            "loc": len(lines),
            "sloc": len(non_blank),
            "language": sample.language,
            "has_annotations": sample.has_smells,
            "num_smells": len(sample.annotations),
        }

    # ------------------------------------------------------------------
    # Train/Validation/Test split (60/20/20)
    # ------------------------------------------------------------------

    def _create_split(self, samples: List[CodeSample]) -> DatasetSplit:
        """
        Create stratified split: 60% train, 20% validation, 20% test.
        Stratified by has_smells to preserve label distribution.
        """
        if len(samples) < 5:
            logger.warning("Too few samples for splitting, putting all in train")
            return DatasetSplit(train=samples, validation=[], test=[])

        # Stratification label: has_smells flag
        labels = [1 if s.has_smells else 0 for s in samples]

        train_ratio = DATA_SPLIT["train"]
        val_ratio = DATA_SPLIT["validation"]
        test_ratio = DATA_SPLIT["test"]

        # First split: train vs (val+test)
        train, temp, train_labels, temp_labels = train_test_split(
            samples, labels,
            train_size=train_ratio,
            random_state=RANDOM_SEED,
            stratify=labels if min(Counter(labels).values()) >= 2 else None,
        )

        # Second split: val vs test (50/50 of remaining = 20/20)
        val_frac = val_ratio / (val_ratio + test_ratio)
        val, test, _, _ = train_test_split(
            temp, temp_labels,
            train_size=val_frac,
            random_state=RANDOM_SEED,
            stratify=temp_labels if min(Counter(temp_labels).values()) >= 2 else None,
        )

        return DatasetSplit(train=train, validation=val, test=test)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_split(self, split: DatasetSplit, output_dir: Optional[Path] = None):
        """Save preprocessed split to JSON files."""
        out = output_dir or PROCESSED_DIR
        out.mkdir(parents=True, exist_ok=True)

        for name, samples in [("train", split.train), ("validation", split.validation), ("test", split.test)]:
            path = out / f"{name}.json"
            data = [s.to_dict() for s in samples]
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(samples)} samples to {path}")

        # Save split metadata
        meta = {
            "split_ratios": dict(DATA_SPLIT),
            "random_seed": RANDOM_SEED,
            "sizes": split.sizes,
            "total": sum(split.sizes.values()),
        }
        with open(out / "split_metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

    @staticmethod
    def load_split(input_dir: Optional[Path] = None) -> DatasetSplit:
        """Load a previously saved split from JSON files."""
        src = input_dir or PROCESSED_DIR
        splits = {}
        for name in ("train", "validation", "test"):
            path = src / f"{name}.json"
            if not path.exists():
                splits[name] = []
                continue
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            splits[name] = [
                CodeSample(
                    sample_id=d["sample_id"],
                    source_code=d["source_code"],
                    language=d["language"],
                    class_name=d.get("class_name", ""),
                    file_path=d.get("file_path", ""),
                    dataset=d.get("dataset", ""),
                    annotations=[SmellAnnotation(**a) for a in d.get("annotations", [])],
                    metadata=d.get("metadata", {}),
                )
                for d in data
            ]
        return DatasetSplit(**splits)


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    from src.data_loader import DatasetLoader

    loader = DatasetLoader()
    samples = loader.load_all()

    preprocessor = DataPreprocessor()
    split = preprocessor.preprocess_and_split(samples)
    print(f"\n{split.summary()}")

    # Show smell type distribution
    all_smells = []
    for s in split.train + split.validation + split.test:
        all_smells.extend(s.smell_types)
    print(f"\nSmell type distribution:")
    for smell, count in Counter(all_smells).most_common():
        print(f"  {smell}: {count}")

    preprocessor.save_split(split)
    print(f"\nSaved to {PROCESSED_DIR}")
