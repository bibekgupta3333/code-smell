"""
Dataset Loader for Code Smell Detection System.
Loads datasets from various formats (CSV, JSON, XML) into a unified data model.

Supports:
  - SmellyCodeDataset (bundled, CSV ground truth + source files)
  - MaRV dataset (JSON/CSV)
  - Qualitas Corpus (Java source files)
  - Generic CSV/JSON datasets
"""

import csv
import json
import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Optional, Iterator

from config import (
    DATASETS_DIR,
    MARV_DATASET_DIR,
    QUALITAS_CORPUS_DIR,
    SMELLY_CODE_DIR,
    GROUND_TRUTH_DIR,
    CODE_SMELL_TYPES,
    SUPPORTED_LANGUAGES,
)

logger = logging.getLogger(__name__)


# ============================================================================
# UNIFIED DATA MODELS
# ============================================================================

@dataclass
class SmellAnnotation:
    """A single code smell annotation."""
    smell_type: str
    category: str
    method: str = ""
    description: str = ""
    severity: str = ""
    confidence: float = 1.0

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class CodeSample:
    """Unified representation of a code sample with annotations."""
    sample_id: str
    source_code: str
    language: str
    class_name: str = ""
    file_path: str = ""
    dataset: str = ""
    annotations: List[SmellAnnotation] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    @property
    def has_smells(self) -> bool:
        return len(self.annotations) > 0

    @property
    def smell_types(self) -> List[str]:
        return [a.smell_type for a in self.annotations]

    def to_dict(self) -> Dict:
        d = asdict(self)
        return d


# ============================================================================
# DATASET LOADERS
# ============================================================================

class DatasetLoader:
    """
    Loads code smell datasets into a unified CodeSample format.

    Example:
        >>> loader = DatasetLoader()
        >>> samples = loader.load_smelly_code_dataset()
        >>> print(f"Loaded {len(samples)} samples")
    """

    def __init__(self):
        self._validate_paths()

    def _validate_paths(self):
        """Log warnings for missing dataset directories."""
        for name, path in [
            ("SmellyCodeDataset", DATASETS_DIR / "SmellyCodeDataset"),
            ("MaRV", MARV_DATASET_DIR),
            ("Qualitas Corpus", QUALITAS_CORPUS_DIR),
        ]:
            if not path.exists() or not any(path.iterdir()):
                logger.warning(f"{name} directory missing or empty: {path}")

    # ------------------------------------------------------------------
    # SmellyCodeDataset loader (bundled dataset with ground truth CSV)
    # ------------------------------------------------------------------

    def load_smelly_code_dataset(self) -> List[CodeSample]:
        """Load the SmellyCodeDataset with ground truth annotations."""
        base = DATASETS_DIR / "SmellyCodeDataset"
        gt_csv = base / "Analysis" / "GroundTruthLevel" / "GroundTruth.csv"
        if not gt_csv.exists():
            logger.error(f"Ground truth CSV not found: {gt_csv}")
            return []

        # Parse ground truth CSV
        annotations_by_key: Dict[str, List[SmellAnnotation]] = {}
        with open(gt_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                lang = row.get("Language", "").strip()
                cls = row.get("Class", "").strip()
                if not lang or not cls:
                    continue
                key = f"{lang}_{cls}"
                ann = SmellAnnotation(
                    smell_type=row.get("Code Smell", "").strip(),
                    category=row.get("Category", "").strip(),
                    method=row.get("Method", "").strip(),
                    description=row.get("Type", "").strip(),
                )
                annotations_by_key.setdefault(key, []).append(ann)

        # Load source files and match with annotations
        samples: List[CodeSample] = []
        for lang_dir in ["Python", "Java", "JavaScript", "C++"]:
            annotated_dir = base / lang_dir / "SmellyAnnotated"
            if not annotated_dir.exists():
                continue
            lang = lang_dir.lower()
            if lang == "c++":
                lang = "cpp"
            ext_map = {"python": ".py", "java": ".java", "javascript": ".js", "cpp": ".cpp"}
            ext = ext_map.get(lang, "")
            for src_file in annotated_dir.iterdir():
                if not src_file.is_file() or (ext and src_file.suffix != ext):
                    continue
                cls_name = src_file.stem
                key = f"{lang_dir}_{cls_name}"
                code = src_file.read_text(encoding="utf-8", errors="replace")
                sample = CodeSample(
                    sample_id=f"smelly_{lang}_{cls_name}",
                    source_code=code,
                    language=lang,
                    class_name=cls_name,
                    file_path=str(src_file.relative_to(base)),
                    dataset="SmellyCodeDataset",
                    annotations=annotations_by_key.get(key, []),
                )
                samples.append(sample)

        logger.info(f"Loaded {len(samples)} samples from SmellyCodeDataset "
                     f"({sum(s.has_smells for s in samples)} with annotations)")
        return samples

    # ------------------------------------------------------------------
    # MaRV dataset loader
    # ------------------------------------------------------------------

    def load_marv_dataset(self) -> List[CodeSample]:
        """
        Load the MaRV dataset.
        Expects JSON or CSV files in data/datasets/marv/raw/.
        Filters for production code smells only (excludes test smells).
        """
        raw_dir = MARV_DATASET_DIR / "raw"
        if not raw_dir.exists():
            logger.warning(f"MaRV raw directory not found: {raw_dir}. "
                           "Run download instructions in data/datasets/marv/README.md")
            return []

        samples: List[CodeSample] = []
        # Try JSON files first
        for json_file in raw_dir.glob("*.json"):
            samples.extend(self._load_json_samples(json_file, dataset="MaRV"))
        # Try CSV files
        for csv_file in raw_dir.glob("*.csv"):
            samples.extend(self._load_csv_samples(csv_file, dataset="MaRV"))

        # Filter: production code only (exclude test files)
        samples = [s for s in samples if self._is_production_code(s)]
        logger.info(f"Loaded {len(samples)} production code samples from MaRV")
        return samples

    # ------------------------------------------------------------------
    # Qualitas Corpus loader (Java source code, no built-in labels)
    # ------------------------------------------------------------------

    def load_qualitas_corpus(self) -> List[CodeSample]:
        """Load Java source files from Qualitas Corpus (unlabeled)."""
        if not QUALITAS_CORPUS_DIR.exists():
            logger.warning(f"Qualitas Corpus directory not found: {QUALITAS_CORPUS_DIR}")
            return []

        samples: List[CodeSample] = []
        for java_file in QUALITAS_CORPUS_DIR.rglob("*.java"):
            # Skip test files
            rel = str(java_file.relative_to(QUALITAS_CORPUS_DIR))
            if self._is_test_file_path(rel):
                continue
            code = java_file.read_text(encoding="utf-8", errors="replace")
            samples.append(CodeSample(
                sample_id=f"qualitas_{java_file.stem}_{len(samples)}",
                source_code=code,
                language="java",
                class_name=java_file.stem,
                file_path=rel,
                dataset="QualitasCorpus",
            ))

        logger.info(f"Loaded {len(samples)} Java files from Qualitas Corpus")
        return samples

    # ------------------------------------------------------------------
    # Generic loaders
    # ------------------------------------------------------------------

    def load_from_json(self, path: Path) -> List[CodeSample]:
        """Load code samples from a JSON file."""
        return self._load_json_samples(path, dataset=path.stem)

    def load_from_csv(self, path: Path) -> List[CodeSample]:
        """Load code samples from a CSV file."""
        return self._load_csv_samples(path, dataset=path.stem)

    def load_from_xml(self, path: Path) -> List[CodeSample]:
        """Load code samples from an XML file (e.g., PMD/Checkstyle reports)."""
        if not path.exists():
            logger.error(f"XML file not found: {path}")
            return []
        tree = ET.parse(path)  # noqa: S314 - trusted local files only
        root = tree.getroot()
        samples: List[CodeSample] = []
        for file_elem in root.iter("file"):
            file_path = file_elem.get("name", "")
            for violation in file_elem.iter("violation"):
                samples.append(CodeSample(
                    sample_id=f"xml_{path.stem}_{len(samples)}",
                    source_code=violation.text or "",
                    language=self._detect_language_from_path(file_path),
                    file_path=file_path,
                    dataset=path.stem,
                    annotations=[SmellAnnotation(
                        smell_type=violation.get("rule", "Unknown"),
                        category=violation.get("ruleset", ""),
                        description=violation.text or "",
                    )],
                ))
        logger.info(f"Loaded {len(samples)} entries from XML: {path.name}")
        return samples

    # ------------------------------------------------------------------
    # Load all available datasets
    # ------------------------------------------------------------------

    def load_all(self) -> List[CodeSample]:
        """Load samples from all available datasets."""
        all_samples: List[CodeSample] = []
        all_samples.extend(self.load_smelly_code_dataset())
        all_samples.extend(self.load_marv_dataset())
        all_samples.extend(self.load_qualitas_corpus())

        # Load any additional JSONs in smelly_code/
        if SMELLY_CODE_DIR.exists():
            for f in SMELLY_CODE_DIR.glob("*.json"):
                all_samples.extend(self._load_json_samples(f, dataset="smelly_code"))

        logger.info(f"Total samples loaded: {len(all_samples)} "
                     f"(annotated: {sum(s.has_smells for s in all_samples)})")
        return all_samples

    # ------------------------------------------------------------------
    # Ground truth loader
    # ------------------------------------------------------------------

    def load_ground_truth(self) -> List[CodeSample]:
        """Load manually verified ground truth from data/ground_truth/."""
        if not GROUND_TRUTH_DIR.exists():
            return []
        samples: List[CodeSample] = []
        for f in GROUND_TRUTH_DIR.glob("*.json"):
            samples.extend(self._load_json_samples(f, dataset="ground_truth"))
        for f in GROUND_TRUTH_DIR.glob("*.csv"):
            samples.extend(self._load_csv_samples(f, dataset="ground_truth"))
        logger.info(f"Loaded {len(samples)} ground truth samples")
        return samples

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_json_samples(self, path: Path, dataset: str) -> List[CodeSample]:
        """Load samples from a JSON file. Supports list-of-dicts or nested formats."""
        if not path.exists():
            return []
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            items = data
        elif isinstance(data, dict) and "samples" in data:
            items = data["samples"]
        else:
            items = [data]

        samples = []
        for i, item in enumerate(items):
            annotations = [
                SmellAnnotation(
                    smell_type=a.get("smell_type", a.get("type", "")),
                    category=a.get("category", ""),
                    method=a.get("method", ""),
                    description=a.get("description", ""),
                )
                for a in item.get("annotations", item.get("smells", []))
            ]
            samples.append(CodeSample(
                sample_id=item.get("id", f"{dataset}_{path.stem}_{i}"),
                source_code=item.get("source_code", item.get("code", "")),
                language=item.get("language", "java"),
                class_name=item.get("class_name", item.get("class", "")),
                file_path=item.get("file_path", ""),
                dataset=dataset,
                annotations=annotations,
                metadata=item.get("metadata", {}),
            ))
        return samples

    def _load_csv_samples(self, path: Path, dataset: str) -> List[CodeSample]:
        """Load samples from a CSV file. Groups rows by sample ID or class name."""
        if not path.exists():
            return []
        samples_dict: Dict[str, CodeSample] = {}
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = row.get("sample_id") or row.get("Class", "") or str(len(samples_dict))
                lang = row.get("language", row.get("Language", "java")).strip().lower()
                if key not in samples_dict:
                    samples_dict[key] = CodeSample(
                        sample_id=f"{dataset}_{key}",
                        source_code=row.get("source_code", row.get("code", "")),
                        language=lang,
                        class_name=row.get("Class", key),
                        dataset=dataset,
                    )
                smell_type = row.get("smell_type", row.get("Code Smell", "")).strip()
                if smell_type:
                    samples_dict[key].annotations.append(SmellAnnotation(
                        smell_type=smell_type,
                        category=row.get("category", row.get("Category", "")),
                        method=row.get("method", row.get("Method", "")),
                        description=row.get("description", row.get("Type", "")),
                    ))
        return list(samples_dict.values())

    @staticmethod
    def _is_production_code(sample: CodeSample) -> bool:
        """Filter: True if sample is production code (not test code)."""
        indicators = ["test", "Test", "mock", "Mock", "spec", "Spec"]
        if any(ind in sample.class_name for ind in indicators):
            return False
        if any(ind in sample.file_path for ind in ["/test/", "/tests/", "/Test"]):
            return False
        return True

    @staticmethod
    def _is_test_file_path(path: str) -> bool:
        """Check if a file path looks like a test file."""
        path_lower = path.lower()
        return any(p in path_lower for p in ["/test/", "/tests/", "test_", "_test."])

    @staticmethod
    def _detect_language_from_path(path: str) -> str:
        """Detect programming language from file extension."""
        ext_map = {".py": "python", ".java": "java", ".js": "javascript", ".cpp": "cpp"}
        for ext, lang in ext_map.items():
            if path.endswith(ext):
                return lang
        return "unknown"


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    loader = DatasetLoader()
    samples = loader.load_all()
    print(f"\nTotal samples: {len(samples)}")
    by_dataset = {}
    for s in samples:
        by_dataset.setdefault(s.dataset, []).append(s)
    for ds, ds_samples in by_dataset.items():
        annotated = sum(1 for s in ds_samples if s.has_smells)
        print(f"  {ds}: {len(ds_samples)} samples ({annotated} annotated)")
