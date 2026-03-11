#!/usr/bin/env python3
"""
Error Analysis Module (Phase 4.1)

Analyzes detection errors to understand failure modes and identify
improvement opportunities. Creates failure taxonomy and case studies.

Architecture: Phase 4 error analysis (WBS Section 4.1)
Reference: Benchmarking Strategy Sections 6-7, Gap #11
"""

import json
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pandas as pd
from enum import Enum

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    CODE_SMELL_TYPES,
    DATA_DIR,
    RESULTS_DIR,
)
from src.data.data_preprocessor import DataPreprocessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

PROCESSED_DIR = DATA_DIR / "processed"
ERROR_ANALYSIS_DIR = RESULTS_DIR / "error_analysis"


class ErrorType(Enum):
    """Classification of detection errors."""

    FALSE_POSITIVE = "false_positive"  # Detected smell not in ground truth
    FALSE_NEGATIVE = "false_negative"  # Missed smell in ground truth
    PARTIAL_MATCH = "partial_match"  # Detected subset of true smells
    WRONG_TYPE = "wrong_type"  # Detected different smell type


class ComplexityLevel(Enum):
    """Code complexity levels for error pattern analysis."""

    SIMPLE = "simple"  # <50 LOC
    MODERATE = "moderate"  # 50-200 LOC
    COMPLEX = "complex"  # >200 LOC


@dataclass
class FailedDetection:
    """Single detection failure with context."""

    sample_id: str
    code: str
    language: str
    error_type: ErrorType
    detected_smells: List[str] = field(default_factory=list)
    true_smells: List[str] = field(default_factory=list)
    loc: int = 0
    complexity: ComplexityLevel = ComplexityLevel.MODERATE
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    pattern: str = ""  # e.g., "missing_large_method", "hallucinated_smell"
    explanation: str = ""


class ErrorAnalyzer:
    """
    Analyzes detection errors and identifies failure patterns.

    Phase 4.1: Categorizes false positives/negatives, identifies patterns,
    analyzes AI-generated vs human code, and creates failure taxonomy.
    """

    def __init__(self):
        """Initialize error analyzer."""
        self.failed_detections: List[FailedDetection] = []
        self.error_summary: Dict[str, Any] = {}
        self.patterns: Dict[str, List[FailedDetection]] = defaultdict(list)

    def add_error(
        self,
        sample_id: str,
        code: str,
        language: str,
        error_type: ErrorType,
        detected_smells: List[str],
        true_smells: List[str],
        template: Dict[str, Any] = None,
    ):
        """Add a failed detection for analysis.

        Args:
            sample_id: Identifier for the code sample.
            code: Source code of the sample.
            language: Programming language.
            error_type: Type of error.
            detected_smells: Smells detected by system.
            true_smells: Ground truth smells.
            template: Optional template with metadata (LOC, complexity, etc).
        """
        template = template or {}

        # Determine complexity
        loc = len(code.split("\n"))
        if loc < 50:
            complexity = ComplexityLevel.SIMPLE
        elif loc < 200:
            complexity = ComplexityLevel.MODERATE
        else:
            complexity = ComplexityLevel.COMPLEX

        # Determine pattern
        pattern = self._classify_pattern(
            error_type, detected_smells, true_smells, len(true_smells)
        )

        failure = FailedDetection(
            sample_id=sample_id,
            code=code,
            language=language,
            error_type=error_type,
            detected_smells=detected_smells,
            true_smells=true_smells,
            loc=loc,
            complexity=complexity,
            confidence_scores=template.get("confidence_scores", {}),
            pattern=pattern,
            explanation=template.get("explanation", ""),
        )

        self.failed_detections.append(failure)
        self.patterns[pattern].append(failure)

        logger.debug(f"Added error: {sample_id} ({error_type.value})")

    def _classify_pattern(
        self,
        error_type: ErrorType,
        detected: List[str],
        true: List[str],
        num_true: int,
    ) -> str:
        """Classify error pattern.

        Args:
            error_type: Type of error.
            detected: Detected smells.
            true: True smells.
            num_true: Count of true smells.

        Returns:
            Pattern name.
        """
        if error_type == ErrorType.FALSE_POSITIVE:
            if len(detected) > 2:
                return "hallucination_multiple"
            else:
                return "hallucination_single"

        elif error_type == ErrorType.FALSE_NEGATIVE:
            if num_true == 1:
                return "missed_single_smell"
            else:
                return "missed_multiple_smells"

        elif error_type == ErrorType.PARTIAL_MATCH:
            return "partial_detection"

        elif error_type == ErrorType.WRONG_TYPE:
            return "wrong_smell_type"

        return "unknown"

    def analyze_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Analyze error patterns and prevalence.

        Returns:
            Dict with pattern statistics.
        """
        analysis = {}

        for pattern_name, failures in self.patterns.items():
            if not failures:
                continue

            analysis[pattern_name] = {
                "count": len(failures),
                "percentage": len(failures) / len(self.failed_detections) * 100
                if self.failed_detections
                else 0,
                "by_complexity": self._count_by_complexity(failures),
                "by_language": self._count_by_language(failures),
                "avg_loc": sum(f.loc for f in failures) / len(failures)
                if failures
                else 0,
            }

        return analysis

    def _count_by_complexity(self, failures: List[FailedDetection]) -> Dict[str, int]:
        """Count failures by code complexity.

        Args:
            failures: List of failures.

        Returns:
            Dict with complexity level counts.
        """
        counts = defaultdict(int)
        for failure in failures:
            counts[failure.complexity.value] += 1
        return dict(counts)

    def _count_by_language(self, failures: List[FailedDetection]) -> Dict[str, int]:
        """Count failures by programming language.

        Args:
            failures: List of failures.

        Returns:
            Dict with language counts.
        """
        counts = defaultdict(int)
        for failure in failures:
            counts[failure.language] += 1
        return dict(counts)

    def select_case_studies(self, num_per_pattern: int = 3) -> Dict[str, List[FailedDetection]]:
        """Select representative case study examples.

        Args:
            num_per_pattern: Number of examples per pattern.

        Returns:
            Dict mapping pattern name to selected case studies.
        """
        case_studies = {}

        for pattern_name, failures in self.patterns.items():
            if not failures:
                continue

            # Select diverse examples
            selected = []

            # Include simple, moderate, and complex examples if available
            for complexity in [ComplexityLevel.SIMPLE, ComplexityLevel.MODERATE, ComplexityLevel.COMPLEX]:
                for failure in failures:
                    if failure.complexity == complexity and len(selected) < num_per_pattern:
                        selected.append(failure)

            # Fill remaining slots with any failures
            for failure in failures:
                if failure not in selected and len(selected) < num_per_pattern:
                    selected.append(failure)

            case_studies[pattern_name] = selected

        return case_studies

    def generate_failure_taxonomy(self) -> Dict[str, Any]:
        """Generate comprehensive failure taxonomy.

        Returns:
            Dict with failure taxonomy structure.
        """
        taxonomy = {
            "timestamp": datetime.now().isoformat(),
            "total_failures": len(self.failed_detections),
            "error_types": self._count_error_types(),
            "patterns": self.analyze_patterns(),
            "complexity_analysis": self._analyze_complexity_correlation(),
            "language_analysis": self._analyze_language_correlation(),
        }

        return taxonomy

    def _count_error_types(self) -> Dict[str, int]:
        """Count failures by error type.

        Returns:
            Dict with error type counts.
        """
        counts = defaultdict(int)
        for failure in self.failed_detections:
            counts[failure.error_type.value] += 1
        return dict(counts)

    def _analyze_complexity_correlation(self) -> Dict[str, Any]:
        """Analyze correlation between code complexity and error rate.

        Returns:
            Dict with complexity analysis.
        """
        by_complexity = defaultdict(list)
        for failure in self.failed_detections:
            by_complexity[failure.complexity.value].append(failure)

        analysis = {}
        for complexity, failures in by_complexity.items():
            analysis[complexity] = {
                "error_count": len(failures),
                "avg_loc": sum(f.loc for f in failures) / len(failures) if failures else 0,
                "patterns": list(set(f.pattern for f in failures)),
            }

        return analysis

    def _analyze_language_correlation(self) -> Dict[str, Any]:
        """Analyze error rate by programming language.

        Returns:
            Dict with language analysis.
        """
        by_language = defaultdict(list)
        for failure in self.failed_detections:
            by_language[failure.language].append(failure)

        analysis = {}
        for language, failures in by_language.items():
            analysis[language] = {
                "error_count": len(failures),
                "error_types": self._count_by_error_type(failures),
            }

        return analysis

    def _count_by_error_type(self, failures: List[FailedDetection]) -> Dict[str, int]:
        """Count failures by error type.

        Args:
            failures: List of failures.

        Returns:
            Dict with error type counts.
        """
        counts = defaultdict(int)
        for failure in failures:
            counts[failure.error_type.value] += 1
        return dict(counts)

    def save_analysis(self, output_dir: Path = None) -> Path:
        """Save error analysis results.

        Args:
            output_dir: Output directory. Defaults to ERROR_ANALYSIS_DIR.

        Returns:
            Path where results were saved.
        """
        output_dir = output_dir or ERROR_ANALYSIS_DIR
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save taxonomy
        taxonomy = self.generate_failure_taxonomy()
        with open(output_dir / f"failure_taxonomy_{timestamp}.json", "w") as f:
            json.dump(taxonomy, f, indent=2, default=str)

        # Save case studies
        case_studies = self.select_case_studies()
        case_studies_data = {}
        for pattern, samples in case_studies.items():
            case_studies_data[pattern] = [
                {
                    "sample_id": s.sample_id,
                    "error_type": s.error_type.value,
                    "complexity": s.complexity.value,
                    "loc": s.loc,
                    "detected_smells": s.detected_smells,
                    "true_smells": s.true_smells,
                    "code_excerpt": s.code[:500],  # First 500 chars
                }
                for s in samples
            ]

        with open(output_dir / f"case_studies_{timestamp}.json", "w") as f:
            json.dump(case_studies_data, f, indent=2, default=str)

        # Save detailed error log
        error_log = []
        for failure in self.failed_detections:
            error_log.append({
                "sample_id": failure.sample_id,
                "error_type": failure.error_type.value,
                "pattern": failure.pattern,
                "language": failure.language,
                "complexity": failure.complexity.value,
                "loc": failure.loc,
                "detected_smells": failure.detected_smells,
                "true_smells": failure.true_smells,
                "confidence": failure.confidence_scores,
            })

        import csv

        csv_file = output_dir / f"error_log_{timestamp}.csv"
        if error_log:
            with open(csv_file, "w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=error_log[0].keys(),
                )
                writer.writeheader()
                for row in error_log:
                    row["confidence"] = str(row["confidence"])
                    writer.writerow(row)

        logger.info(f"Error analysis saved to {output_dir}")
        return output_dir

    def print_summary(self):
        """Print error analysis summary."""
        if not self.failed_detections:
            print("No errors to analyze")
            return

        print("\n" + "=" * 70)
        print("ERROR ANALYSIS SUMMARY")
        print("=" * 70)

        print(f"\nTotal Errors: {len(self.failed_detections)}")

        # By error type
        error_counts = self._count_error_types()
        print("\nBy Error Type:")
        for error_type, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True):
            pct = count / len(self.failed_detections) * 100
            print(f"  {error_type:25s} {count:4d} ({pct:5.1f}%)")

        # By pattern
        patterns = self.analyze_patterns()
        print("\nTop Error Patterns:")
        sorted_patterns = sorted(
            patterns.items(),
            key=lambda x: x[1]["count"],
            reverse=True,
        )
        for pattern, stats in sorted_patterns[:5]:
            print(f"  {pattern:30s} {stats['count']:4d} ({stats['percentage']:5.1f}%)")

        # By complexity
        complexity_analysis = self._analyze_complexity_correlation()
        print("\nBy Code Complexity:")
        for complexity, stats in complexity_analysis.items():
            print(f"  {complexity:15s} {stats['error_count']:4d} errors, avg LOC: {stats['avg_loc']:.0f}")

        print("=" * 70)


def main():
    """Example error analysis workflow."""
    import argparse

    parser = argparse.ArgumentParser(description="Phase 4.1: Error Analysis")
    parser.add_argument(
        "--test-split",
        type=Path,
        default=PROCESSED_DIR / "test.json",
        help="Path to test.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ERROR_ANALYSIS_DIR,
        help="Output directory",
    )
    args = parser.parse_args()

    # Initialize analyzer
    analyzer = ErrorAnalyzer()

    # Load samples for context
    try:
        split = DataPreprocessor.load_split(args.test_split.parent)
        samples = split.test

        logger.info(f"Loaded {len(samples)} test samples for error context")

    except Exception as e:
        logger.warning(f"Could not load test samples: {e}")
        samples = []

    # Example: Add some errors (in real usage, would come from evaluation results)
    logger.info("Error analysis module ready for integration with evaluation.py")
    logger.info("To use: from src.error_analysis import ErrorAnalyzer")

    # Save empty analysis as template
    analyzer.save_analysis(args.output_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
