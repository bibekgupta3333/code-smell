"""
Quantitative Evaluation Module (Phase 4.1)

Comprehensive evaluation framework for code smell detection systems.
Loads ground truth, predictions from multiple systems, normalizes formats,
and produces detailed metrics, statistical tests, and visualizations.

Architecture: Phase 4 evaluation (WBS Sections 4.1-4.3)
Reference: Benchmarking Strategy Sections 1-7
"""

import json
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    CODE_SMELL_TYPES,
    DATA_DIR,
    METRICS_DIR,
    RESULTS_DIR,
)
from src.utils.benchmark_utils import (
    calculate_metrics,
    build_confusion_matrix,
    per_smell_breakdown,
    mcnemars_test,
    paired_t_test,
    cohens_d,
    bootstrap_confidence_interval,
)
from src.data.data_preprocessor import DataPreprocessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

PROCESSED_DIR = DATA_DIR / "processed"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"


@dataclass
class EvaluationResult:
    """Complete evaluation result for one system."""

    system_name: str
    metrics: Dict[str, float]
    per_smell_breakdown: pd.DataFrame
    confusion_matrix: pd.DataFrame
    y_true: List[str]
    y_pred: List[str]
    quality_metrics: Dict[str, Any]
    timestamp: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class EvaluationFramework:
    """
    Comprehensive evaluation framework for comparing multiple detection systems.

    Phase 4.1: Loads ground truth, predictions, calculates metrics,
    runs statistical tests, and compares baseline vs LLM approaches.
    """

    def __init__(self, test_split_path: Path = None):
        """Initialize evaluation framework.

        Args:
            test_split_path: Path to test.json with ground truth.
        """
        self.test_split_path = test_split_path or PROCESSED_DIR / "test.json"
        self.ground_truth = None
        self.ground_truth_labels = None  # For each sample: [smell1, smell2, ...]
        self.results: Dict[str, EvaluationResult] = {}

    def load_ground_truth(self) -> bool:
        """Load ground truth from test.json.

        Returns:
            True if successful, False otherwise.
        """
        if not self.test_split_path.exists():
            logger.error(f"Test split not found: {self.test_split_path}")
            return False

        try:
            # Load via DataPreprocessor
            split = DataPreprocessor.load_split(self.test_split_path.parent)
            samples = split.test

            self.ground_truth = samples
            self.ground_truth_labels = [
                sample.smell_types if sample.smell_types else ["no_smell"]
                for sample in samples
            ]

            logger.info(f"Loaded {len(samples)} ground truth samples")
            return True

        except Exception as e:
            logger.error(f"Failed to load ground truth: {e}")
            return False

    def normalize_predictions(
        self,
        predictions: Dict[str, Any],
        system_type: str = "llm",
    ) -> List[List[str]]:
        """Normalize predictions from different systems to smell_type lists.

        Args:
            predictions: Raw predictions from system.
            system_type: Type of system ('llm', 'baseline_tool', 'json_list').

        Returns:
            List of predicted smell lists for each sample.
        """
        if system_type == "json_list":
            # Simple list of lists format
            if isinstance(predictions, list):
                return predictions

        elif system_type == "llm":
            # LLM predictions: dict with detected_smells or similar
            if isinstance(predictions, dict):
                # Try common keys
                for key in ["detected_smells", "smells", "findings"]:
                    if key in predictions:
                        pred_list = predictions[key]
                        if isinstance(pred_list, list):
                            return pred_list
                        return [[pred_list]]

            if isinstance(predictions, list):
                return predictions

        elif system_type == "baseline_tool":
            # Baseline tool predictions: may vary by tool
            # e.g., {"issues": [{"type": "...", ...}, ...]}
            if isinstance(predictions, dict):
                # Try common structures
                for key in ["issues", "findings", "violations", "detections"]:
                    if key in predictions:
                        items = predictions[key]
                        if isinstance(items, list):
                            # Extract type/category from each item
                            return [[
                                item.get("type", item.get("category", "unknown"))
                                for item in items
                            ]]

        logger.warning(f"Could not normalize predictions of type {system_type}")
        return [[]]

    def evaluate_system(
        self,
        system_name: str,
        predictions: Dict[str, Any],
        system_type: str = "llm",
    ) -> EvaluationResult:
        """Evaluate a single detection system.

        Args:
            system_name: Name of the system (e.g., "LLM-Vanilla", "PMD").
            predictions: Predictions from the system.
            system_type: Type of system for normalization.

        Returns:
            EvaluationResult with all metrics.
        """
        if self.ground_truth is None:
            raise ValueError("Ground truth not loaded. Call load_ground_truth() first.")

        # Normalize predictions
        pred_lists = self.normalize_predictions(predictions, system_type)

        if not pred_lists:
            logger.error(f"No predictions normalized for {system_name}")
            return EvaluationResult(
                system_name=system_name,
                metrics={},
                per_smell_breakdown=pd.DataFrame(),
                confusion_matrix=pd.DataFrame(),
                y_true=[],
                y_pred=[],
                quality_metrics={},
            )

        # Flatten: Convert multi-smell to per-smell labels for metrics
        y_true_flat = []
        y_pred_flat = []

        for i, sample in enumerate(self.ground_truth):
            true_smells = self.ground_truth_labels[i]
            if i < len(pred_lists):
                pred_smells = pred_lists[i] if pred_lists[i] else ["no_smell"]
            else:
                pred_smells = ["no_smell"]

            # Normalize empty/None to no_smell
            true_smells = [s for s in true_smells if s] or ["no_smell"]
            pred_smells = [s for s in pred_smells if s] or ["no_smell"]

            # For each ground truth smell, check if predicted
            for smell in true_smells:
                y_true_flat.append(smell)
                y_pred_flat.append(smell if smell in pred_smells else "missed")

            # For predicted smells not in ground truth, mark as false positive
            for smell in pred_smells:
                if smell not in true_smells and smell != "no_smell":
                    y_true_flat.append("no_smell")
                    y_pred_flat.append(smell)

        # Calculate metrics
        metrics = calculate_metrics(y_true_flat, y_pred_flat)
        breakdown = per_smell_breakdown(y_true_flat, y_pred_flat, CODE_SMELL_TYPES)
        cm = build_confusion_matrix(y_true_flat, y_pred_flat)

        # Quality metrics
        quality = self._calculate_quality_metrics(
            self.ground_truth, predictions, pred_lists
        )

        result = EvaluationResult(
            system_name=system_name,
            metrics=metrics,
            per_smell_breakdown=breakdown,
            confusion_matrix=cm,
            y_true=y_true_flat,
            y_pred=y_pred_flat,
            quality_metrics=quality,
        )

        self.results[system_name] = result
        logger.info(f"Evaluated {system_name}: F1={metrics.get('f1', 0):.4f}")

        return result

    def _calculate_quality_metrics(
        self,
        samples: List[Any],
        predictions: Dict[str, Any],
        pred_lists: List[List[str]],
    ) -> Dict[str, Any]:
        """Calculate quality metrics beyond standard ML metrics.

        Args:
            samples: Ground truth samples.
            predictions: Raw predictions (may contain confidence scores, etc).
            pred_lists: Normalized prediction lists.

        Returns:
            Dict with quality metrics.
        """
        metrics = {}

        # True Positive Rate (Recall)
        if hasattr(self, "ground_truth_labels") and self.ground_truth_labels:
            annotated = sum(1 for labels in self.ground_truth_labels if labels != ["no_smell"])
            if annotated > 0:
                tp = sum(
                    1 for i, sample_preds in enumerate(pred_lists)
                    if i < len(self.ground_truth_labels)
                    and any(smell in sample_preds for smell in self.ground_truth_labels[i])
                    if self.ground_truth_labels[i] != ["no_smell"]
                )
                metrics["true_positive_rate"] = tp / annotated

        # False Positive Rate
        if hasattr(self, "ground_truth_labels") and self.ground_truth_labels:
            non_annotated = sum(1 for labels in self.ground_truth_labels if labels == ["no_smell"])
            if non_annotated > 0:
                fp = sum(
                    1 for i, sample_preds in enumerate(pred_lists)
                    if i < len(self.ground_truth_labels)
                    and sample_preds != ["no_smell"]
                    if self.ground_truth_labels[i] == ["no_smell"]
                )
                metrics["false_positive_rate"] = fp / non_annotated

        # Confidence calibration (if confidence scores available)
        if isinstance(predictions, dict) and "confidence_scores" in predictions:
            confidences = predictions["confidence_scores"]
            if confidences:
                metrics["mean_confidence"] = float(np.mean(confidences))
                metrics["confidence_std"] = float(np.std(confidences))

        return metrics

    def compare_systems(
        self,
        system_a: str,
        system_b: str,
        report_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Compare two systems with statistical tests.

        Args:
            system_a: Name of first system.
            system_b: Name of second system.
            report_path: Optional path to save comparison report.

        Returns:
            Dict with comparison results.
        """
        if system_a not in self.results or system_b not in self.results:
            logger.error(f"Systems not found: {system_a}, {system_b}")
            return {}

        result_a = self.results[system_a]
        result_b = self.results[system_b]

        comparison = {
            "system_a": system_a,
            "system_b": system_b,
            "metrics_diff": {
                "precision": result_a.metrics["precision"] - result_b.metrics["precision"],
                "recall": result_a.metrics["recall"] - result_b.metrics["recall"],
                "f1": result_a.metrics["f1"] - result_b.metrics["f1"],
            },
        }

        # Statistical tests (if we have per-sample F1 scores)
        if hasattr(result_a, "_per_sample_f1s") and hasattr(result_b, "_per_sample_f1s"):
            comparison["paired_t_test"] = paired_t_test(
                result_a._per_sample_f1s, result_b._per_sample_f1s
            )
            comparison["cohens_d"] = cohens_d(
                result_a._per_sample_f1s, result_b._per_sample_f1s
            )

        if report_path:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            with open(report_path, "w") as f:
                json.dump(comparison, f, indent=2, default=str)

        return comparison

    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate overall evaluation summary.

        Returns:
            Summary dict with key metrics and rankings.
        """
        summary = {
            "generated_at": datetime.now().isoformat(),
            "num_samples": len(self.ground_truth),
            "systems_evaluated": len(self.results),
            "system_rankings": {},
            "overall_metrics": {},
        }

        # Rank systems by F1 score
        sorted_systems = sorted(
            self.results.items(),
            key=lambda x: x[1].metrics.get("f1", 0),
            reverse=True,
        )

        for rank, (name, result) in enumerate(sorted_systems, 1):
            summary["system_rankings"][rank] = {
                "system": name,
                "f1": round(result.metrics.get("f1", 0), 4),
                "precision": round(result.metrics.get("precision", 0), 4),
                "recall": round(result.metrics.get("recall", 0), 4),
                "accuracy": round(result.metrics.get("accuracy", 0), 4),
            }

        return summary

    def save_results(self, output_dir: Path = None) -> Path:
        """Save all evaluation results to disk.

        Args:
            output_dir: Directory to save results. Defaults to METRICS_DIR.

        Returns:
            Path where results were saved.
        """
        output_dir = output_dir or METRICS_DIR
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save each system's results
        for system_name, result in self.results.items():
            system_dir = output_dir / system_name
            system_dir.mkdir(parents=True, exist_ok=True)

            # Metrics JSON
            with open(system_dir / f"metrics_{timestamp}.json", "w") as f:
                json.dump(result.metrics, f, indent=2)

            # Per-smell breakdown CSV
            result.per_smell_breakdown.to_csv(
                system_dir / f"per_smell_breakdown_{timestamp}.csv"
            )

            # Confusion matrix CSV
            result.confusion_matrix.to_csv(
                system_dir / f"confusion_matrix_{timestamp}.csv"
            )

        # Save summary report
        summary = self.generate_summary_report()
        with open(output_dir / f"summary_report_{timestamp}.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info(f"Results saved to {output_dir}")
        return output_dir


def load_predictions_from_files(directory: Path) -> Dict[str, Dict[str, Any]]:
    """Load all prediction files from results/predictions/ directory.

    Args:
        directory: Path to predictions directory.

    Returns:
        Dict mapping system_name to predictions dict.
    """
    predictions = {}

    if not directory.exists():
        logger.warning(f"Predictions directory not found: {directory}")
        return predictions

    for system_dir in directory.iterdir():
        if not system_dir.is_dir():
            continue

        system_name = system_dir.name

        # Look for most recent prediction JSON
        json_files = list(system_dir.glob("*.json"))
        if not json_files:
            logger.warning(f"No prediction files in {system_name}")
            continue

        latest_file = max(json_files, key=lambda p: p.stat().st_mtime)

        try:
            with open(latest_file) as f:
                predictions[system_name] = json.load(f)
                logger.info(f"Loaded predictions for {system_name}")
        except Exception as e:
            logger.error(f"Failed to load {latest_file}: {e}")

    return predictions


def main():
    """Run comprehensive Phase 4.1 evaluation."""
    import argparse

    parser = argparse.ArgumentParser(description="Phase 4.1: Quantitative Evaluation")
    parser.add_argument(
        "--test-split",
        type=Path,
        default=PROCESSED_DIR / "test.json",
        help="Path to test.json",
    )
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        default=PREDICTIONS_DIR,
        help="Path to predictions directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=METRICS_DIR,
        help="Output directory for results",
    )
    args = parser.parse_args()

    # Initialize framework
    framework = EvaluationFramework(args.test_split)
    if not framework.load_ground_truth():
        logger.error("Failed to load ground truth")
        return 1

    # Load predictions
    predictions_dict = load_predictions_from_files(args.predictions_dir)
    if not predictions_dict:
        logger.warning("No predictions found to evaluate")
        return 1

    # Evaluate each system
    for system_name, predictions in predictions_dict.items():
        framework.evaluate_system(system_name, predictions, system_type="llm")

    # Generate report
    summary = framework.generate_summary_report()
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Systems evaluated: {summary['systems_evaluated']}")
    print(f"Test samples: {summary['num_samples']}\n")

    print("Ranking by F1-Score:")
    for rank, metrics in summary["system_rankings"].items():
        print(
            f"  {rank}. {metrics['system']:20s} "
            f"F1={metrics['f1']:.4f} P={metrics['precision']:.4f} "
            f"R={metrics['recall']:.4f}"
        )

    # Save results
    framework.save_results(args.output_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
