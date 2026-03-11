#!/usr/bin/env python3
"""
Ablation Study Runner (Phase 4.2)

Systematically runs ablations to evaluate impact of design choices.
Uses local llama model (no external APIs) with parametric variations.

Architecture: Phase 4.2 ablations (WBS Section 4.2)
Reference: Benchmarking Strategy Section 5 (Cross-Validation)
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    CODE_SMELL_TYPES,
    RESULTS_DIR,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

ABLATION_DIR = RESULTS_DIR / "ablation_studies"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"


class AblationStudy:
    """
    Systematic ablation study runner.

    Phase 4.2: Tests impact of:
    - RAG vs No RAG (RQ2)
    - Top-k retrieval values
    - Few-shot examples count
    - Temperature settings
    - Embedding models (with local llama only)
    """

    def __init__(self, config_path: Path):
        """Initialize ablation study from config.

        Args:
            config_path: Path to ablation config.json
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.results: Dict[str, Dict[str, Any]] = {}

    def _load_config(self) -> Dict[str, Any]:
        """Load ablation configuration.

        Returns:
            Configuration dict.
        """
        if not self.config_path.exists():
            logger.error(f"Config not found: {self.config_path}")
            return {}

        with open(self.config_path) as f:
            return json.load(f)

    def run_ablation(self, ablation_name: str) -> bool:
        """Run a single ablation variation.

        Args:
            ablation_name: Name of ablation to run.

        Returns:
            True if successful.
        """
        if ablation_name not in self.config.get("ablations", {}):
            logger.error(f"Ablation not found: {ablation_name}")
            return False

        ablation_config = self.config["ablations"][ablation_name]
        logger.info(f"Running ablation: {ablation_name}")
        logger.info(f"Config: {json.dumps(ablation_config, indent=2)}")

        # Import runner here to avoid circular deps
        try:
            from scripts.run_experiment import ExperimentExecutor

            executor = ExperimentExecutor(experiment_config=ablation_config)
            metrics = executor.run_batch(
                max_samples=ablation_config.get("max_samples", 100),
            )

            self.results[ablation_name] = {
                "config": ablation_config,
                "metrics": metrics,
                "timestamp": datetime.now().isoformat(),
            }

            logger.info(f"✓ Ablation {ablation_name} completed: F1={metrics.get('f1', 0):.4f}")
            return True

        except Exception as e:
            logger.error(f"✗ Ablation {ablation_name} failed: {e}")
            return False

    def run_all_ablations(self) -> int:
        """Run all configured ablations.

        Returns:
            Number of successful ablations.
        """
        ablations = self.config.get("ablations", {})
        if not ablations:
            logger.error("No ablations configured")
            return 0

        successful = 0
        for ablation_name in ablations:
            if self.run_ablation(ablation_name):
                successful += 1

        return successful

    def save_results(self, output_dir: Path = None) -> Path:
        """Save ablation results to disk.

        Args:
            output_dir: Output directory. Defaults to ABLATION_DIR.

        Returns:
            Path where results were saved.
        """
        output_dir = output_dir or ABLATION_DIR
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = output_dir / f"ablation_results_{timestamp}.json"

        with open(results_file, "w") as f:
            json.dump(self.results, f, indent=2, default=str)

        logger.info(f"Results saved to {results_file}")

        # Also save summary CSV for easier analysis
        self._save_summary_csv(output_dir, timestamp)

        return output_dir

    def _save_summary_csv(self, output_dir: Path, timestamp: str):
        """Save ablation results as CSV for analysis.

        Args:
            output_dir: Output directory.
            timestamp: Timestamp for filename.
        """
        import csv

        summary_file = output_dir / f"ablation_summary_{timestamp}.csv"

        with open(summary_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Ablation",
                "F1-Score",
                "Precision",
                "Recall",
                "Accuracy",
                "Config",
                "Timestamp",
            ])

            for ablation_name, result in sorted(self.results.items()):
                metrics = result.get("metrics", {})
                config_str = json.dumps(result.get("config", {}), default=str)

                writer.writerow([
                    ablation_name,
                    f"{metrics.get('f1', 0):.4f}",
                    f"{metrics.get('precision', 0):.4f}",
                    f"{metrics.get('recall', 0):.4f}",
                    f"{metrics.get('accuracy', 0):.4f}",
                    config_str[:100],  # Truncate for readability
                    result.get("timestamp", ""),
                ])

        logger.info(f"Summary saved to {summary_file}")

    def print_results(self):
        """Print ablation results to console."""
        if not self.results:
            logger.warning("No results to print")
            return

        print("\n" + "=" * 80)
        print("ABLATION STUDY RESULTS")
        print("=" * 80)

        # Sort by F1-Score descending
        sorted_results = sorted(
            self.results.items(),
            key=lambda x: x[1].get("metrics", {}).get("f1", 0),
            reverse=True,
        )

        print(f"\n{'Ablation':<40} {'F1':>10} {'P':>10} {'R':>10} {'Acc':>10}")
        print("-" * 80)

        for ablation_name, result in sorted_results:
            metrics = result.get("metrics", {})
            print(
                f"{ablation_name:<40} "
                f"{metrics.get('f1', 0):>10.4f} "
                f"{metrics.get('precision', 0):>10.4f} "
                f"{metrics.get('recall', 0):>10.4f} "
                f"{metrics.get('accuracy', 0):>10.4f}"
            )

        print("=" * 80)


def main():
    """Run ablation studies."""
    parser = argparse.ArgumentParser(description="Phase 4.2: Ablation Studies")
    parser.add_argument(
        "--config",
        type=Path,
        default=ABLATION_DIR / "config.json",
        help="Path to ablation config",
    )
    parser.add_argument(
        "--ablation",
        type=str,
        help="Run specific ablation (if not specified, runs all)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ABLATION_DIR,
        help="Output directory for results",
    )
    args = parser.parse_args()

    # Create ablation directory if needed
    ABLATION_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize study
    study = AblationStudy(args.config)

    if not study.config:
        logger.error("Failed to load ablation config")
        return 1

    # Run ablations
    if args.ablation:
        study.run_ablation(args.ablation)
    else:
        num_successful = study.run_all_ablations()
        logger.info(f"Completed {num_successful} ablations")

    # Save and print results
    study.save_results(args.output_dir)
    study.print_results()

    return 0


if __name__ == "__main__":
    sys.exit(main())
