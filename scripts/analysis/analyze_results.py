"""
Result Analysis and Aggregation (Phase 4.1)

Aggregates results from multiple experiment runs, generates summary tables,
creates visualizations, and exports publication-ready figures/tables.

Architecture: Phase 4 analysis (WBS Section 4.1)
Reference: Benchmarking Strategy Sections 6-7
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    CODE_SMELL_TYPES,
    METRICS_DIR,
    RESULTS_DIR,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

PREDICTIONS_DIR = RESULTS_DIR / "predictions"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
REPORTS_DIR = RESULTS_DIR / "reports"


class ResultAggregator:
    """
    Aggregates multiple experiment runs and generates summary statistics.

    Phase 4.1: Processes Phase 3.3 experiment results, computes mean/std,
    generates benchmark tables (Tables 1-3), and creates publication figures.
    """

    def __init__(self):
        """Initialize result aggregator."""
        self.runs: Dict[str, List[Dict]] = defaultdict(list)  # system -> [runs]
        self.summary: Dict[str, Any] = {}

    def load_metrics_from_directory(self, metrics_dir: Path = METRICS_DIR):
        """Load metrics from evaluation output directory.

        Expected structure:
            metrics_dir/
            ├── {system_name}/
            │   ├── metrics_*.json
            │   ├── per_smell_breakdown_*.csv
            │   └── confusion_matrix_*.csv
            └── summary_report_*.json

        Args:
            metrics_dir: Path to metrics directory.
        """
        if not metrics_dir.exists():
            logger.warning(f"Metrics directory not found: {metrics_dir}")
            return False

        # Load system results
        for system_dir in metrics_dir.iterdir():
            if not system_dir.is_dir() or system_dir.name.startswith("."):
                continue

            system_name = system_dir.name
            metrics_files = list(system_dir.glob("metrics_*.json"))

            for metrics_file in metrics_files:
                try:
                    with open(metrics_file) as f:
                        metrics = json.load(f)
                        self.runs[system_name].append({
                            "timestamp": metrics_file.stem.split("_", 1)[1],
                            "metrics": metrics,
                        })
                except Exception as e:
                    logger.error(f"Failed to load {metrics_file}: {e}")

        logger.info(f"Loaded {sum(len(runs) for runs in self.runs.values())} runs")
        return True

    def load_metrics_from_predictions(self, predictions_dir: Path = PREDICTIONS_DIR):
        """Load metrics from experiment result JSON files in predictions dir.

        Expected format: results/predictions/{system}/{tool_timestamp}.json
        With ExperimentMetrics structure containing "metrics" field.

        Args:
            predictions_dir: Path to predictions directory.
        """
        if not predictions_dir.exists():
            logger.warning(f"Predictions directory not found: {predictions_dir}")
            return

        for system_dir in predictions_dir.iterdir():
            if not system_dir.is_dir():
                continue

            system_name = system_dir.name
            json_files = list(system_dir.glob("*.json"))

            if not json_files:
                logger.info(f"No results in {system_name}")
                continue

            # Use most recent file
            latest_file = max(json_files, key=lambda p: p.stat().st_mtime)

            try:
                with open(latest_file) as f:
                    data = json.load(f)
                    # Handle ExperimentMetrics structure
                    if "metrics" in data:
                        metrics = data["metrics"]
                    elif "precision" in data:
                        metrics = data
                    else:
                        logger.warning(f"Unexpected format in {latest_file}")
                        continue

                    self.runs[system_name].append({
                        "timestamp": latest_file.stem,
                        "metrics": metrics,
                        "quality_metrics": data.get("quality_metrics", {}),
                        "resource_profile": data.get("resource_profile", {}),
                    })
                    logger.info(f"Loaded {system_name}: F1={metrics.get('f1', 0):.4f}")

            except Exception as e:
                logger.error(f"Failed to load {latest_file}: {e}")

    def aggregate_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Aggregate metrics across all runs per system.

        Returns:
            Dict mapping system_name to aggregated metrics (mean, std).
        """
        self.summary = {}

        for system_name, runs in self.runs.items():
            if not runs:
                continue

            metrics_list = [run["metrics"] for run in runs]

            # Extract metric values
            f1_scores = [m.get("f1", 0) for m in metrics_list]
            precisions = [m.get("precision", 0) for m in metrics_list]
            recalls = [m.get("recall", 0) for m in metrics_list]
            accuracies = [m.get("accuracy", 0) for m in metrics_list]

            self.summary[system_name] = {
                "f1": {
                    "mean": float(np.mean(f1_scores)),
                    "std": float(np.std(f1_scores)) if len(f1_scores) > 1 else 0.0,
                    "min": float(np.min(f1_scores)),
                    "max": float(np.max(f1_scores)),
                    "n_runs": len(runs),
                },
                "precision": {
                    "mean": float(np.mean(precisions)),
                    "std": float(np.std(precisions)) if len(precisions) > 1 else 0.0,
                },
                "recall": {
                    "mean": float(np.mean(recalls)),
                    "std": float(np.std(recalls)) if len(recalls) > 1 else 0.0,
                },
                "accuracy": {
                    "mean": float(np.mean(accuracies)),
                    "std": float(np.std(accuracies)) if len(accuracies) > 1 else 0.0,
                },
            }

            # Include quality metrics if available
            if runs[0].get("quality_metrics"):
                quality = runs[0]["quality_metrics"]
                self.summary[system_name]["quality"] = quality

            # Include resource metrics if available
            if runs[0].get("resource_profile"):
                resources = runs[0]["resource_profile"]
                self.summary[system_name]["resources"] = resources

        return self.summary

    def generate_table1_overall_comparison(self) -> pd.DataFrame:
        """Generate Table 1: Overall Performance Comparison (6+ tools).

        Returns:
            DataFrame with systems as rows, metrics as columns.
        """
        rows = []

        for system_name in sorted(self.summary.keys()):
            metrics = self.summary[system_name]
            rows.append({
                "System": system_name,
                "Precision": f"{metrics['precision']['mean']:.4f} ± {metrics['precision']['std']:.4f}",
                "Recall": f"{metrics['recall']['mean']:.4f} ± {metrics['recall']['std']:.4f}",
                "F1-Score": f"{metrics['f1']['mean']:.4f} ± {metrics['f1']['std']:.4f}",
                "Accuracy": f"{metrics['accuracy']['mean']:.4f} ± {metrics['accuracy']['std']:.4f}",
                "Runs": metrics['f1']['n_runs'],
            })

        df = pd.DataFrame(rows)
        df.sort_values("F1-Score", ascending=False, key=lambda x: x.str.split().str[0].astype(float), inplace=True)
        return df

    def generate_table2_per_smell_performance(self, system_name: str = None) -> pd.DataFrame:
        """Generate Table 2: Per-Smell-Type Performance.

        Args:
            system_name: Specific system. If None, averages across all systems.

        Returns:
            DataFrame with smell types as rows, metrics as columns.
        """
        # For now, create a template structure
        # In full implementation, this would load per-smell data from metrics files
        rows = []

        for smell in CODE_SMELL_TYPES:
            rows.append({
                "Code Smell": smell,
                "Precision": "0.0000",  # Placeholder
                "Recall": "0.0000",
                "F1-Score": "0.0000",
                "Support": 0,
            })

        return pd.DataFrame(rows)

    def generate_table3_resource_requirements(self) -> pd.DataFrame:
        """Generate Table 3: Resource Requirements (latency, memory, throughput).

        Returns:
            DataFrame with systems as rows, resource metrics as columns.
        """
        rows = []

        for system_name in sorted(self.summary.keys()):
            metrics = self.summary[system_name]
            resources = metrics.get("resources", {})

            rows.append({
                "System": system_name,
                "Latency (ms)": resources.get("avg_latency_ms", "N/A"),
                "Peak Memory (MB)": resources.get("peak_memory_mb", "N/A"),
                "CPU (%)": resources.get("avg_cpu_percent", "N/A"),
            })

        return pd.DataFrame(rows)

    def save_results(self, output_dir: Path = None) -> Path:
        """Save aggregated results and tables.

        Args:
            output_dir: Output directory. Defaults to combined metrics/tables.

        Returns:
            Path where results were saved.
        """
        output_dir = output_dir or (RESULTS_DIR / "analysis_results")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save aggregated summary
        with open(output_dir / "aggregated_metrics.json", "w") as f:
            json.dump(self.summary, f, indent=2, default=str)

        # Save tables
        table1 = self.generate_table1_overall_comparison()
        table2 = self.generate_table2_per_smell_performance()
        table3 = self.generate_table3_resource_requirements()

        table1.to_csv(output_dir / "table1_overall_comparison.csv", index=False)
        table2.to_csv(output_dir / "table2_per_smell_performance.csv", index=False)
        table3.to_csv(output_dir / "table3_resource_requirements.csv", index=False)

        # Save LaTeX versions
        self._save_latex_tables(output_dir, table1, table2, table3)

        logger.info(f"Results saved to {output_dir}")
        return output_dir

    def _save_latex_tables(
        self,
        output_dir: Path,
        table1: pd.DataFrame,
        table2: pd.DataFrame,
        table3: pd.DataFrame,
    ):
        """Save tables in LaTeX format for paper.

        Args:
            output_dir: Output directory.
            table1-3: DataFrames to export.
        """
        # Table 1: Overall Performance
        latex1 = table1.to_latex(index=False, escape=False)
        latex1 = self._wrap_latex_table(
            latex1,
            caption="Overall Performance Comparison (6+ detection systems)",
            label="tab:overall_comparison",
        )
        with open(output_dir / "table1_overall_comparison.tex", "w") as f:
            f.write(latex1)

        # Table 2: Per-Smell Performance
        latex2 = table2.to_latex(index=False, escape=False)
        latex2 = self._wrap_latex_table(
            latex2,
            caption="Per-Smell-Type Performance Metrics",
            label="tab:per_smell_performance",
        )
        with open(output_dir / "table2_per_smell_performance.tex", "w") as f:
            f.write(latex2)

        # Table 3: Resource Requirements
        latex3 = table3.to_latex(index=False, escape=False)
        latex3 = self._wrap_latex_table(
            latex3,
            caption="Resource Requirements and Performance Characteristics",
            label="tab:resource_requirements",
        )
        with open(output_dir / "table3_resource_requirements.tex", "w") as f:
            f.write(latex3)

        logger.info("LaTeX tables saved")

    def _wrap_latex_table(
        self,
        latex_str: str,
        caption: str,
        label: str,
    ) -> str:
        """Wrap LaTeX table code with proper formatting.

        Args:
            latex_str: Raw LaTeX table code.
            caption: Table caption.
            label: Table label reference.

        Returns:
            Wrapped LaTeX code.
        """
        return f"""
\\begin{{table}}[ht]
\\centering
\\caption{{{caption}}}
\\label{{{label}}}
{latex_str}
\\end{{table}}
"""

    def print_summary(self):
        """Print summary to console."""
        print("\n" + "=" * 70)
        print("AGGREGATED EVALUATION RESULTS")
        print("=" * 70)

        table1 = self.generate_table1_overall_comparison()
        print("\nTable 1: Overall Performance Comparison")
        print(table1.to_string(index=False))

        print("\n" + "=" * 70)
        print("System Rankings by F1-Score:")
        for idx, row in table1.iterrows():
            f1_str = row["F1-Score"].split()[0]
            print(f"  {idx + 1}. {row['System']:20s} F1={f1_str} (n={row['Runs']} runs)")


def main():
    """Run Phase 4.1 result analysis."""
    import argparse

    parser = argparse.ArgumentParser(description="Phase 4.1: Result Analysis & Aggregation")
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        default=PREDICTIONS_DIR,
        help="Path to predictions directory",
    )
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=METRICS_DIR,
        help="Path to metrics directory (from evaluation.py)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR / "analysis_results",
        help="Output directory for aggregated results",
    )
    args = parser.parse_args()

    # Initialize aggregator
    aggregator = ResultAggregator()

    # Try to load from evaluation metrics first
    if args.metrics_dir.exists():
        aggregator.load_metrics_from_directory(args.metrics_dir)

    # Fallback: load from predictions directory
    if not aggregator.runs:
        aggregator.load_metrics_from_predictions(args.predictions_dir)

    if not aggregator.runs:
        logger.error("No metrics found. Run evaluation.py or experiments first.")
        return 1

    # Aggregate
    aggregator.aggregate_metrics()

    # Generate and save results
    aggregator.save_results(args.output_dir)
    aggregator.print_summary()

    return 0


if __name__ == "__main__":
    sys.exit(main())
