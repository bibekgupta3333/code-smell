"""
Result Exporter for Code Smell Detection Research

Export benchmarking results to LaTeX tables, CSV files, and publication-quality
figures (matplotlib/seaborn). All outputs follow academic paper conventions.

Architecture: Supports Phase 5 paper writing (WBS Section 5)
Output dirs: paper/figures/, results/tables/, results/figures/
"""

import logging
import json
import csv
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for scripts
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import (
    CODE_SMELL_TYPES,
    FIGURES_DIR,
    PAPER_DIR,
    PERFORMANCE_TARGETS,
    RESULTS_DIR,
    TABLES_DIR,
)

logger = logging.getLogger(__name__)

# Publication style defaults
PAPER_FIGSIZE = (8, 5)
PAPER_DPI = 300
PAPER_FONT_SIZE = 11

sns.set_theme(
    style="whitegrid",
    font_scale=1.1,
    rc={
        "figure.figsize": PAPER_FIGSIZE,
        "figure.dpi": PAPER_DPI,
        "font.size": PAPER_FONT_SIZE,
        "axes.titlesize": PAPER_FONT_SIZE + 1,
        "axes.labelsize": PAPER_FONT_SIZE,
        "xtick.labelsize": PAPER_FONT_SIZE - 1,
        "ytick.labelsize": PAPER_FONT_SIZE - 1,
    },
)

# Consistent color palette for tools
TOOL_COLORS = {
    "SonarQube": "#4A90D9",
    "PMD": "#E8783A",
    "Checkstyle": "#50B848",
    "SpotBugs": "#E04B4B",
    "IntelliJ": "#9B59B6",
    "LLM_Vanilla": "#F4C542",
    "LLM_RAG": "#2ECC71",
}


def _get_color(tool_name: str) -> str:
    """Return consistent color for a tool name."""
    return TOOL_COLORS.get(tool_name, "#95A5A6")


def _save_figure(fig: plt.Figure, name: str, output_dir: Optional[Path] = None) -> Path:
    """Save figure to both results/figures/ and paper/figures/.

    Args:
        fig: Matplotlib figure.
        name: Filename stem (without extension).
        output_dir: Primary output dir (default: FIGURES_DIR).

    Returns:
        Path to saved figure.
    """
    dirs = [output_dir or FIGURES_DIR]
    paper_fig_dir = PAPER_DIR / "figures"
    if paper_fig_dir.exists():
        dirs.append(paper_fig_dir)

    saved_path = None
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{name}.pdf"
        fig.savefig(path, bbox_inches="tight", dpi=PAPER_DPI)
        # Also save PNG for quick preview
        fig.savefig(d / f"{name}.png", bbox_inches="tight", dpi=150)
        if saved_path is None:
            saved_path = path

    plt.close(fig)
    logger.info("Figure saved: %s", name)
    return saved_path


# ============================================================================
# 1. LaTeX Table Export
# ============================================================================


def to_latex_table(
    df: pd.DataFrame,
    caption: str,
    label: str,
    float_format: str = "%.3f",
    bold_max: bool = True,
    output_dir: Optional[Path] = None,
) -> str:
    """Export DataFrame to a LaTeX table (booktabs style, paper-ready).

    Args:
        df: DataFrame to export.
        caption: Table caption for the paper.
        label: LaTeX label (e.g., "tab:overall_results").
        float_format: Printf-style format for float values.
        bold_max: Bold the best value in each numeric column.
        output_dir: Output directory (default: TABLES_DIR).

    Returns:
        LaTeX string.
    """
    export_df = df.copy()

    # Bold best values per column
    if bold_max:
        for col in export_df.select_dtypes(include=[np.number]).columns:
            max_val = export_df[col].max()
            export_df[col] = export_df[col].apply(
                lambda x, m=max_val, fmt=float_format: (
                    f"\\textbf{{{fmt % x}}}" if x == m else fmt % x
                )
            )

    latex = export_df.to_latex(
        caption=caption,
        label=label,
        escape=False,
        float_format=float_format,
        column_format="l" + "c" * len(export_df.columns),
    )

    # Save to file
    output_dir = output_dir or TABLES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{label.replace('tab:', '')}.tex"
    path.write_text(latex)
    logger.info("LaTeX table saved to %s", path)

    return latex


def export_overall_comparison(
    comparison_df: pd.DataFrame,
    output_dir: Optional[Path] = None,
) -> str:
    """Export Table 1: Overall performance comparison.

    Args:
        comparison_df: DataFrame from benchmark_utils.compare_tools().
        output_dir: Output directory.

    Returns:
        LaTeX string.
    """
    cols = ["precision", "recall", "f1", "accuracy"]
    display_df = comparison_df[[c for c in cols if c in comparison_df.columns]].copy()
    display_df.columns = [c.capitalize() for c in display_df.columns]

    return to_latex_table(
        display_df,
        caption="Overall performance comparison across all code smell types.",
        label="tab:overall_comparison",
        output_dir=output_dir,
    )


def export_per_smell_table(
    breakdown_df: pd.DataFrame,
    output_dir: Optional[Path] = None,
) -> str:
    """Export Table 2: Per-smell-type performance with delta-F1.

    Args:
        breakdown_df: DataFrame from benchmark_utils.per_smell_breakdown().
        output_dir: Output directory.

    Returns:
        LaTeX string.
    """
    display_df = breakdown_df[["precision", "recall", "f1", "support"]].copy()
    display_df.columns = ["Precision", "Recall", "F1", "Support"]

    return to_latex_table(
        display_df,
        caption="Per-smell-type detection performance (F1-score breakdown).",
        label="tab:per_smell",
        output_dir=output_dir,
    )


# ============================================================================
# 2. CSV Export
# ============================================================================


def to_csv(
    data: pd.DataFrame,
    name: str,
    output_dir: Optional[Path] = None,
) -> Path:
    """Export DataFrame to CSV.

    Args:
        data: DataFrame to export.
        name: Filename stem.
        output_dir: Output directory (default: TABLES_DIR).

    Returns:
        Path to saved CSV.
    """
    output_dir = output_dir or TABLES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}.csv"
    data.to_csv(path)
    logger.info("CSV saved to %s", path)
    return path


def export_predictions_csv(
    predictions: List[Dict[str, Any]],
    tool_name: str,
    output_dir: Optional[Path] = None,
) -> Path:
    """Export normalized predictions to CSV.

    Args:
        predictions: List of normalized prediction dicts.
        tool_name: Tool that generated predictions.
        output_dir: Output directory.

    Returns:
        Path to saved CSV.
    """
    output_dir = output_dir or RESULTS_DIR / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{tool_name}_predictions.csv"

    df = pd.DataFrame(predictions)
    df.to_csv(path, index=False)
    logger.info("Predictions CSV saved to %s", path)
    return path


# ============================================================================
# 3. Visualization — Publication-Quality Figures
# ============================================================================


def plot_f1_comparison(
    comparison_df: pd.DataFrame,
    title: str = "F1-Score Comparison Across Tools",
    output_name: str = "f1_comparison",
) -> Path:
    """Bar chart comparing F1-scores of all tools (Figure 1 in paper).

    Args:
        comparison_df: DataFrame with tool names as index and 'f1' column.
        title: Plot title.
        output_name: Output filename stem.

    Returns:
        Path to saved figure.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    tools = comparison_df.index.tolist()
    colors = [_get_color(t) for t in tools]
    bars = ax.bar(tools, comparison_df["f1"], color=colors, edgecolor="white", width=0.6)

    # Value labels on bars
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005, f"{h:.3f}",
                ha="center", va="bottom", fontsize=PAPER_FONT_SIZE - 1)

    ax.set_ylabel("F1-Score")
    ax.set_title(title)
    ax.set_ylim(0, min(1.05, comparison_df["f1"].max() + 0.1))
    ax.axhline(y=0.8, color="gray", linestyle="--", alpha=0.5, label="Target (0.80)")
    ax.legend(loc="upper right")

    return _save_figure(fig, output_name)


def plot_per_smell_heatmap(
    results: Dict[str, pd.DataFrame],
    metric: str = "f1",
    output_name: str = "per_smell_heatmap",
) -> Path:
    """Heatmap of per-smell F1 across tools (Figure 2 in paper).

    Args:
        results: Mapping tool_name → per-smell breakdown DataFrame.
        metric: Metric column to plot ('f1', 'precision', 'recall').
        output_name: Output filename stem.

    Returns:
        Path to saved figure.
    """
    # Build matrix: rows=smell types, columns=tools
    all_smells = set()
    for df in results.values():
        all_smells.update(df.index.tolist())
    smells = sorted(all_smells)
    tools = sorted(results.keys())

    matrix = pd.DataFrame(index=smells, columns=tools, dtype=float)
    for tool, df in results.items():
        for smell in smells:
            if smell in df.index:
                matrix.loc[smell, tool] = df.loc[smell, metric]
    matrix = matrix.fillna(0)

    fig, ax = plt.subplots(figsize=(max(8, len(tools) * 1.5), max(6, len(smells) * 0.5)))
    sns.heatmap(
        matrix.astype(float), annot=True, fmt=".2f", cmap="RdYlGn",
        vmin=0, vmax=1, linewidths=0.5, ax=ax, cbar_kws={"label": metric.upper()},
    )
    ax.set_title(f"Per-Smell {metric.upper()} Across Tools")
    ax.set_ylabel("Code Smell Type")
    ax.set_xlabel("Tool")
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)

    return _save_figure(fig, output_name)


def plot_confusion_matrix(
    cm_df: pd.DataFrame,
    title: str = "Confusion Matrix",
    output_name: str = "confusion_matrix",
) -> Path:
    """Annotated confusion matrix heatmap.

    Args:
        cm_df: Confusion matrix as DataFrame (from benchmark_utils.build_confusion_matrix).
        title: Plot title.
        output_name: Output filename stem.

    Returns:
        Path to saved figure.
    """
    n = len(cm_df)
    fig, ax = plt.subplots(figsize=(max(8, n * 0.8), max(6, n * 0.7)))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", linewidths=0.5, ax=ax)
    ax.set_title(title)
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    return _save_figure(fig, output_name)


def plot_latency_comparison(
    latency_data: Dict[str, Dict[str, float]],
    output_name: str = "latency_comparison",
) -> Path:
    """Stacked bar chart of component latencies per tool.

    Args:
        latency_data: Mapping tool → {component: seconds}.
        output_name: Output filename stem.

    Returns:
        Path to saved figure.
    """
    df = pd.DataFrame(latency_data).T.fillna(0)

    fig, ax = plt.subplots(figsize=(10, 6))
    df.plot(kind="bar", stacked=True, ax=ax, edgecolor="white")
    ax.set_ylabel("Latency (seconds)")
    ax.set_title("Component Latency Breakdown")
    ax.axhline(
        y=PERFORMANCE_TARGETS["latency_max_seconds"],
        color="red", linestyle="--", alpha=0.7, label=f"Target ({PERFORMANCE_TARGETS['latency_max_seconds']}s)",
    )
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.xticks(rotation=30, ha="right")

    return _save_figure(fig, output_name)


def plot_confidence_calibration(
    calibration_data: Dict[str, Any],
    tool_name: str = "LLM",
    output_name: str = "confidence_calibration",
) -> Path:
    """Reliability diagram for confidence calibration.

    Args:
        calibration_data: Dict from benchmark_utils.confidence_calibration().
        tool_name: Tool name for the title.
        output_name: Output filename stem.

    Returns:
        Path to saved figure.
    """
    bins = calibration_data.get("bins", [])
    if not bins:
        logger.warning("No calibration bins to plot")
        return FIGURES_DIR / f"{output_name}.pdf"

    fig, ax = plt.subplots(figsize=(7, 7))

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")

    # Bin data
    confs = [b["avg_confidence"] for b in bins]
    accs = [b["avg_accuracy"] for b in bins]
    counts = [b["count"] for b in bins]

    ax.bar(confs, accs, width=0.08, alpha=0.6, color="#2ECC71", edgecolor="white", label="Observed")
    ax.scatter(confs, accs, color="#E74C3C", zorder=5, s=40)

    ax.set_xlabel("Mean Predicted Confidence")
    ax.set_ylabel("Fraction of Correct Predictions")
    ax.set_title(f"Confidence Calibration — {tool_name} (ECE={calibration_data.get('ece', 0):.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()

    return _save_figure(fig, output_name)


def plot_resource_usage(
    profiles: Dict[str, Dict[str, Any]],
    output_name: str = "resource_usage",
) -> Path:
    """Grouped bar chart of resource usage across tools.

    Args:
        profiles: Mapping tool → resource profile dict.
        output_name: Output filename stem.

    Returns:
        Path to saved figure.
    """
    tools = sorted(profiles.keys())
    cpu_vals = [profiles[t].get("cpu_mean_percent", 0) for t in tools]
    mem_vals = [profiles[t].get("memory_peak_mb", 0) for t in tools]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    colors = [_get_color(t) for t in tools]

    ax1.bar(tools, cpu_vals, color=colors, edgecolor="white")
    ax1.set_ylabel("CPU Usage (%)")
    ax1.set_title("Mean CPU Usage")
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha="right")

    ax2.bar(tools, mem_vals, color=colors, edgecolor="white")
    ax2.set_ylabel("Peak Memory (MB)")
    ax2.set_title("Peak Memory Usage")
    ax2.axhline(
        y=PERFORMANCE_TARGETS["memory_max_gb"] * 1024,
        color="red", linestyle="--", alpha=0.7, label=f"Target ({PERFORMANCE_TARGETS['memory_max_gb']} GB)",
    )
    ax2.legend()
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right")

    fig.suptitle("Resource Usage Comparison", fontsize=PAPER_FONT_SIZE + 2)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    return _save_figure(fig, output_name)


def plot_delta_f1(
    vanilla_f1: Dict[str, float],
    rag_f1: Dict[str, float],
    output_name: str = "delta_f1_rag",
) -> Path:
    """Bar chart showing F1 improvement from RAG enhancement.

    Args:
        vanilla_f1: Per-smell F1 from vanilla LLM.
        rag_f1: Per-smell F1 from RAG-enhanced LLM.
        output_name: Output filename stem.

    Returns:
        Path to saved figure.
    """
    smells = sorted(set(vanilla_f1.keys()) & set(rag_f1.keys()))
    deltas = [rag_f1[s] - vanilla_f1[s] for s in smells]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#2ECC71" if d >= 0 else "#E74C3C" for d in deltas]
    bars = ax.barh(smells, deltas, color=colors, edgecolor="white")

    ax.set_xlabel("ΔF1 (RAG − Vanilla)")
    ax.set_title("F1 Improvement with RAG Enhancement")
    ax.axvline(x=0, color="black", linewidth=0.8)

    for bar, delta in zip(bars, deltas):
        x = bar.get_width()
        ax.text(x + 0.005 * (1 if x >= 0 else -1), bar.get_y() + bar.get_height() / 2,
                f"{delta:+.3f}", va="center", fontsize=PAPER_FONT_SIZE - 2)

    return _save_figure(fig, output_name)
