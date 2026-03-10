"""
Generate Baseline Tool Report

Reads baseline tool outputs and generates:
1. Summary statistics table
2. Detailed findings breakdown
3. Visualizations (heatmap, bar chart, pie chart)
4. CSV exports for paper writing
5. LaTeX tables for publication

Usage:
    python scripts/generate_baseline_report.py --output results/reports/
    python scripts/generate_baseline_report.py --language python --tool pylint
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PREDICTIONS_DIR = PROJECT_ROOT / "results" / "predictions" / "baseline"
REPORTS_DIR = PROJECT_ROOT / "results" / "reports"


def load_baseline_results(language: Optional[str] = None, tool: Optional[str] = None) -> Dict[str, Any]:
    """Load all baseline results from JSON files.

    Args:
        language: Filter by language (java, python, javascript). None = all.
        tool: Filter by tool name. None = all.

    Returns:
        Dict with structure: {language: {tool: {data}}}
    """
    all_results = {}

    if not PREDICTIONS_DIR.exists():
        logger.error("Predictions directory not found: %s", PREDICTIONS_DIR)
        return all_results

    # Only match files with the naming convention: language_tool.json
    valid_prefixes = {"java", "python", "javascript"}
    for json_file in PREDICTIONS_DIR.glob("*_*.json"):
        # Skip files that don't match our naming convention (e.g., timestamped raw outputs)
        stem = json_file.stem
        prefix = stem.split("_", 1)[0]
        if prefix not in valid_prefixes:
            continue
        try:
            with open(json_file) as f:
                data = json.load(f)

            lang = data.get("language", "unknown")
            tool_name = data.get("tool", "unknown")

            # Filter if specified
            if language and lang != language:
                continue
            if tool and tool_name != tool:
                continue

            if lang not in all_results:
                all_results[lang] = {}

            all_results[lang][tool_name] = data
            logger.info("Loaded %s/%s: %d findings", lang, tool_name, data.get("total_findings", 0))

        except json.JSONDecodeError as e:
            logger.warning("Failed to parse %s: %s", json_file.name, e)

    return all_results


def generate_summary_table(all_results: Dict[str, Any]) -> pd.DataFrame:
    """Generate summary statistics table."""
    rows = []

    for lang, tools in all_results.items():
        for tool_name, data in tools.items():
            findings = data.get("findings", [])
            elapsed = data.get("elapsed_seconds", 0)

            # Categorize by smell type
            smell_types = {}
            severities = {}

            for finding in findings:
                smell = finding.get("smell_type", "Unknown")
                severity = finding.get("severity", "UNKNOWN")

                smell_types[smell] = smell_types.get(smell, 0) + 1
                severities[severity] = severities.get(severity, 0) + 1

            rows.append({
                "Language": lang.upper(),
                "Tool": tool_name.upper(),
                "Total Findings": len(findings),
                "Unique Smells": len(smell_types),
                "Time (s)": round(elapsed, 2),
                "CRITICAL": severities.get("CRITICAL", 0),
                "HIGH": severities.get("HIGH", 0),
                "MEDIUM": severities.get("MEDIUM", 0),
                "LOW": severities.get("LOW", 0),
            })

    return pd.DataFrame(rows)


def generate_smell_breakdown(all_results: Dict[str, Any]) -> pd.DataFrame:
    """Generate code smell type breakdown."""
    smell_counts = {}

    for lang, tools in all_results.items():
        for tool_name, data in tools.items():
            for finding in data.get("findings", []):
                smell = finding.get("smell_type", "Unknown")
                key = f"{lang}_{tool_name}"

                if smell not in smell_counts:
                    smell_counts[smell] = {}

                smell_counts[smell][key] = smell_counts[smell].get(key, 0) + 1

    return pd.DataFrame(smell_counts).fillna(0).astype(int)


def plot_tool_comparison(summary_df: pd.DataFrame, output_dir: Path):
    """Generate bar chart: findings per tool."""
    if summary_df.empty:
        logger.warning("No data for tool comparison plot")
        return

    plt.figure(figsize=(12, 6))
    df_sorted = summary_df.sort_values("Total Findings", ascending=False)
    tool_labels = df_sorted["Language"] + "_" + df_sorted["Tool"]

    plt.bar(range(len(df_sorted)), df_sorted["Total Findings"], color="steelblue")
    plt.xticks(range(len(df_sorted)), tool_labels, rotation=45, ha="right")
    plt.ylabel("Total Findings", fontsize=12)
    plt.title("Baseline Tool Findings Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()

    output_path = output_dir / "baseline_tool_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info("Saved: %s", output_path)
    plt.close()


def plot_smell_distribution(summary_df: pd.DataFrame, output_dir: Path):
    """Generate stacked bar chart: severity distribution."""
    if summary_df.empty:
        logger.warning("No data for severity distribution plot")
        return

    severity_cols = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    severity_data = summary_df[severity_cols].fillna(0)
    tool_labels = summary_df["Language"] + "_" + summary_df["Tool"]

    fig, ax = plt.subplots(figsize=(12, 6))

    severity_data.plot(
        kind="bar",
        stacked=True,
        ax=ax,
        color=["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4"],
    )

    ax.set_xticklabels(tool_labels, rotation=45, ha="right")
    ax.set_ylabel("Number of Findings", fontsize=12)
    ax.set_title("Findings by Severity Level", fontsize=14, fontweight="bold")
    ax.legend(title="Severity", loc="upper right")
    plt.tight_layout()

    output_path = output_dir / "baseline_severity_distribution.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info("Saved: %s", output_path)
    plt.close()


def plot_smell_types_heatmap(smell_df: pd.DataFrame, output_dir: Path):
    """Generate heatmap: smell types vs tools."""
    if smell_df.empty:
        logger.warning("No data for smell types heatmap")
        return

    plt.figure(figsize=(14, 8))
    sns.heatmap(
        smell_df,
        annot=True,
        fmt="d",
        cmap="YlOrRd",
        cbar_kws={"label": "Occurrences"},
        linewidths=0.5,
    )
    plt.title("Code Smell Types: Tool Comparison", fontsize=14, fontweight="bold")
    plt.xlabel("Tool (Language)", fontsize=12)
    plt.ylabel("Smell Type", fontsize=12)
    plt.tight_layout()

    output_path = output_dir / "baseline_smell_heatmap.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info("Saved: %s", output_path)
    plt.close()


def export_to_csv(summary_df: pd.DataFrame, smell_df: pd.DataFrame, output_dir: Path):
    """Export dataframes to CSV."""
    summary_path = output_dir / "baseline_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info("Saved: %s", summary_path)

    if not smell_df.empty:
        smell_path = output_dir / "baseline_smell_breakdown.csv"
        smell_df.to_csv(smell_path)
        logger.info("Saved: %s", smell_path)


def export_to_latex(summary_df: pd.DataFrame, output_dir: Path):
    """Export summary table to LaTeX."""
    if summary_df.empty:
        logger.warning("No data for LaTeX export")
        return

    # Select columns for LaTeX
    latex_cols = ["Language", "Tool", "Total Findings", "Unique Smells", "CRITICAL", "HIGH", "MEDIUM"]
    latex_df = summary_df[latex_cols].copy()

    latex_code = latex_df.to_latex(
        index=False,
        float_format="{:.0f}".format,
        caption="Baseline Tool Findings Summary",
        label="tab:baseline_summary",
        column_format="lllrrrr",
    )

    output_path = output_dir / "baseline_summary.tex"
    with open(output_path, "w") as f:
        f.write(latex_code)

    logger.info("Saved: %s", output_path)


def generate_text_report(all_results: Dict[str, Any], summary_df: pd.DataFrame, output_dir: Path):
    """Generate human-readable text report."""
    output_path = output_dir / "baseline_report.txt"

    with open(output_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("BASELINE TOOL ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")

        # Summary statistics
        f.write("SUMMARY STATISTICS\n")
        f.write("-" * 80 + "\n")
        total_findings = summary_df["Total Findings"].sum()
        total_tools = len(summary_df)
        f.write(f"Total Tools Analyzed: {total_tools}\n")
        f.write(f"Total Findings: {total_findings}\n")
        f.write(f"Average Findings per Tool: {total_findings / total_tools:.1f}\n\n")

        # Tool breakdown
        f.write("FINDINGS BY TOOL\n")
        f.write("-" * 80 + "\n")
        for _, row in summary_df.iterrows():
            f.write(f"{row['Language']}/{row['Tool']}: {int(row['Total Findings'])} findings ")
            f.write(f"(Critical: {int(row['CRITICAL'])}, High: {int(row['HIGH'])}, ")
            f.write(f"Medium: {int(row['MEDIUM'])}, Low: {int(row['LOW'])})\n")
        f.write("\n")

        # Most common smells
        f.write("MOST COMMON CODE SMELL TYPES\n")
        f.write("-" * 80 + "\n")
        all_smells = {}
        for lang, tools in all_results.items():
            for tool_name, data in tools.items():
                for finding in data.get("findings", []):
                    smell = finding.get("smell_type", "Unknown")
                    all_smells[smell] = all_smells.get(smell, 0) + 1

        sorted_smells = sorted(all_smells.items(), key=lambda x: x[1], reverse=True)
        for i, (smell, count) in enumerate(sorted_smells[:10], 1):
            f.write(f"{i:2d}. {smell}: {count} occurrences\n")
        f.write("\n")

        # Findings by severity
        f.write("FINDINGS BY SEVERITY\n")
        f.write("-" * 80 + "\n")
        for severity in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
            total = int(summary_df[severity].sum())
            f.write(f"{severity}: {total} findings\n")

    logger.info("Saved: %s", output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Generate baseline tool analysis report with visualizations and exports."
    )
    parser.add_argument(
        "--language", "-l",
        choices=["java", "python", "javascript"],
        default=None,
        help="Filter by language.",
    )
    parser.add_argument(
        "--tool", "-t",
        default=None,
        help="Filter by tool name.",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=REPORTS_DIR,
        help="Output directory for report.",
    )

    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    # Load results
    logger.info("Loading baseline results...")
    all_results = load_baseline_results(language=args.language, tool=args.tool)

    if not all_results:
        logger.error("No baseline results found!")
        return

    # Generate tables
    logger.info("Generating summary statistics...")
    summary_df = generate_summary_table(all_results)

    logger.info("Generating smell breakdown...")
    smell_df = generate_smell_breakdown(all_results)

    # Generate visualizations
    logger.info("Creating visualizations...")
    plot_tool_comparison(summary_df, args.output)
    plot_smell_distribution(summary_df, args.output)
    if not smell_df.empty:
        plot_smell_types_heatmap(smell_df, args.output)

    # Export data
    logger.info("Exporting data...")
    export_to_csv(summary_df, smell_df, args.output)
    export_to_latex(summary_df, args.output)

    # Generate text report
    generate_text_report(all_results, summary_df, args.output)

    # Display summary
    logger.info("\n" + "=" * 80)
    logger.info("BASELINE SUMMARY")
    logger.info("=" * 80)
    print(summary_df.to_string(index=False))
    logger.info("\nReport saved to: %s", args.output)


if __name__ == "__main__":
    main()
