"""
Comparison routes for code smell detection API.
Compare LLM results against baseline tools (SonarQube, PMD, etc).
"""

import asyncio
from fastapi import APIRouter, HTTPException
from datetime import datetime, timedelta
from typing import Dict, List
import logging

from src.api.models import (
    ComparisonResponse,
    ComparisonMetrics,
    BaselineToolResult,
    CodeSmellFindingResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# In-memory storage for demonstration (use database in production).
# M6: Mirror the TTL + lock pattern applied to analysis_state (C2) so this
# cache cannot grow unbounded once writes start happening.
comparison_cache: Dict[str, Dict] = {}
comparison_cache_lock = asyncio.Lock()
COMPARISON_CACHE_TTL = timedelta(hours=24)
COMPARISON_CLEANUP_INTERVAL_SECONDS = 3600


def store_comparison(analysis_id: str, data: Dict) -> None:
    """Insert/refresh a comparison entry with a TTL stamp.

    Keep callers simple: they don't need to know about expires_at.
    """
    now = datetime.utcnow()
    data = dict(data)
    data.setdefault("created_at", now)
    data["expires_at"] = now + COMPARISON_CACHE_TTL
    comparison_cache[analysis_id] = data


async def cleanup_expired_comparisons() -> int:
    """Evict entries whose expires_at has passed. Returns eviction count."""
    now = datetime.utcnow()
    async with comparison_cache_lock:
        expired_ids = [
            cid for cid, entry in comparison_cache.items()
            if entry.get("expires_at") and entry["expires_at"] < now
        ]
        for cid in expired_ids:
            comparison_cache.pop(cid, None)
    if expired_ids:
        logger.info("🧹 Evicted %d expired comparison entries", len(expired_ids))
    return len(expired_ids)


async def comparison_cache_cleanup_loop() -> None:
    """Background task: periodically evict expired comparisons."""
    while True:
        try:
            await asyncio.sleep(COMPARISON_CLEANUP_INTERVAL_SECONDS)
            await cleanup_expired_comparisons()
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 - keep loop alive
            logger.warning("comparison_cache cleanup loop error: %s", e)


# ============================================================================
# Helper Functions for Long Method Refactoring
# ============================================================================

def _get_empty_summary() -> Dict:
    """Return empty comparison summary when cache is empty."""
    return {
        "total_comparisons": 0,
        "average_precision": 0.0,
        "average_recall": 0.0,
        "average_f1": 0.0,
        "tool_coverage": {},
        "findings_distribution": {
            "high_agreement": 0,
            "medium_agreement": 0,
            "low_agreement": 0,
            "llm_only": 0,
            "baseline_only": 0,
        },
        "performance": {
            "avg_analysis_time_ms": 0,
            "baseline_avg_time_ms": 0,
            "slowest_tool": None,
            "fastest_tool": None,
        },
    }


def _collect_metrics_from_cache():
    """Extract metrics and counts from comparison cache."""
    precisions = []
    recalls = []
    f1_scores = []
    tool_counts: Dict[str, int] = {}
    total_analysis_time = 0
    baseline_times: Dict[str, float] = {}
    baseline_time_count: Dict[str, int] = {}

    high_agreement = 0
    medium_agreement = 0
    low_agreement = 0
    llm_only_count = 0
    baseline_only_count = 0

    for comp_data in comparison_cache.values():
        metrics = comp_data.get("metrics", {})
        precisions.append(metrics.get("precision", 0.0))
        recalls.append(metrics.get("recall", 0.0))
        f1_scores.append(metrics.get("f1_score", 0.0))

        total_analysis_time += comp_data["llm_results"].get("analysis_time_ms", 0)

        # Count tool coverage
        for tool_name in comp_data.get("baseline_results", {}).keys():
            tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1
            baseline_times[tool_name] = baseline_times.get(tool_name, 0) + comp_data["baseline_results"][
                tool_name
            ].get("analysis_time_ms", 0)
            baseline_time_count[tool_name] = baseline_time_count.get(tool_name, 0) + 1

        # Calculate agreement
        llm_count = len(comp_data["llm_results"].get("findings", []))
        baseline_count = sum(
            len(tool["findings"]) for tool in comp_data.get("baseline_results", {}).values()
        )

        if llm_count > 0 and baseline_count > 0:
            agreement_ratio = min(llm_count, baseline_count) / max(llm_count, baseline_count)
            if agreement_ratio > 0.8:
                high_agreement += 1
            elif agreement_ratio > 0.5:
                medium_agreement += 1
            else:
                low_agreement += 1
        elif llm_count > baseline_count:
            llm_only_count += 1
        else:
            baseline_only_count += 1

    return {
        "precisions": precisions,
        "recalls": recalls,
        "f1_scores": f1_scores,
        "tool_counts": tool_counts,
        "total_analysis_time": total_analysis_time,
        "baseline_times": baseline_times,
        "baseline_time_count": baseline_time_count,
        "agreement": {
            "high": high_agreement,
            "medium": medium_agreement,
            "low": low_agreement,
            "llm_only": llm_only_count,
            "baseline_only": baseline_only_count,
        },
    }


def _format_summary_response(metrics_data: Dict, total: int) -> Dict:
    """Format collected metrics into summary response."""
    precisions = metrics_data["precisions"]
    recalls = metrics_data["recalls"]
    f1_scores = metrics_data["f1_scores"]
    baseline_times = metrics_data["baseline_times"]
    baseline_time_count = metrics_data["baseline_time_count"]

    # Calculate averages
    avg_precision = sum(precisions) / len(precisions) if precisions else 0.0
    avg_recall = sum(recalls) / len(recalls) if recalls else 0.0
    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    avg_analysis_time = metrics_data["total_analysis_time"] / total if total > 0 else 0

    # Calculate baseline averages
    baseline_avg_times: Dict[str, float] = {}
    for tool, total_time in baseline_times.items():
        baseline_avg_times[tool] = (
            total_time / baseline_time_count[tool] if baseline_time_count[tool] > 0 else 0
        )

    slowest_tool = (
        max(baseline_avg_times.items(), key=lambda x: x[1])[0] if baseline_avg_times else None
    )
    fastest_tool = (
        min(baseline_avg_times.items(), key=lambda x: x[1])[0] if baseline_avg_times else None
    )
    baseline_avg = sum(baseline_avg_times.values()) / len(baseline_avg_times) if baseline_avg_times else 0

    return {
        "total_comparisons": total,
        "average_precision": round(avg_precision, 3),
        "average_recall": round(avg_recall, 3),
        "average_f1": round(avg_f1, 3),
        "tool_coverage": metrics_data["tool_counts"],
        "findings_distribution": {
            "high_agreement": metrics_data["agreement"]["high"],
            "medium_agreement": metrics_data["agreement"]["medium"],
            "low_agreement": metrics_data["agreement"]["low"],
            "llm_only": metrics_data["agreement"]["llm_only"],
            "baseline_only": metrics_data["agreement"]["baseline_only"],
        },
        "performance": {
            "avg_analysis_time_ms": round(avg_analysis_time, 1),
            "baseline_avg_time_ms": round(baseline_avg, 1),
            "slowest_tool": slowest_tool,
            "fastest_tool": fastest_tool,
        },
    }


@router.get(
    "/compare/{analysis_id}",
    response_model=ComparisonResponse,
    status_code=200,
    summary="Compare LLM vs baseline tools",
    description="Compare LLM analysis results with results from baseline tools (SonarQube, PMD, etc).",
)
async def compare_analysis(analysis_id: str) -> ComparisonResponse:
    """
    Compare LLM findings against baseline tools for a given analysis.

    Args:
        analysis_id: Unique analysis identifier from /api/v1/analyze

    Returns:
        ComparisonResponse with LLM findings, baseline findings, and metrics

    Raises:
        HTTPException: 404 if analysis not found, 400 if comparison not available

    Example:
        GET /api/v1/compare/abc-123-def
        Response:
        {
            "analysis_id": "abc-123-def",
            "llm_results": {...},
            "baseline_results": {
                "sonarqube": {...},
                "pmd": {...}
            },
            "metrics": {
                "precision": 0.85,
                "recall": 0.90,
                "f1_score": 0.87,
                "accuracy": 0.88,
                "true_positives": 17,
                "false_positives": 3,
                "false_negatives": 2
            },
            "summary": "LLM detected 20 findings, SonarQube detected 22..."
        }
    """
    logger.info(f"Comparing analysis results for {analysis_id}")

    # Check if analysis exists in cache
    if analysis_id not in comparison_cache:
        logger.warning(f"Comparison not found for analysis_id: {analysis_id}")
        raise HTTPException(
            status_code=404,
            detail=f"Comparison data not found for analysis {analysis_id}. Ensure analysis is completed and baseline tools have been run.",
        )

    comparison_data = comparison_cache[analysis_id]

    # M6: lazily drop an entry that has already expired so we don't serve
    # stale data between scheduled cleanup ticks.
    expires_at = comparison_data.get("expires_at")
    if expires_at and expires_at < datetime.utcnow():
        comparison_cache.pop(analysis_id, None)
        raise HTTPException(
            status_code=404,
            detail=f"Comparison data for analysis {analysis_id} has expired.",
        )

    # Build comparison response
    try:
        response = ComparisonResponse(
            analysis_id=analysis_id,
            llm_results=comparison_data["llm_results"],
            baseline_results=comparison_data["baseline_results"],
            metrics=comparison_data["metrics"],
            summary=_generate_comparison_summary(
                comparison_data["llm_results"],
                comparison_data["baseline_results"],
                comparison_data["metrics"],
            ),
        )
        logger.info(f"✓ Comparison retrieved for {analysis_id}")
        return response
    except Exception as e:
        logger.error(f"Error retrieving comparison for {analysis_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve comparison data")


@router.get(
    "/comparison/summary",
    status_code=200,
    summary="Cross-tool summary statistics",
    description="Get aggregated comparison metrics across all completed analyses.",
)
async def comparison_summary():
    """
    Get aggregated comparison statistics across all analyses.

    Returns:
        Dictionary with aggregated metrics and tool coverage
    """
    logger.info("Generating comparison summary")

    if not comparison_cache:
        logger.info("No comparisons available for summary")
        return _get_empty_summary()

    total = len(comparison_cache)
    metrics_data = _collect_metrics_from_cache()
    summary = _format_summary_response(metrics_data, total)

    logger.info(f"✓ Summary generated: {total} comparisons")
    return summary


# ============================================================================
# Helper Functions
# ============================================================================


def _generate_comparison_summary(llm_results, baseline_results, metrics) -> str:
    """
    Generate human-readable comparison summary.

    Args:
        llm_results: LLM analysis results
        baseline_results: Dictionary of baseline tool results
        metrics: Comparison metrics

    Returns:
        Summary string
    """
    llm_findings = len(llm_results.get("findings", []))
    baseline_findings_total = sum(
        len(tool.get("findings", [])) for tool in baseline_results.values()
    )
    tool_names = ", ".join(baseline_results.keys())

    summary = f"LLM detected {llm_findings} findings, baseline tools ({tool_names}) detected {baseline_findings_total} findings. "
    summary += f"Precision: {metrics.get('precision', 0):.2f}, Recall: {metrics.get('recall', 0):.2f}, F1: {metrics.get('f1_score', 0):.2f}. "
    summary += f"True positives: {metrics.get('true_positives', 0)}, False positives: {metrics.get('false_positives', 0)}, "
    summary += f"False negatives: {metrics.get('false_negatives', 0)}."

    return summary
