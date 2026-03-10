"""
Benchmarking Utilities for Code Smell Detection Research

Metrics calculation, statistical tests, profiling, and architecture-specific
quality measurements for comparing LLM-based detection vs baseline tools.

Architecture: Supports Phase 4 evaluation (WBS Sections 4.1-4.3)
Reference: Benchmarking Strategy Sections 1-10
"""

import time
import math
import threading
import logging
import json
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix as sk_confusion_matrix,
    classification_report,
)
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

from config import (
    CODE_SMELL_TYPES,
    METRICS_DIR,
    PERFORMANCE_DIR,
    PERFORMANCE_TARGETS,
    RANDOM_SEED,
    RESOURCE_MONITORING_INTERVAL,
    RESOURCES_DIR,
    RESULTS_DIR,
)

logger = logging.getLogger(__name__)

# Reproducibility
np.random.seed(RANDOM_SEED)


# ============================================================================
# 1. Core Metrics
# ============================================================================


def calculate_metrics(
    y_true: List[str],
    y_pred: List[str],
    average: str = "weighted",
) -> Dict[str, float]:
    """Calculate precision, recall, F1-score, and accuracy.

    Supports both binary and multi-class classification.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        average: Averaging method ('weighted', 'macro', 'micro').

    Returns:
        Dict with precision, recall, f1, accuracy, and support count.
    """
    labels = sorted(set(y_true) | set(y_pred))

    return {
        "precision": float(precision_score(y_true, y_pred, labels=labels, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, labels=labels, average=average, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, labels=labels, average=average, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "support": len(y_true),
        "average": average,
    }


def build_confusion_matrix(
    y_true: List[str],
    y_pred: List[str],
    labels: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Build a labeled confusion matrix as a pandas DataFrame.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        labels: Ordered label list. Auto-detected if None.

    Returns:
        DataFrame with rows=true labels, columns=predicted labels.
    """
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))

    cm = sk_confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(cm, index=labels, columns=labels)


def per_smell_breakdown(
    y_true: List[str],
    y_pred: List[str],
    smell_types: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Per-smell-type precision, recall, F1, and support.

    Args:
        y_true: Ground truth smell labels.
        y_pred: Predicted smell labels.
        smell_types: Smell types to report. Uses all observed if None.

    Returns:
        DataFrame indexed by smell type with P/R/F1/support columns.
    """
    if smell_types is None:
        smell_types = sorted(set(y_true) | set(y_pred))

    report = classification_report(
        y_true, y_pred, labels=smell_types, output_dict=True, zero_division=0,
    )

    rows = []
    for smell in smell_types:
        if smell in report:
            rows.append({
                "smell_type": smell,
                "precision": report[smell]["precision"],
                "recall": report[smell]["recall"],
                "f1": report[smell]["f1-score"],
                "support": int(report[smell]["support"]),
            })

    df = pd.DataFrame(rows).set_index("smell_type")
    df.sort_values("f1", ascending=False, inplace=True)
    return df


# ============================================================================
# 2. Statistical Tests
# ============================================================================


def mcnemars_test(
    y_true: List[int],
    y_pred_a: List[int],
    y_pred_b: List[int],
) -> Dict[str, Any]:
    """McNemar's test for comparing two classifiers on paired data.

    Tests whether two classifiers have the same error rate.

    Args:
        y_true: Ground truth binary labels.
        y_pred_a: Predictions from classifier A.
        y_pred_b: Predictions from classifier B.

    Returns:
        Dict with test statistic, p-value, significance, and discordant counts.
    """
    correct_a = np.array(y_pred_a) == np.array(y_true)
    correct_b = np.array(y_pred_b) == np.array(y_true)

    # Discordant pairs
    b = int(np.sum(correct_a & ~correct_b))   # A right, B wrong
    c = int(np.sum(~correct_a & correct_b))   # A wrong, B right

    if b + c == 0:
        return {
            "statistic": 0.0, "p_value": 1.0, "significant": False,
            "n_discordant": 0, "a_better_count": 0, "b_better_count": 0,
        }

    # Chi-squared with continuity correction
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = 1 - scipy_stats.chi2.cdf(chi2, df=1)

    return {
        "statistic": float(chi2),
        "p_value": float(p_value),
        "significant": p_value < 0.05,
        "n_discordant": b + c,
        "a_better_count": b,
        "b_better_count": c,
    }


def paired_t_test(
    scores_a: List[float],
    scores_b: List[float],
) -> Dict[str, Any]:
    """Paired t-test comparing two sets of F1 scores across runs.

    Args:
        scores_a: F1 scores from system A (one per run).
        scores_b: F1 scores from system B (one per run).

    Returns:
        Dict with t-statistic, p-value, confidence interval, and means.
    """
    a, b = np.array(scores_a), np.array(scores_b)
    t_stat, p_value = scipy_stats.ttest_rel(a, b)

    diff = a - b
    mean_diff = float(np.mean(diff))
    se_diff = float(scipy_stats.sem(diff)) if len(diff) > 1 else 0.0

    # 95% CI for mean difference
    ci_low, ci_high = (0.0, 0.0)
    if len(diff) > 1:
        ci = scipy_stats.t.interval(0.95, df=len(diff) - 1, loc=mean_diff, scale=se_diff)
        ci_low, ci_high = float(ci[0]), float(ci[1])

    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "significant": p_value < 0.05,
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "mean_difference": mean_diff,
        "ci_95_low": ci_low,
        "ci_95_high": ci_high,
    }


def cohens_d(scores_a: List[float], scores_b: List[float]) -> Dict[str, Any]:
    """Cohen's d effect size for practical significance.

    Args:
        scores_a: Scores from system A.
        scores_b: Scores from system B.

    Returns:
        Dict with d value and interpretation.
    """
    a, b = np.array(scores_a), np.array(scores_b)
    n_a, n_b = len(a), len(b)
    var_a = float(np.var(a, ddof=1)) if n_a > 1 else 0.0
    var_b = float(np.var(b, ddof=1)) if n_b > 1 else 0.0

    pooled_std = math.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / max(n_a + n_b - 2, 1))

    d = (float(np.mean(a)) - float(np.mean(b))) / pooled_std if pooled_std > 0 else 0.0

    # Interpretation thresholds (Cohen, 1988)
    abs_d = abs(d)
    if abs_d < 0.2:
        interpretation = "negligible"
    elif abs_d < 0.5:
        interpretation = "small"
    elif abs_d < 0.8:
        interpretation = "medium"
    else:
        interpretation = "large"

    return {"d": float(d), "interpretation": interpretation}


def bootstrap_confidence_interval(
    scores: List[float],
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
) -> Dict[str, float]:
    """Bootstrap 95% confidence interval for a metric.

    Args:
        scores: Sample scores.
        confidence: Confidence level (default 0.95).
        n_bootstrap: Number of bootstrap iterations.

    Returns:
        Dict with mean, ci_low, ci_high.
    """
    scores_arr = np.array(scores)
    boot_means = np.array([
        np.mean(np.random.choice(scores_arr, size=len(scores_arr), replace=True))
        for _ in range(n_bootstrap)
    ])

    alpha = (1 - confidence) / 2
    ci_low = float(np.percentile(boot_means, alpha * 100))
    ci_high = float(np.percentile(boot_means, (1 - alpha) * 100))

    return {
        "mean": float(np.mean(scores_arr)),
        "std": float(np.std(scores_arr, ddof=1)) if len(scores_arr) > 1 else 0.0,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "confidence": confidence,
        "n_bootstrap": n_bootstrap,
    }


def statistical_tests(
    y_true: List[int],
    y_pred_a: List[int],
    y_pred_b: List[int],
    f1_scores_a: Optional[List[float]] = None,
    f1_scores_b: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Run full statistical comparison between two systems.

    Combines McNemar's test, paired t-test (if multi-run scores provided),
    Cohen's d, and confidence intervals.

    Args:
        y_true: Ground truth labels.
        y_pred_a: Predictions from system A.
        y_pred_b: Predictions from system B.
        f1_scores_a: F1 from multiple runs of A (optional).
        f1_scores_b: F1 from multiple runs of B (optional).

    Returns:
        Dict with all statistical test results.
    """
    results = {"mcnemar": mcnemars_test(y_true, y_pred_a, y_pred_b)}

    if f1_scores_a and f1_scores_b:
        results["paired_t_test"] = paired_t_test(f1_scores_a, f1_scores_b)
        results["cohens_d"] = cohens_d(f1_scores_a, f1_scores_b)
        results["ci_a"] = bootstrap_confidence_interval(f1_scores_a)
        results["ci_b"] = bootstrap_confidence_interval(f1_scores_b)

    return results


# ============================================================================
# 3. Latency & Resource Profiling
# ============================================================================


@contextmanager
def latency_profiler(
    component: str,
    tracker: Optional[Dict[str, float]] = None,
):
    """Context manager to measure component execution time.

    Args:
        component: Name of the component being measured.
        tracker: Dict to store timing results. If None, logs only.

    Usage:
        timings = {}
        with latency_profiler("embedding", timings):
            embed(text)
        with latency_profiler("retrieval", timings):
            retrieve(query)
        # timings == {"embedding": 0.5, "retrieval": 1.2}
    """
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start

    if tracker is not None:
        tracker[component] = round(elapsed, 4)

    logger.debug("Latency [%s]: %.4fs", component, elapsed)


@dataclass
class ResourceSnapshot:
    """Single resource usage measurement."""
    timestamp: float
    cpu_percent: float
    memory_mb: float
    memory_percent: float


class ResourceMonitor:
    """Background thread that samples CPU and memory usage.

    Designed for M4 Pro: tracks process-level and system-level resources.
    Requires psutil for full functionality; degrades gracefully without it.

    Usage:
        monitor = ResourceMonitor()
        monitor.start()
        # ... run analysis ...
        profile = monitor.stop()
        print(profile["peak_memory_mb"])
    """

    def __init__(self, interval: float = RESOURCE_MONITORING_INTERVAL):
        self._interval = interval
        self._snapshots: List[ResourceSnapshot] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start background monitoring thread."""
        if not HAS_PSUTIL:
            logger.warning("psutil not installed — resource monitoring disabled")
            return

        self._running = True
        self._snapshots.clear()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        logger.debug("Resource monitor started (interval=%.1fs)", self._interval)

    def stop(self) -> Dict[str, Any]:
        """Stop monitoring and return aggregated profile.

        Returns:
            Dict with peak, mean, and timeline of CPU/memory usage.
        """
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

        if not self._snapshots:
            return {"error": "no data collected (psutil missing or monitor not started)"}

        cpu_values = [s.cpu_percent for s in self._snapshots]
        mem_values = [s.memory_mb for s in self._snapshots]

        profile = {
            "samples": len(self._snapshots),
            "duration_seconds": round(
                self._snapshots[-1].timestamp - self._snapshots[0].timestamp, 2
            ),
            "cpu_mean_percent": round(float(np.mean(cpu_values)), 1),
            "cpu_peak_percent": round(float(np.max(cpu_values)), 1),
            "memory_mean_mb": round(float(np.mean(mem_values)), 1),
            "memory_peak_mb": round(float(np.max(mem_values)), 1),
            "memory_peak_percent": round(
                float(max(s.memory_percent for s in self._snapshots)), 1
            ),
            "within_target": float(np.max(mem_values)) / 1024 <= PERFORMANCE_TARGETS["memory_max_gb"],
        }

        return profile

    def _sample_loop(self) -> None:
        """Internal sampling loop running on background thread."""
        process = psutil.Process()
        while self._running:
            try:
                mem = process.memory_info()
                self._snapshots.append(ResourceSnapshot(
                    timestamp=time.time(),
                    cpu_percent=process.cpu_percent(),
                    memory_mb=mem.rss / (1024 * 1024),
                    memory_percent=process.memory_percent(),
                ))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            time.sleep(self._interval)


def save_latency_profile(
    tool_name: str,
    timings: Dict[str, float],
    output_dir: Optional[Path] = None,
) -> Path:
    """Save latency profile to CSV.

    Args:
        tool_name: Tool or system name.
        timings: Component → seconds mapping.
        output_dir: Output directory (default: PERFORMANCE_DIR).

    Returns:
        Path to saved CSV.
    """
    output_dir = output_dir or PERFORMANCE_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"{tool_name}_timing_{timestamp}.csv"

    df = pd.DataFrame([
        {"component": k, "seconds": v} for k, v in timings.items()
    ])
    df.loc[len(df)] = {"component": "total", "seconds": sum(timings.values())}
    df.to_csv(path, index=False)

    logger.info("Latency profile saved to %s", path)
    return path


def save_resource_profile(
    tool_name: str,
    profile: Dict[str, Any],
    output_dir: Optional[Path] = None,
) -> Path:
    """Save resource profile to JSON.

    Args:
        tool_name: Tool or system name.
        profile: Profile dict from ResourceMonitor.stop().
        output_dir: Output directory (default: RESOURCES_DIR).

    Returns:
        Path to saved JSON.
    """
    output_dir = output_dir or RESOURCES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"{tool_name}_profile_{timestamp}.json"

    with open(path, "w") as f:
        json.dump({"tool": tool_name, "timestamp": timestamp, **profile}, f, indent=2)

    logger.info("Resource profile saved to %s", path)
    return path


# ============================================================================
# 4. Architecture-Specific Metrics (Section 10.1)
# ============================================================================


def calculate_hallucination_rate(
    predictions: List[Dict[str, Any]],
    source_files: Dict[str, List[str]],
) -> Dict[str, Any]:
    """Detect hallucinated predictions by verifying against source code.

    A prediction is considered a hallucination if:
    - It references a file that doesn't exist in the source.
    - It references a line number beyond the file's length.

    Args:
        predictions: List of prediction dicts with 'file' and 'line' keys.
        source_files: Mapping of filename → list of source lines.

    Returns:
        Dict with hallucination rate, count, and details.
    """
    total = len(predictions)
    if total == 0:
        return {"rate": 0.0, "hallucinations": 0, "total": 0, "details": []}

    hallucinations = []
    for pred in predictions:
        filename = pred.get("file", "")
        line = pred.get("line", 0)
        reason = None

        if filename not in source_files:
            reason = f"file '{filename}' not in source"
        elif line > len(source_files[filename]) or line < 1:
            reason = f"line {line} out of range (file has {len(source_files[filename])} lines)"

        if reason:
            hallucinations.append({"prediction": pred, "reason": reason})

    rate = len(hallucinations) / total
    within_target = rate <= PERFORMANCE_TARGETS["hallucination_rate_max"]

    return {
        "rate": round(rate, 4),
        "hallucinations": len(hallucinations),
        "total": total,
        "within_target": within_target,
        "target": PERFORMANCE_TARGETS["hallucination_rate_max"],
        "details": hallucinations[:20],  # Cap details for readability
    }


def cache_hit_rate(hits: int, total: int) -> Dict[str, Any]:
    """Calculate cache hit rate and compare against target.

    Args:
        hits: Number of cache hits.
        total: Total cache lookups.

    Returns:
        Dict with rate, within_target flag.
    """
    rate = hits / total if total > 0 else 0.0
    return {
        "rate": round(rate, 4),
        "hits": hits,
        "misses": total - hits,
        "total": total,
        "within_target": rate >= PERFORMANCE_TARGETS["cache_hit_rate_min"],
        "target": PERFORMANCE_TARGETS["cache_hit_rate_min"],
    }


def validation_failure_rate(total: int, failures: int) -> Dict[str, Any]:
    """Calculate JSON parse / schema validation failure rate.

    Args:
        total: Total LLM responses.
        failures: Number of parse/validation failures.

    Returns:
        Dict with failure rate and success rate.
    """
    rate = failures / total if total > 0 else 0.0
    return {
        "failure_rate": round(rate, 4),
        "success_rate": round(1 - rate, 4),
        "failures": failures,
        "total": total,
    }


def confidence_calibration(
    confidences: List[float],
    correct: List[int],
    n_bins: int = 10,
) -> Dict[str, Any]:
    """Evaluate confidence calibration (predicted confidence vs actual accuracy).

    Calculates Expected Calibration Error (ECE) and bin-level statistics
    for plotting reliability diagrams.

    Args:
        confidences: Predicted confidence scores in [0, 1].
        correct: Binary array (1 if prediction correct, 0 otherwise).
        n_bins: Number of calibration bins.

    Returns:
        Dict with ECE, per-bin data, and correlation coefficient.
    """
    conf = np.array(confidences)
    corr = np.array(correct, dtype=float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bins_data = []

    for i in range(n_bins):
        mask = (conf > bin_boundaries[i]) & (conf <= bin_boundaries[i + 1])
        if mask.sum() == 0:
            continue

        bin_conf = float(conf[mask].mean())
        bin_acc = float(corr[mask].mean())
        count = int(mask.sum())

        ece += (count / len(conf)) * abs(bin_acc - bin_conf)
        bins_data.append({
            "bin_low": round(float(bin_boundaries[i]), 2),
            "bin_high": round(float(bin_boundaries[i + 1]), 2),
            "avg_confidence": round(bin_conf, 4),
            "avg_accuracy": round(bin_acc, 4),
            "count": count,
        })

    # Pearson correlation between confidence and correctness
    r_val = 0.0
    if len(conf) > 2:
        r_val, _ = scipy_stats.pearsonr(conf, corr)

    return {
        "ece": round(float(ece), 4),
        "correlation": round(float(r_val), 4),
        "correlation_target": 0.6,
        "correlation_met": abs(r_val) >= 0.6,
        "n_bins": n_bins,
        "bins": bins_data,
    }


def rag_retrieval_quality(
    relevance_scores: List[float],
    k_values: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Calculate retrieval quality metrics: Precision@k and NDCG@k.

    Args:
        relevance_scores: Relevance scores in retrieval order (1.0=relevant, 0.0=not).
        k_values: Cutoff positions to evaluate. Defaults to [1, 3, 5, 10].

    Returns:
        Dict with Precision@k and NDCG@k for each k.
    """
    if k_values is None:
        k_values = [1, 3, 5, 10]

    scores = np.array(relevance_scores)
    results = {}

    for k in k_values:
        top_k = scores[:k]

        # Precision@k
        precision_k = float(np.sum(top_k > 0)) / k if k > 0 else 0.0

        # NDCG@k
        ndcg_k = _ndcg_at_k(scores, k)

        results[f"precision@{k}"] = round(precision_k, 4)
        results[f"ndcg@{k}"] = round(ndcg_k, 4)

    return results


def _ndcg_at_k(relevance: np.ndarray, k: int) -> float:
    """Compute NDCG@k.

    Args:
        relevance: Full relevance score array in retrieved order.
        k: Cutoff position.

    Returns:
        NDCG@k value in [0, 1].
    """
    rel = relevance[:k]
    if len(rel) == 0:
        return 0.0

    # DCG@k
    positions = np.arange(1, len(rel) + 1)
    dcg = float(np.sum((2 ** rel - 1) / np.log2(positions + 1)))

    # IDCG@k (sort all scores descending, take top k)
    ideal = np.sort(relevance)[::-1][:k]
    idcg = float(np.sum((2 ** ideal - 1) / np.log2(np.arange(1, len(ideal) + 1) + 1)))

    return dcg / idcg if idcg > 0 else 0.0


# ============================================================================
# 5. Aggregate Results & Persistence
# ============================================================================


def compare_tools(
    ground_truth: List[str],
    predictions: Dict[str, List[str]],
    tool_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Compare multiple tools/systems on the same ground truth.

    Args:
        ground_truth: True labels.
        predictions: Mapping tool_name → predicted labels.
        tool_names: Ordered list of tools (default: all keys).

    Returns:
        DataFrame with one row per tool and P/R/F1/Accuracy columns.
    """
    if tool_names is None:
        tool_names = sorted(predictions.keys())

    rows = []
    for name in tool_names:
        if name not in predictions:
            continue
        metrics = calculate_metrics(ground_truth, predictions[name])
        rows.append({"tool": name, **metrics})

    df = pd.DataFrame(rows).set_index("tool")
    df.sort_values("f1", ascending=False, inplace=True)
    return df


def save_metrics(
    metrics: Dict[str, Any],
    name: str,
    output_dir: Optional[Path] = None,
) -> Path:
    """Save metrics dict to JSON file.

    Args:
        metrics: Metrics dictionary.
        name: Descriptive name (used in filename).
        output_dir: Output directory (default: METRICS_DIR).

    Returns:
        Path to saved JSON file.
    """
    output_dir = output_dir or METRICS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"{name}_{timestamp}.json"

    with open(path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    logger.info("Metrics saved to %s", path)
    return path
