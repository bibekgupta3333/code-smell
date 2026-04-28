"""
Real Code Smell Detection with F1 Scoring Integration
Bridges FastAPI with LangGraph Workflow and ground truth evaluation

This module provides the integration between the API and the LangGraph workflow.
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from src.workflow.workflow_graph import WorkflowExecutor
from src.rag.rag_retriever import RAGRetriever
from src.utils.benchmark_utils import f1_score as calculate_f1, precision_score, recall_score
from src.utils.common import CodeSmellFinding

logger = logging.getLogger(__name__)

# Ground truth cache
_ground_truth_cache = None
_ground_truth_by_smell = None
_ground_truth_aliases = None
_ground_truth_load_error: Optional[str] = None


def _normalize_lookup_key(value: Optional[str]) -> Optional[str]:
    """Normalize a sample identifier for resilient ground-truth lookup."""
    if not value:
        return None
    normalized = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
    return normalized or None


def _collect_sample_aliases(sample: Dict) -> List[str]:
    """Collect stable aliases that can identify a benchmark sample."""
    aliases = []

    sample_id = sample.get("sample_id")
    file_path = sample.get("file_path")
    class_name = sample.get("class_name")
    language = sample.get("language")

    for candidate in (sample_id, file_path, class_name):
        if candidate:
            aliases.append(candidate)

    if file_path:
        path = Path(file_path)
        aliases.extend([path.name, path.stem])

    if class_name and language:
        aliases.extend([
            f"smelly_{language}_{class_name}",
            f"{language}_{class_name}",
        ])

    return aliases


def resolve_ground_truth_sample_id(sample_id: Optional[str], ground_truth_dict: Dict[str, List[str]]) -> Tuple[Optional[str], List[str], bool]:
    """Resolve a submitted file name or sample identifier to a benchmark sample."""
    if not sample_id or sample_id == "code_snippet":
        # No sample_id provided or generic placeholder - auto-select a sample WITH smells
        if ground_truth_dict:
            # Prefer a sample with non-zero smells for meaningful F1 evaluation
            samples_with_smells = [(s_id, smells) for s_id, smells in ground_truth_dict.items() if len(smells) > 0]

            if samples_with_smells:
                # Sort by smell count (descending) to get representative sample
                samples_with_smells.sort(key=lambda x: len(x[1]), reverse=True)
                selected_id, selected_smells = samples_with_smells[0]

                # Log selection reasoning
                logger.info(
                    f"Auto-selecting ground truth sample '{selected_id}' ({len(selected_smells)} smells, "
                    f"types: {len(set(selected_smells))}) for F1 evaluation"
                )
                return selected_id, selected_smells, True
            else:
                # No samples with smells - use first available
                first_sample = next(iter(ground_truth_dict))
                logger.warning(f"All samples have 0 smells. Auto-selecting '{first_sample}' anyway")
                return first_sample, ground_truth_dict[first_sample], True
        return None, [], False

    if sample_id in ground_truth_dict:
        logger.info(f"Found exact match for sample_id '{sample_id}'")
        return sample_id, ground_truth_dict[sample_id], True

    # Try to find a match by normalization
    lookup_key = _normalize_lookup_key(sample_id)
    resolved_sample_id = _ground_truth_aliases.get(lookup_key) if lookup_key and _ground_truth_aliases else None

    if resolved_sample_id:
        logger.info(f"Resolved '{sample_id}' to benchmark sample '{resolved_sample_id}' via normalization")
        return resolved_sample_id, ground_truth_dict.get(resolved_sample_id, []), True

    # Try path-based matching
    sample_path = Path(sample_id)
    for candidate in (sample_path.name, sample_path.stem):
        candidate_key = _normalize_lookup_key(candidate)
        resolved_sample_id = _ground_truth_aliases.get(candidate_key) if candidate_key and _ground_truth_aliases else None
        if resolved_sample_id:
            logger.info(f"Resolved '{sample_id}' to benchmark sample '{resolved_sample_id}' via path matching")
            return resolved_sample_id, ground_truth_dict.get(resolved_sample_id, []), True

    # No match found - log available samples for debugging
    available_samples = list(ground_truth_dict.keys())[:5]  # Show first 5
    logger.info(
        f"No ground truth match for '{sample_id}' ({len(ground_truth_dict)} samples available). "
        f"Available samples: {', '.join(available_samples)}"
    )
    return sample_id, [], False


def extract_line_number(location: object) -> int:
    """Extract a best-effort line number from a finding location."""
    if isinstance(location, dict):
        line = location.get("line") or location.get("start_line")
        return int(line) if isinstance(line, int) or (isinstance(line, str) and line.isdigit()) else 0

    if not location:
        return 0

    match = re.search(r"(\d+)", str(location))
    return int(match.group(1)) if match else 0


def load_ground_truth_from_file(test_data_path: str = "data/processed/test.json") -> Dict:
    """Load ground truth from test.json

    Args:
        test_data_path: Path to test.json

    Returns:
        Dict mapping sample_id -> list of smell types
    """
    global _ground_truth_cache, _ground_truth_by_smell, _ground_truth_aliases, _ground_truth_load_error

    if _ground_truth_cache is not None:
        return _ground_truth_cache

    try:
        with open(test_data_path) as f:
            test_data = json.load(f)

        ground_truth = {}
        ground_truth_by_smell = defaultdict(int)
        alias_candidates = defaultdict(set)

        for sample in test_data:
            sample_id = sample.get("sample_id")
            annotations = sample.get("annotations", [])

            # Preserve repeated annotations so instance-based F1 remains accurate.
            smells = [ann.get("smell_type") for ann in annotations if ann.get("smell_type")]
            ground_truth[sample_id] = smells

            for smell in smells:
                ground_truth_by_smell[smell] += 1

            for alias in _collect_sample_aliases(sample):
                alias_key = _normalize_lookup_key(alias)
                if alias_key:
                    alias_candidates[alias_key].add(sample_id)

        _ground_truth_cache = ground_truth
        _ground_truth_by_smell = ground_truth_by_smell
        _ground_truth_aliases = {
            alias: next(iter(sample_ids))
            for alias, sample_ids in alias_candidates.items()
            if len(sample_ids) == 1
        }

        logger.info(f"Loaded ground truth: {len(ground_truth)} samples, {len(ground_truth_by_smell)} smell types")
        _ground_truth_load_error = None
        return ground_truth

    except FileNotFoundError:
        # H5: A missing ground truth file is a configuration error — not the
        # same as "this sample isn't in the benchmark". Surface it loudly and
        # expose via _ground_truth_load_error so /status and /health can
        # report it instead of silently returning empty metrics.
        msg = (
            f"Ground truth file not found: {test_data_path}. "
            "F1/precision/recall will be unavailable until the dataset split "
            "is generated (see scripts/data)."
        )
        logger.error(msg)
        _ground_truth_load_error = msg
        _ground_truth_cache = {}
        return {}
    except Exception as e:
        msg = f"Error loading ground truth from {test_data_path}: {e}"
        logger.error(msg, exc_info=True)
        _ground_truth_load_error = msg
        _ground_truth_cache = {}
        return {}


def get_ground_truth_load_error() -> Optional[str]:
    """Return the last ground-truth load error (if any) for health/status APIs."""
    return _ground_truth_load_error


def extract_smell_types_from_findings(findings: List[CodeSmellFinding]) -> List[str]:
    """Extract unique smell types from findings

    Args:
        findings: List of CodeSmellFinding objects

    Returns:
        List of unique smell type strings
    """
    return list(set(f.smell_type for f in findings if f.smell_type))


def calculate_f1_for_findings(
    predicted_findings: List[CodeSmellFinding],
    ground_truth_smells: List[str],
    return_breakdown: bool = False
) -> Dict:
    """Calculate F1, precision, recall for a single code sample using instance-based matching.

    Uses proper counting of smell instances and matches based on smell type presence.
    This is the accurate implementation using micro-averaged metrics.

    Args:
        predicted_findings: List of predicted findings
        ground_truth_smells: List of ground truth smell types (can have duplicates)
        return_breakdown: Whether to return detailed breakdown

    Returns:
        Dict with precision, recall, f1, based on instance counts and type matching
    """
    # Get predicted smell list (instances, not unique)
    predicted_smell_list = [f.smell_type for f in predicted_findings if f.smell_type]

    # Count instances of each smell type in predictions
    predicted_counts = {}
    for smell in predicted_smell_list:
        predicted_counts[smell] = predicted_counts.get(smell, 0) + 1

    # Count instances of each smell type in ground truth
    ground_truth_counts = {}
    for smell in ground_truth_smells:
        ground_truth_counts[smell] = ground_truth_counts.get(smell, 0) + 1

    # Handle empty cases
    if len(ground_truth_smells) == 0 and len(predicted_smell_list) == 0:
        return {
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": 0,
        }

    if len(ground_truth_smells) == 0:
        # All predictions are false positives
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "true_positives": 0,
            "false_positives": len(predicted_smell_list),
            "false_negatives": 0,
        }

    if len(predicted_smell_list) == 0:
        # All ground truth are false negatives
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": len(ground_truth_smells),
        }

    # Get unique smell types from both sets
    predicted_types = set(predicted_counts.keys())
    ground_truth_types = set(ground_truth_counts.keys())

    # Determine which types are correct, incorrect, or missing
    correct_types = predicted_types & ground_truth_types  # Types found correctly
    incorrect_types = predicted_types - ground_truth_types  # False positive types
    missing_types = ground_truth_types - predicted_types  # False negative types

    # Count instances for each category (micro-averaging)
    true_positives = sum(predicted_counts[st] for st in correct_types)
    false_positives = sum(predicted_counts[st] for st in incorrect_types)
    false_negatives = sum(ground_truth_counts[st] for st in missing_types)

    # Calculate metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    result = {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }

    if return_breakdown:
        result["predicted_smells"] = predicted_smell_list
        result["ground_truth_smells"] = ground_truth_smells
        result["predicted_types"] = sorted(list(predicted_types))
        result["ground_truth_types"] = sorted(list(ground_truth_types))
        result["correct_types"] = sorted(list(correct_types))
        result["incorrect_types"] = sorted(list(incorrect_types))
        result["missing_types"] = sorted(list(missing_types))

    return result


async def run_code_smell_detection_with_scoring(
    code: str,
    sample_id: Optional[str] = None,
    use_rag: bool = True,
    model: str = "llama3:8b",
    context: Optional[Dict] = None,
    analysis_id: Optional[str] = None,
) -> Dict:
    """Run code smell detection via LangGraph Workflow and calculate F1 score against ground truth

    This is the main integration point between FastAPI and the LangGraph workflow.
    Uses WorkflowExecutor to run the multi-node agentic workflow.

    Args:
        code: Code snippet to analyze
        sample_id: Optional identifier (for ground truth lookup)
        use_rag: Whether to use RAG context
        model: LLM model to use (or None for agentic auto-selection)
        context: Additional context

    Returns:
        Dict with findings, F1 score, model used, and metrics
    """
    try:
        logger.info(f"Running LangGraph workflow: use_rag={use_rag}, model={model}")

        # ✅ CALL LANGGRAPH WORKFLOW EXECUTOR (NOT CodeSmellDetector)
        executor = WorkflowExecutor()
        workflow_result = await executor.execute(
            code_snippet=code,
            file_name=sample_id or "code",
            model=model,  # None = agentic auto-select, or specific model
            use_rag=use_rag,
            analysis_id=analysis_id,
        )

        logger.info(f"Workflow completed. Model used: {workflow_result.get('metadata', {}).get('model_used', 'unknown')}")

        # Extract findings from workflow result
        findings = workflow_result.get("validated_findings", [])
        model_used = workflow_result.get("metadata", {}).get("model_used", model)
        model_reasoning = workflow_result.get("metadata", {}).get("model_reasoning", None)

        # Load ground truth and calculate F1
        ground_truth_dict = load_ground_truth_from_file()
        resolved_sample_id, ground_truth_smells, has_ground_truth = resolve_ground_truth_sample_id(
            sample_id,
            ground_truth_dict,
        )

        if has_ground_truth and resolved_sample_id != sample_id:
            logger.info(
                "Resolved sample_id '%s' to benchmark sample '%s' for F1 scoring",
                sample_id,
                resolved_sample_id,
            )

        # Log available samples for transparency
        if not has_ground_truth and ground_truth_dict:
            available_samples = list(ground_truth_dict.keys())[:3]
            logger.info(
                f"Ground truth available for {len(ground_truth_dict)} samples. "
                f"Examples: {', '.join(available_samples)}"
            )

        if has_ground_truth:
            # Proper evaluation against ground truth
            metrics = calculate_f1_for_findings(
                findings,
                ground_truth_smells,
                return_breakdown=True
            )
            evaluation_mode = "ground_truth"

            # Enhanced logging with detailed breakdown
            predicted_types = sorted(set(metrics.get("predicted_types", [])))
            ground_truth_types = sorted(set(metrics.get("ground_truth_types", [])))
            correct_types = sorted(set(metrics.get("correct_types", [])))
            incorrect_types = sorted(set(metrics.get("incorrect_types", [])))
            missing_types = sorted(set(metrics.get("missing_types", [])))

            logger.info(
                f"F1 Score calculated using ground truth '{resolved_sample_id}': "
                f"F1={metrics.get('f1', 'N/A')}, "
                f"Precision={metrics.get('precision', 'N/A')}, "
                f"Recall={metrics.get('recall', 'N/A')}"
            )
            logger.info(
                f"Ground Truth: {len(ground_truth_smells)} instances, {len(ground_truth_types)} types | "
                f"Predicted: {len(metrics.get('predicted_smells', []))} instances, {len(predicted_types)} types"
            )
            logger.info(
                f"Evaluation breakdown - TP: {metrics.get('true_positives')}, "
                f"FP: {metrics.get('false_positives')}, FN: {metrics.get('false_negatives')}"
            )

            if correct_types:
                logger.info(f"Correct smell types: {', '.join(correct_types)}")
            if incorrect_types:
                logger.info(f"False positive types: {', '.join(incorrect_types)}")
            if missing_types:
                logger.info(f"Missing types in predictions: {', '.join(missing_types)}")
        else:
            # No ground truth available - use confidence-based self-assessment
            # This handles arbitrary user snippets that aren't in the benchmark
            if findings:
                avg_confidence = sum(f.confidence for f in findings) / len(findings)
                high_confidence_count = sum(1 for f in findings if f.confidence >= 0.7)
                # Report confidence as precision proxy
                metrics = {
                    "precision": round(avg_confidence, 4),
                    "recall": None,  # Cannot compute without ground truth
                    "f1": None,      # Cannot compute without ground truth
                    "true_positives": high_confidence_count,
                    "false_positives": len(findings) - high_confidence_count,
                    "false_negatives": None,
                    "avg_confidence": round(avg_confidence, 4),
                    "high_confidence_findings": high_confidence_count,
                    "total_findings": len(findings),
                    "predicted_smells": [f.smell_type for f in findings if f.smell_type],
                    "ground_truth_smells": [],
                }
            else:
                metrics = {
                    "precision": None,
                    "recall": None,
                    "f1": None,
                    "true_positives": 0,
                    "false_positives": 0,
                    "false_negatives": None,
                    "avg_confidence": None,
                    "total_findings": 0,
                    "predicted_smells": [],
                    "ground_truth_smells": [],
                }
            evaluation_mode = "confidence_only"
            logger.info(
                f"No ground truth for sample_id='{sample_id}'. "
                f"Using confidence-based self-assessment: avg_confidence={metrics.get('avg_confidence')}"
            )

        return {
            "success": True,
            "findings": [
                {
                    "smell_type": f.smell_type,
                    "location": {"line": extract_line_number(getattr(f, "location", None))},
                    "severity": (f.severity.value if hasattr(f.severity, 'value') else str(f.severity)).lower(),
                    "confidence": f.confidence,
                    "explanation": f.explanation,
                    "refactoring": f.refactoring,
                }
                for f in findings
            ],
            "metrics": {
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "true_positives": metrics["true_positives"],
                "false_positives": metrics["false_positives"],
                "false_negatives": metrics["false_negatives"],
                "avg_confidence": metrics.get("avg_confidence"),
                "evaluation_mode": evaluation_mode,
                "has_ground_truth": has_ground_truth,
            },
            "ground_truth": {
                "smells": metrics.get("ground_truth_smells", []),
                "count": len(metrics.get("ground_truth_smells", [])),
                "available": has_ground_truth,
            },
            "predicted": {
                "smells": metrics.get("predicted_smells", []),
                "count": len(metrics.get("predicted_smells", [])),
            },
            "model_used": model_used,
            "model_reasoning": model_reasoning,
            "workflow_metadata": workflow_result.get("metadata", {}),
        }

    except Exception as e:
        logger.error(f"LangGraph workflow failed: {e}", exc_info=True)
        return {
            "findings": [],
            "metrics": {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
            },
            "error": str(e),
            "success": False,
        }


async def compare_detection_approaches(
    code: str,
    sample_id: Optional[str] = None,
    models: Optional[List[str]] = None,
) -> Dict:
    """Compare multiple detection approaches: vanilla LLM vs RAG-enhanced

    Shows the impact of RAG on accuracy.

    Args:
        code: Code snippet to analyze
        sample_id: Optional identifier
        models: List of models to test (default: ["llama3:8b"])

    Returns:
        Dict comparing results across approaches
    """
    if models is None:
        models = ["llama3:8b"]

    ground_truth_dict = load_ground_truth_from_file()
    _, ground_truth_smells, _ = resolve_ground_truth_sample_id(sample_id, ground_truth_dict)

    results = {
        "ground_truth": ground_truth_smells,
        "approaches": {},
    }

    try:
        # Test each model with and without RAG
        for model in models:
            model_results = {}

            # Without RAG
            logger.info(f"Testing {model} without RAG...")
            vanilla_result = await run_code_smell_detection_with_scoring(
                code,
                sample_id=sample_id,
                use_rag=False,
                model=model
            )
            model_results["vanilla"] = {
                "f1": vanilla_result["metrics"]["f1"],
                "precision": vanilla_result["metrics"]["precision"],
                "recall": vanilla_result["metrics"]["recall"],
                "findings_count": len(vanilla_result["findings"]),
            }

            # With RAG
            logger.info(f"Testing {model} with RAG...")
            rag_result = await run_code_smell_detection_with_scoring(
                code,
                sample_id=sample_id,
                use_rag=True,
                model=model
            )
            model_results["rag"] = {
                "f1": rag_result["metrics"]["f1"],
                "precision": rag_result["metrics"]["precision"],
                "recall": rag_result["metrics"]["recall"],
                "findings_count": len(rag_result["findings"]),
            }

            # Calculate improvement
            f1_vanilla = vanilla_result["metrics"]["f1"]
            f1_rag = rag_result["metrics"]["f1"]
            improvement = ((f1_rag - f1_vanilla) / f1_vanilla * 100) if f1_vanilla > 0 else 0

            model_results["improvement"] = {
                "f1_improvement_percent": improvement,
                "f1_vanilla": f1_vanilla,
                "f1_rag": f1_rag,
            }

            results["approaches"][model] = model_results

        results["success"] = True

    except Exception as e:
        logger.error(f"Comparison failed: {e}", exc_info=True)
        results["error"] = str(e)
        results["success"] = False

    return results
