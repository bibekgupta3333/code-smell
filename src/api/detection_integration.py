"""
Real Code Smell Detection with F1 Scoring Integration
Bridges FastAPI with LangGraph Workflow and ground truth evaluation

This module provides the integration between the API and the LangGraph workflow.
"""

import json
import logging
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


def load_ground_truth_from_file(test_data_path: str = "data/processed/test.json") -> Dict:
    """Load ground truth from test.json

    Args:
        test_data_path: Path to test.json

    Returns:
        Dict mapping sample_id -> list of smell types
    """
    global _ground_truth_cache, _ground_truth_by_smell

    if _ground_truth_cache is not None:
        return _ground_truth_cache

    try:
        with open(test_data_path) as f:
            test_data = json.load(f)

        ground_truth = {}
        ground_truth_by_smell = defaultdict(int)

        for sample in test_data:
            sample_id = sample.get("sample_id")
            annotations = sample.get("annotations", [])

            # Extract unique smell types
            smells = list(set(ann.get("smell_type") for ann in annotations if ann.get("smell_type")))
            ground_truth[sample_id] = smells

            for smell in smells:
                ground_truth_by_smell[smell] += 1

        _ground_truth_cache = ground_truth
        _ground_truth_by_smell = ground_truth_by_smell

        logger.info(f"Loaded ground truth: {len(ground_truth)} samples, {len(ground_truth_by_smell)} smell types")
        return ground_truth

    except FileNotFoundError:
        logger.error(f"Ground truth file not found: {test_data_path}")
        return {}
    except Exception as e:
        logger.error(f"Error loading ground truth: {e}")
        return {}


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
    """Calculate F1, precision, recall for a single code sample

    Args:
        predicted_findings: List of predicted findings
        ground_truth_smells: List of ground truth smell types
        return_breakdown: Whether to return detailed breakdown

    Returns:
        Dict with precision, recall, f1, and optionally breakdown
    """
    predicted_smells = extract_smell_types_from_findings(predicted_findings)

    # Handle empty cases
    if not ground_truth_smells and not predicted_smells:
        return {
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": 0,
        }

    if not ground_truth_smells:
        # All predictions are false positives
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "true_positives": 0,
            "false_positives": len(predicted_smells),
            "false_negatives": 0,
        }

    if not predicted_smells:
        # All ground truth are false negatives
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": len(ground_truth_smells),
        }

    # Calculate metrics
    true_positives = len(set(predicted_smells) & set(ground_truth_smells))
    false_positives = len(set(predicted_smells) - set(ground_truth_smells))
    false_negatives = len(set(ground_truth_smells) - set(predicted_smells))

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    result = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }

    if return_breakdown:
        result["predicted_smells"] = predicted_smells
        result["ground_truth_smells"] = ground_truth_smells
        result["correct_predictions"] = list(set(predicted_smells) & set(ground_truth_smells))

    return result


async def run_code_smell_detection_with_scoring(
    code: str,
    sample_id: Optional[str] = None,
    use_rag: bool = True,
    model: str = "llama3:8b",
    context: Optional[Dict] = None,
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
            use_rag=use_rag
        )

        logger.info(f"Workflow completed. Model used: {workflow_result.get('metadata', {}).get('model_used', 'unknown')}")

        # Extract findings from workflow result
        findings = workflow_result.get("validated_findings", [])
        model_used = workflow_result.get("metadata", {}).get("model_used", model)
        model_reasoning = workflow_result.get("metadata", {}).get("model_reasoning", None)

        # Load ground truth and calculate F1
        ground_truth_dict = load_ground_truth_from_file()
        ground_truth_smells = ground_truth_dict.get(sample_id, []) if sample_id else []

        metrics = calculate_f1_for_findings(
            findings,
            ground_truth_smells,
            return_breakdown=True
        )

        return {
            "success": True,
            "findings": [
                {
                    "smell_type": f.smell_type,
                    "location": {"line": f.location.line if hasattr(f, 'location') else 0},
                    "severity": f.severity.value if hasattr(f.severity, 'value') else str(f.severity),
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
            },
            "ground_truth": {
                "smells": metrics.get("ground_truth_smells", []),
                "count": len(metrics.get("ground_truth_smells", [])),
            },
            "predicted": {
                "smells": metrics.get("predicted_smells", []),
                "count": len(metrics.get("predicted_smells", [])),
            },
            "model_used": model_used,
            "model_reasoning": model_reasoning,
            "workflow_metadata": workflow_result.get("metadata", {}),
            "success": True,
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
    ground_truth_smells = ground_truth_dict.get(sample_id, []) if sample_id else []

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
