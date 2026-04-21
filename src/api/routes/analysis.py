"""
Code Analysis Routes
API endpoints for submitting code for analysis and retrieving results.

Endpoints:
  POST   /api/v1/analyze             - Submit code for analysis
  GET    /api/v1/results/{id}        - Retrieve analysis results
  GET    /api/v1/progress/{id}       - Get real-time progress
"""

import asyncio
import hashlib
import logging
import uuid
from datetime import datetime
from typing import Dict, Optional

from fastapi import APIRouter, HTTPException, BackgroundTasks, Query

from src.api.models import (
    CodeSubmissionRequest,
    CodeSubmissionResponse,
    AnalysisResultResponse,
    CodeSmellFindingResponse,
    CodeSmellLocation,
    CodeMetricsResponse,
    ProgressResponse,
)
from src.api.detection_integration import (
    run_code_smell_detection_with_scoring,
    compare_detection_approaches,
)
from src.database.database_manager import DatabaseManager
from src.workflow.workflow_graph import AnalysisState

logger = logging.getLogger(__name__)

router = APIRouter()

# Global state for tracking analyses (in-memory for now)
# In production, use Redis or database
analysis_state: Dict[str, Dict] = {}


# ============================================================================
# Helper Functions
# ============================================================================

def generate_analysis_id() -> str:
    """Generate unique analysis ID."""
    return str(uuid.uuid4())


def compute_code_hash(code: str) -> str:
    """Compute SHA256 hash of code snippet."""
    return hashlib.sha256(code.encode()).hexdigest()


async def run_analysis_task(
    analysis_id: str,
    code: str,
    language: Optional[str],
    file_name: str,
    include_rag: bool,
    model: Optional[str] = None,
) -> None:
    """Background task to run code analysis with LangGraph workflow and F1 scoring.

    Args:
        analysis_id: Unique identifier for this analysis
        code: Code snippet to analyze
        language: Programming language
        file_name: Source file name
        include_rag: Whether to use RAG context
        model: Optional LLM model to use (if None, LangGraph agent auto-selects)
    """
    try:
        # Update state
        analysis_state[analysis_id]["status"] = "processing"
        analysis_state[analysis_id]["start_time"] = datetime.utcnow()

        logger.info(f"Starting LangGraph analysis {analysis_id} (RAG={include_rag}, model={model or 'auto-select'})")

        # ✅ CALL LANGGRAPH WORKFLOW VIA DETECTION INTEGRATION
        detection_result = await run_code_smell_detection_with_scoring(
            code=code,
            sample_id=file_name,
            use_rag=include_rag,
            model=model,  # Pass specified model, or None for agentic auto-selection
        )

        if not detection_result["success"]:
            raise Exception(detection_result.get("error", "LangGraph workflow failed"))

        # Convert findings to response format
        findings = [
            CodeSmellFindingResponse(
                smell_type=f["smell_type"],
                location=CodeSmellLocation(line=f["location"]["line"], column=0),
                severity=f["severity"],
                confidence=f["confidence"],
                explanation=f["explanation"],
                suggested_refactoring=f["refactoring"]
            )
            for f in detection_result["findings"]
        ]

        metrics = CodeMetricsResponse(
            lines_of_code=len(code.split("\n")),
            cyclomatic_complexity=None,
            halstead_complexity=None,
            maintainability_index=None
        )

        # ✅ USE ACTUAL MODEL RETURNED FROM WORKFLOW (including auto-selected)
        actual_model = detection_result.get("model_used", model or "llama3:8b")
        model_reasoning = detection_result.get("model_reasoning", None)

        result = AnalysisResultResponse(
            analysis_id=analysis_id,
            code_hash=compute_code_hash(code),
            language=language or "python",
            findings=findings,
            metrics=metrics,
            analysis_time_ms=(datetime.utcnow() - analysis_state[analysis_id]["start_time"]).total_seconds() * 1000,
            model_used=actual_model,  # ✅ From LangGraph workflow
            model_reasoning=model_reasoning,  # ✅ From agentic selection
            cache_hit=False,
            completed_at=datetime.utcnow(),
        )

        # Store results in state
        analysis_state[analysis_id]["status"] = "completed"
        analysis_state[analysis_id]["result"] = result
        analysis_state[analysis_id]["metrics"] = detection_result["metrics"]
        analysis_state[analysis_id]["ground_truth"] = detection_result["ground_truth"]
        analysis_state[analysis_id]["model_used"] = actual_model
        analysis_state[analysis_id]["model_reasoning"] = model_reasoning

        logger.info(
            f"LangGraph analysis {analysis_id} complete: {len(findings)} findings, "
            f"F1={detection_result['metrics']['f1']:.3f}, model={actual_model}"
        )

    except Exception as e:
        logger.error(f"LangGraph analysis {analysis_id} failed: {str(e)}", exc_info=True)
        analysis_state[analysis_id]["status"] = "failed"
        analysis_state[analysis_id]["error"] = str(e)


# ============================================================================
# Endpoints
# ============================================================================

@router.post(
    "/analyze",
    response_model=CodeSubmissionResponse,
    summary="Submit Code for Analysis",
    tags=["analysis"],
    status_code=202,
)
async def submit_code_for_analysis(
    request: CodeSubmissionRequest,
    background_tasks: BackgroundTasks,
) -> CodeSubmissionResponse:
    """
    Submit code snippet for code smell detection analysis with optional model selection.

    - **code**: Code to analyze (required)
    - **language**: Programming language (auto-detected if not provided)
    - **file_name**: Source file name for context
    - **include_rag**: Enable RAG-based context retrieval (default: true)
    - **timeout_seconds**: Analysis timeout (default: 300)
    - **model**: Optional specific LLM model from Ollama (if None, agent auto-selects)

    Returns analysis_id for tracking progress and retrieving results.

    Example:
    ```json
    {
        "code": "public class VeryLongMethod { ... }",
        "language": "java",
        "file_name": "VeryLongMethod.java",
        "include_rag": true,
        "model": "llama3:8b",
        "timeout_seconds": 300
    }
    ```
    """
    try:
        # Validate code
        if not request.code or len(request.code.strip()) < 10:
            raise HTTPException(
                status_code=400,
                detail="Code snippet must be at least 10 characters"
            )

        # Generate analysis ID
        analysis_id = generate_analysis_id()

        # Initialize tracking state
        analysis_state[analysis_id] = {
            "status": "queued",
            "created_at": datetime.utcnow(),
            "code_length": len(request.code),
            "language": request.language,
            "file_name": request.file_name,
            "include_rag": request.include_rag,
            "requested_model": request.model,  # Optional model selection from request
        }

        # Add background task for analysis
        background_tasks.add_task(
            run_analysis_task,
            analysis_id=analysis_id,
            code=request.code,
            language=request.language,
            file_name=request.file_name,
            include_rag=request.include_rag,
            model=request.model,  # Pass requested model
        )

        logger.info(
            f"Code analysis submitted: {analysis_id} "
            f"(language={request.language}, size={len(request.code)}, "
            f"model={request.model or 'auto-select'})"
        )

        return CodeSubmissionResponse(
            analysis_id=analysis_id,
            status="queued",
            created_at=datetime.utcnow(),
            estimated_completion_time=None,
            message=f"Analysis queued with ID: {analysis_id}"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting code: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to submit code for analysis"
        )


@router.get(
    "/results/{analysis_id}",
    response_model=AnalysisResultResponse,
    summary="Get Analysis Results",
    tags=["analysis"],
)
async def get_analysis_results(
    analysis_id: str,
) -> AnalysisResultResponse:
    """
    Retrieve analysis results by analysis ID.

    - **analysis_id**: Unique analysis identifier (from /analyze response)

    Returns:
    - findings: List of detected code smells
    - metrics: Code metrics (complexity, LOC, etc.)
    - model_used: LLM model that performed analysis
    - cache_hit: Whether result was from cache

    HTTP Status:
    - 200: Results ready
    - 202: Still processing (try again later)
    - 404: Analysis not found or expired
    - 500: Analysis failed
    """
    try:
        # Check if analysis exists
        if analysis_id not in analysis_state:
            raise HTTPException(
                status_code=404,
                detail=f"Analysis '{analysis_id}' not found"
            )

        state = analysis_state[analysis_id]

        # Handle different states
        if state["status"] == "queued" or state["status"] == "processing":
            raise HTTPException(
                status_code=202,
                detail="Analysis still processing. Check /progress endpoint."
            )

        if state["status"] == "failed":
            raise HTTPException(
                status_code=500,
                detail=f"Analysis failed: {state.get('error', 'Unknown error')}"
            )

        if state["status"] != "completed":
            raise HTTPException(
                status_code=400,
                detail=f"Invalid analysis state: {state['status']}"
            )

        # Return results
        result = state.get("result")
        if not result:
            raise HTTPException(
                status_code=500,
                detail="Results not available"
            )

        # ✅ ADD F1 METRICS TO RESPONSE
        metrics = state.get("metrics", {})
        result.f1_score = metrics.get("f1", 0.0)
        result.precision = metrics.get("precision", 0.0)
        result.recall = metrics.get("recall", 0.0)
        result.ground_truth_count = state.get("ground_truth", {}).get("count", 0)

        logger.info(
            f"Retrieved results for analysis {analysis_id}: "
            f"F1={metrics.get('f1', 0.0):.2f}"
        )
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving results: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve analysis results"
        )


@router.get(
    "/models",
    response_model=dict,
    summary="Get Available LLM Models",
    tags=["analysis"],
)
async def get_available_models() -> dict:
    """
    Get list of available LLM models from Ollama for agentic model selection.

    Returns:
    - models: List of available model names
    - default_model: Recommended default model
    - model_info: Details about each model (optional)

    This endpoint allows clients to:
    1. Display available models in UI for manual selection
    2. Understand which models are available for auto-selection
    3. Provide hints about model capabilities

    Example response:
    ```json
    {
        "models": ["llama3:8b", "mistral:7b", "codellama:13b"],
        "default_model": "llama3:8b",
        "agentic_selection": "Enabled - agent will choose best model based on code characteristics"
    }
    ```
    """
    try:
        from src.llm.llm_client import OllamaClient

        client = OllamaClient()
        available_models = client.get_available_models()

        logger.info(f"Available models: {available_models}")

        return {
            "models": available_models,
            "default_model": "llama3:8b",
            "count": len(available_models),
            "agentic_selection": "Enabled - agent will select best model based on code size, language, and specialization",
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Error fetching available models: {str(e)}", exc_info=True)
        # Return fallback models
        return {
            "models": ["llama3:8b", "mistral:7b", "codellama:13b"],
            "default_model": "llama3:8b",
            "count": 3,
            "agentic_selection": "Enabled - using fallback models",
            "warning": "Could not connect to Ollama, using fallback models",
            "timestamp": datetime.utcnow().isoformat()
        }


@router.get(
    "/progress/{analysis_id}",
    response_model=ProgressResponse,
    summary="Get Analysis Progress",
    tags=["analysis"],
)
async def get_analysis_progress(
    analysis_id: str,
) -> ProgressResponse:
    """
    Get real-time progress of an ongoing analysis.

    - **analysis_id**: Unique analysis identifier

    Returns:
    - status: Current step (parsing, rag_retrieval, inference, validation)
    - percentage: Progress percentage (0-100)
    - current_step: Human-readable step description
    - estimated_remaining_ms: Estimated time remaining

    Use this endpoint to poll for progress updates.

    Example response:
    ```json
    {
        "analysis_id": "abc-123-def",
        "status": "inference",
        "percentage": 65,
        "current_step": "Running LLM inference...",
        "estimated_remaining_ms": 15000
    }
    ```
    """
    try:
        # Check if analysis exists
        if analysis_id not in analysis_state:
            raise HTTPException(
                status_code=404,
                detail=f"Analysis '{analysis_id}' not found"
            )

        state = analysis_state[analysis_id]
        status = state["status"]

        # Map status to progress
        progress_map = {
            "queued": (5, "Queued for processing"),
            "parsing": (15, "Parsing code..."),
            "rag_retrieval": (35, "Retrieving similar examples..."),
            "inference": (65, "Running LLM inference..."),
            "validation": (85, "Validating results..."),
            "completed": (100, "Analysis complete"),
            "failed": (100, "Analysis failed"),
        }

        percentage, step_desc = progress_map.get(
            status,
            (0, "Unknown status")
        )

        # Estimate remaining time
        if status == "completed" or status == "failed":
            estimated_remaining_ms = 0
        else:
            elapsed_ms = (
                datetime.utcnow() - state["created_at"]
            ).total_seconds() * 1000
            # Rough estimate: assume linear progress
            total_estimated_ms = 5000 if status == "queued" else 10000
            estimated_remaining_ms = max(0, int(total_estimated_ms - elapsed_ms))

        logger.info(f"Returned progress for analysis {analysis_id}: {percentage}%")

        return ProgressResponse(
            analysis_id=analysis_id,
            status=status,
            percentage=percentage,
            current_step=step_desc,
            estimated_remaining_ms=estimated_remaining_ms if estimated_remaining_ms > 0 else None,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting progress: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve analysis progress"
        )


# ============================================================================
# Comparison Endpoints (RAG Impact Analysis)
# ============================================================================

async def compare_rag_impact_task(
    comparison_id: str,
    code: str,
    file_name: str,
) -> None:
    """Background task to compare RAG impact."""
    try:
        analysis_state[comparison_id]["status"] = "processing"
        logger.info(f"Starting RAG comparison {comparison_id}")

        # Run comparison
        comparison_result = await compare_detection_approaches(
            code=code,
            sample_id=file_name,
            models=["llama3:8b"],
        )

        # Store results
        analysis_state[comparison_id]["status"] = "completed"
        analysis_state[comparison_id]["result"] = comparison_result

        logger.info(f"RAG comparison {comparison_id} complete")

    except Exception as e:
        logger.error(f"Comparison {comparison_id} failed: {e}")
        analysis_state[comparison_id]["status"] = "failed"
        analysis_state[comparison_id]["error"] = str(e)


@router.post(
    "/compare-rag-impact",
    response_model=CodeSubmissionResponse,
    summary="Compare Detection with and without RAG",
    tags=["comparison"],
    status_code=202,
)
async def compare_rag_impact(
    request: CodeSubmissionRequest,
    background_tasks: BackgroundTasks,
) -> CodeSubmissionResponse:
    """Compare code smell detection with and without RAG context.

    Shows the impact of retrieval-augmented generation on accuracy.

    - **code**: Code snippet to analyze

    Returns comparison_id for tracking and retrieving results.
    """
    comparison_id = generate_analysis_id()
    analysis_state[comparison_id] = {
        "status": "queued",
        "type": "comparison",
        "created_at": datetime.utcnow(),
    }

    background_tasks.add_task(
        compare_rag_impact_task,
        comparison_id,
        request.code,
        request.file_name,
    )

    return CodeSubmissionResponse(
        analysis_id=comparison_id,
        status="queued",
        created_at=datetime.utcnow(),
        estimated_completion_time=None,
        message=f"RAG comparison queued with ID: {comparison_id}",
    )


# ============================================================================
# Utility Endpoints
# ============================================================================

@router.delete(
    "/results/{analysis_id}",
    summary="Delete Analysis Results",
    tags=["analysis"],
)
async def delete_analysis_results(analysis_id: str) -> Dict[str, str]:
    """
    Delete analysis results (cleanup).

    - **analysis_id**: Unique analysis identifier

    Returns confirmation of deletion.
    """
    try:
        if analysis_id not in analysis_state:
            raise HTTPException(
                status_code=404,
                detail=f"Analysis '{analysis_id}' not found"
            )

        del analysis_state[analysis_id]

        logger.info(f"Deleted analysis results: {analysis_id}")
        return {"message": f"Analysis results deleted: {analysis_id}"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting results: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to delete analysis results"
        )


@router.get(
    "/analyses/active",
    summary="List Active Analyses",
    tags=["analysis"],
)
async def list_active_analyses(
    limit: int = Query(10, ge=1, le=100, description="Maximum results to return"),
) -> Dict[str, list]:
    """
    List active and recent analyses.

    - **limit**: Maximum number of results (1-100, default: 10)

    Returns list of analysis IDs with their status.
    """
    try:
        # Sort by creation time, most recent first
        sorted_analyses = sorted(
            analysis_state.items(),
            key=lambda x: x[1]["created_at"],
            reverse=True
        )[:limit]

        analyses = [
            {
                "analysis_id": analysis_id,
                "status": state["status"],
                "created_at": state["created_at"],
                "language": state.get("language"),
                "file_name": state.get("file_name"),
            }
            for analysis_id, state in sorted_analyses
        ]

        logger.info(f"Listed {len(analyses)} active analyses")
        return {"analyses": analyses}

    except Exception as e:
        logger.error(f"Error listing analyses: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to list analyses"
        )
