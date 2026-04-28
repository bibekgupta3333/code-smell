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
from datetime import datetime, timedelta
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
from src.workflow.workflow_graph import (
    AnalysisState,
    register_progress_callback,
    unregister_progress_callback,
)
from src.utils.colored_logger import (
    log_inference_start,
    log_inference_end,
    log_inference_error,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Global state for tracking analyses (in-memory for now)
# In production, use Redis or database
analysis_state: Dict[str, Dict] = {}

# Async lock guarding concurrent writes to analysis_state.
analysis_state_lock = asyncio.Lock()

# Default time-to-live for finished/failed analyses before they are evicted.
ANALYSIS_STATE_TTL = timedelta(hours=24)

# How often the background cleanup loop runs.
ANALYSIS_CLEANUP_INTERVAL_SECONDS = 3600


async def cleanup_expired_analyses() -> int:
    """Remove analysis_state entries whose expires_at is in the past.

    Returns the number of evicted entries.
    """
    now = datetime.utcnow()
    async with analysis_state_lock:
        expired_ids = [
            aid for aid, state in analysis_state.items()
            if state.get("expires_at") and state["expires_at"] < now
        ]
        for aid in expired_ids:
            analysis_state.pop(aid, None)
    if expired_ids:
        logger.info(f"🧹 Evicted {len(expired_ids)} expired analyses")
    return len(expired_ids)


async def analysis_state_cleanup_loop() -> None:
    """Background task that periodically evicts expired analyses."""
    while True:
        try:
            await asyncio.sleep(ANALYSIS_CLEANUP_INTERVAL_SECONDS)
            await cleanup_expired_analyses()
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 - keep loop alive
            logger.warning(f"analysis_state cleanup loop error: {e}")


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
    start_time = datetime.utcnow()
    code_lines = len(code.split("\n"))

    try:
        # Update state
        async with analysis_state_lock:
            if analysis_id in analysis_state:
                analysis_state[analysis_id]["status"] = "processing"
                analysis_state[analysis_id]["start_time"] = start_time
                analysis_state[analysis_id]["workflow_step"] = "parsing"

        # Register a lightweight progress callback so each workflow node can
        # publish its step into analysis_state[analysis_id]["workflow_step"].
        def _record_step(step: str) -> None:
            entry = analysis_state.get(analysis_id)
            if entry is not None:
                entry["workflow_step"] = step
                logger.debug(f"Workflow step: {step}")  # Log each step

        register_progress_callback(analysis_id, _record_step)

        # Log inference start with color
        log_inference_start(
            logger,
            analysis_id=analysis_id,
            code_lines=code_lines,
            use_rag=include_rag,
            model=model or "auto-select"
        )

        # ✅ CALL LANGGRAPH WORKFLOW VIA DETECTION INTEGRATION
        detection_result = await run_code_smell_detection_with_scoring(
            code=code,
            sample_id=file_name,
            use_rag=include_rag,
            model=model,  # Pass specified model, or None for agentic auto-selection
            analysis_id=analysis_id,
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
        metrics_dict = detection_result.get("metrics", {})

        analysis_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

        # Log metrics details
        logger.debug(
            f"Analysis metrics: F1={metrics_dict.get('f1')}, Mode={metrics_dict.get('evaluation_mode')}, "
            f"Has GT={metrics_dict.get('has_ground_truth')}"
        )

        result = AnalysisResultResponse(
            analysis_id=analysis_id,
            code_hash=compute_code_hash(code),
            language=language or "python",
            findings=findings,
            metrics=metrics,
            analysis_time_ms=analysis_time_ms,
            model_used=actual_model,  # ✅ From LangGraph workflow
            model_reasoning=model_reasoning,  # ✅ From agentic selection
            cache_hit=False,
            completed_at=datetime.utcnow(),
        )

        # Store results in state
        async with analysis_state_lock:
            if analysis_id in analysis_state:
                entry = analysis_state[analysis_id]
                entry["status"] = "completed"
                entry["result"] = result
                entry["metrics"] = detection_result["metrics"]
                entry["ground_truth"] = detection_result["ground_truth"]
                entry["model_used"] = actual_model
                entry["model_reasoning"] = model_reasoning
                entry["completed_at"] = datetime.utcnow()
                entry["expires_at"] = entry["completed_at"] + ANALYSIS_STATE_TTL

        # Format F1 score safely (can be None for user snippets without ground truth)
        f1_val = detection_result['metrics']['f1']
        eval_mode = detection_result['metrics'].get('evaluation_mode', 'unknown')

        # Log inference end with color
        log_inference_end(
            logger,
            analysis_id=analysis_id,
            findings_count=len(findings),
            f1_score=f1_val,
            time_ms=analysis_time_ms,
            model=actual_model
        )

    except Exception as e:
        import traceback
        tb_str = traceback.format_exc()
        analysis_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

        # Log inference error with color + full traceback
        log_inference_error(
            logger,
            analysis_id=analysis_id,
            error_msg=str(e),
            time_ms=analysis_time_ms
        )
        logger.error("Analysis %s failed", analysis_id, exc_info=True)

        async with analysis_state_lock:
            if analysis_id in analysis_state:
                entry = analysis_state[analysis_id]
                entry["status"] = "failed"
                entry["error"] = str(e)
                entry["traceback"] = tb_str
                entry["completed_at"] = datetime.utcnow()
                entry["expires_at"] = entry["completed_at"] + ANALYSIS_STATE_TTL
    finally:
        unregister_progress_callback(analysis_id)


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
        now = datetime.utcnow()
        async with analysis_state_lock:
            analysis_state[analysis_id] = {
                "status": "queued",
                "created_at": now,
                "expires_at": now + ANALYSIS_STATE_TTL,
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
    - 404: Analysis not found or evicted
    - 410: Results older than the retention window (24h)
    - 425: Still processing (try again later)
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

        # L3: Results age out after ANALYSIS_STATE_TTL. Surface expired entries
        # as 410 Gone instead of serving stale data in the window between
        # scheduled cleanup ticks. Entry is also evicted lazily here.
        expires_at = state.get("expires_at")
        if expires_at and expires_at < datetime.utcnow() and state.get("status") in ("completed", "failed"):
            async with analysis_state_lock:
                analysis_state.pop(analysis_id, None)
            raise HTTPException(
                status_code=410,
                detail=(
                    f"Results for analysis '{analysis_id}' have expired "
                    "(retention window: 24h)."
                ),
            )

        # Handle different states
        if state["status"] == "queued" or state["status"] == "processing":
            # L2: 202 Accepted is semantically reserved for the POST that
            # *accepted* the job. Polling an unfinished job is a "too early"
            # condition, so use 425 Too Early.
            raise HTTPException(
                status_code=425,
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

        # ✅ ADD F1 METRICS TO RESPONSE (can be None when no ground truth exists)
        metrics = state.get("metrics", {})
        result.f1_score = metrics.get("f1")  # None if no ground truth
        result.precision = metrics.get("precision")  # Confidence proxy or None
        result.recall = metrics.get("recall")  # None if no ground truth
        result.ground_truth_count = state.get("ground_truth", {}).get("count", 0)

        logger.info(
            f"Retrieved results for analysis {analysis_id}: "
            f"F1={metrics.get('f1') if metrics.get('f1') is not None else 'N/A'}, "
            f"mode={metrics.get('evaluation_mode', 'unknown')}"
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

        # Use the real workflow step published by each LangGraph node (if any).
        # Fall back to coarse status only when no step has been reported yet.
        if status == "processing":
            status = state.get("workflow_step") or "parsing"

        progress_map = {
            "queued": (5, "Queued for processing"),
            "parsing": (15, "Parsing code..."),
            "model_selection": (25, "Selecting LLM model..."),
            "chunking": (35, "Chunking code..."),
            "rag_retrieval": (45, "Retrieving similar examples..."),
            "inference": (70, "Running LLM inference..."),
            "validation": (85, "Validating findings..."),
            "aggregating": (95, "Aggregating results..."),
            "completed": (100, "Analysis complete"),
            "failed": (100, "Analysis failed"),
        }

        percentage, step_desc = progress_map.get(
            status,
            (10, "Processing...")
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
        async with analysis_state_lock:
            if comparison_id in analysis_state:
                analysis_state[comparison_id]["status"] = "processing"
        logger.info(f"Starting RAG comparison {comparison_id}")

        # Run comparison
        comparison_result = await compare_detection_approaches(
            code=code,
            sample_id=file_name,
            models=["llama3:8b"],
        )

        # Store results
        async with analysis_state_lock:
            if comparison_id in analysis_state:
                entry = analysis_state[comparison_id]
                entry["status"] = "completed"
                entry["result"] = comparison_result
                entry["completed_at"] = datetime.utcnow()
                entry["expires_at"] = entry["completed_at"] + ANALYSIS_STATE_TTL

        logger.info(f"RAG comparison {comparison_id} complete")

    except Exception as e:
        logger.error(f"Comparison {comparison_id} failed: {e}")
        async with analysis_state_lock:
            if comparison_id in analysis_state:
                entry = analysis_state[comparison_id]
                entry["status"] = "failed"
                entry["error"] = str(e)
                entry["completed_at"] = datetime.utcnow()
                entry["expires_at"] = entry["completed_at"] + ANALYSIS_STATE_TTL


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
    now = datetime.utcnow()
    async with analysis_state_lock:
        analysis_state[comparison_id] = {
            "status": "queued",
            "type": "comparison",
            "created_at": now,
            "expires_at": now + ANALYSIS_STATE_TTL,
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
        async with analysis_state_lock:
            if analysis_id not in analysis_state:
                raise HTTPException(
                    status_code=404,
                    detail=f"Analysis '{analysis_id}' not found"
                )
            analysis_state.pop(analysis_id, None)

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


@router.get(
    "/ground-truth/samples",
    summary="List Available Ground Truth Samples",
    tags=["evaluation"],
)
async def list_ground_truth_samples() -> Dict:
    """
    List all available ground truth samples for F1 score evaluation.

    Use these sample names in the file_name field to enable F1 score calculation
    instead of confidence-based assessment.

    Returns:
        - samples: List of available sample IDs
        - total: Total number of available samples
        - smells_per_sample: Mapping of sample_id to number of detected smells
    """
    try:
        from src.api.detection_integration import load_ground_truth_from_file

        ground_truth = load_ground_truth_from_file()

        samples = list(ground_truth.keys())
        smells_per_sample = {
            sample_id: len(smells)
            for sample_id, smells in ground_truth.items()
        }

        logger.info(f"Listed {len(samples)} ground truth samples")

        return {
            "samples": samples,
            "total": len(samples),
            "smells_per_sample": smells_per_sample,
            "info": "Use any of these sample names in the file_name field to get F1 score calculation"
        }

    except Exception as e:
        logger.error(f"Error listing ground truth samples: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to list ground truth samples"
        )
    """
    List all available ground truth samples for F1 score evaluation.

    Use these sample names in the file_name field to enable F1 score calculation
    instead of confidence-based assessment.

    Returns:
        - samples: List of available sample IDs
        - total: Total number of available samples
        - smells_per_sample: Mapping of sample_id to number of detected smells
    """
    try:
        from src.api.detection_integration import load_ground_truth_from_file

        ground_truth = load_ground_truth_from_file()

        samples = list(ground_truth.keys())
        smells_per_sample = {
            sample_id: len(smells)
            for sample_id, smells in ground_truth.items()
        }

        logger.info(f"Listed {len(samples)} ground truth samples")

        return {
            "samples": samples,
            "total": len(samples),
            "smells_per_sample": smells_per_sample,
            "info": "Use any of these sample names in the file_name field to get F1 score calculation"
        }

    except Exception as e:
        logger.error(f"Error listing ground truth samples: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to list ground truth samples"
        )
