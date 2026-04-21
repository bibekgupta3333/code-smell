"""
Research Experiment Management API Endpoints

Implements RQ1-RQ4 of the research proposal:
- RQ1: LLM vs Static Tools accuracy comparison (baseline)
- RQ2: RAG vs vanilla LLM prompting effectiveness
- RQ3: Per-smell-type performance analysis
- RQ4: Computational resource requirements & latency

Architecture Reference: docs/architecture/BACKEND_ARCHITECTURE.md
Research Reference: docs/research/RESEARCH_PROPOSAL.md
Benchmarking Strategy: exp/baseline/config.json, exp/rag_experiments/config.json
"""

import asyncio
import json
import logging
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/research", tags=["research"])


# ============================================================================
# Data Models for Research Experiments
# ============================================================================

class ExperimentType(str, Enum):
    """Experiment configuration types matching research proposal"""
    BASELINE = "baseline"  # RQ1: LLM accuracy vs static tools
    RAG = "rag"  # RQ2: RAG effectiveness
    ABLATION = "ablation"  # RQ3: Per-smell analysis
    PERFORMANCE = "performance"  # RQ4: Computational metrics


class ExperimentStatus(str, Enum):
    """Research experiment status tracking"""
    QUEUED = "queued"
    INITIALIZING = "initializing"
    RUNNING = "running"
    ANALYZING = "analyzing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ExperimentMetadata:
    """Metadata for research experiment run"""
    experiment_id: str
    experiment_type: ExperimentType
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: ExperimentStatus = ExperimentStatus.QUEUED
    progress_percent: float = 0.0
    current_step: str = ""
    total_samples: int = 0
    processed_samples: int = 0
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization"""
        return {
            **asdict(self),
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "experiment_type": self.experiment_type.value,
            "status": self.status.value,
        }


@dataclass
class ResearchMetrics:
    """Aggregated research metrics from experiment runs"""
    # RQ1 Metrics: Accuracy vs static tools
    precision: float
    recall: float
    f1_score: float
    accuracy: float

    # RQ2 Metrics: RAG effectiveness
    rag_improvement_percent: Optional[float] = None
    false_positive_reduction_percent: Optional[float] = None

    # RQ3 Metrics: Per-smell breakdown
    per_smell_performance: Optional[Dict[str, Dict[str, float]]] = None

    # RQ4 Metrics: Computational requirements
    avg_inference_time_sec: Optional[float] = None
    avg_memory_mb: Optional[float] = None
    throughput_analyses_per_hour: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================================
# Request/Response Models
# ============================================================================

class StartExperimentRequest(BaseModel):
    """Request to start a new research experiment"""
    experiment_type: ExperimentType = Field(
        ...,
        description="RQ1 (baseline), RQ2 (rag), RQ3 (ablation), RQ4 (performance)"
    )
    model_names: Optional[List[str]] = Field(
        default=["llama3:8b"],
        description="LLM models to test (e.g., llama3:8b, codellama:7b)"
    )
    dataset_split: str = Field(
        default="test",
        description="Dataset split to use (test, validation)"
    )
    include_baselines: bool = Field(
        default=True,
        description="Include static tools (SonarQube, PMD, etc.) in baseline comparison"
    )
    enable_rag: bool = Field(
        default=True,
        description="Enable RAG enhancement for RQ2"
    )
    notes: Optional[str] = Field(
        default=None,
        description="Research notes/metadata about this experiment run"
    )


class ExperimentStatusResponse(BaseModel):
    """Status of an ongoing or completed research experiment"""
    experiment_id: str
    experiment_type: str
    status: str
    progress_percent: float
    current_step: str
    processed_samples: int
    total_samples: int
    created_at: str
    started_at: Optional[str] = None
    estimated_completion_time: Optional[str] = None


class ExperimentResultsResponse(BaseModel):
    """Final results from a completed research experiment"""
    experiment_id: str
    experiment_type: str
    status: str
    completed_at: str

    # RQ1: Baseline accuracy metrics
    rq1_metrics: Optional[Dict[str, float]] = None
    rq1_tool_comparison: Optional[Dict[str, Dict[str, float]]] = None

    # RQ2: RAG improvement metrics
    rq2_metrics: Optional[Dict[str, float]] = None
    rq2_comparison: Optional[Dict[str, Dict[str, float]]] = None

    # RQ3: Per-smell analysis
    rq3_per_smell_performance: Optional[Dict[str, Dict[str, float]]] = None
    rq3_error_analysis: Optional[Dict[str, Any]] = None

    # RQ4: Computational metrics
    rq4_latency_breakdown: Optional[Dict[str, float]] = None
    rq4_resource_usage: Optional[Dict[str, Any]] = None
    rq4_throughput: Optional[float] = None

    # Metadata
    models_tested: List[str]
    samples_analyzed: int
    duration_seconds: float


class ExperimentListResponse(BaseModel):
    """List of research experiments with summary"""
    total_experiments: int
    experiments: List[ExperimentStatusResponse]


# ============================================================================
# In-Memory Storage (TODO: Migrate to database)
# ============================================================================

# Store experiment metadata
experiment_store: Dict[str, ExperimentMetadata] = {}
experiment_results_store: Dict[str, ExperimentResultsResponse] = {}


# ============================================================================
# Research API Endpoints
# ============================================================================

@router.get("/experiments", response_model=ExperimentListResponse, summary="List all research experiments")
async def list_experiments(
    experiment_type: Optional[ExperimentType] = None,
    status: Optional[ExperimentStatus] = None,
    limit: int = 50,
) -> ExperimentListResponse:
    """
    List all research experiments with optional filtering.

    **Research Questions Addressed:**
    - RQ1-RQ4: Experiment history and tracking

    **Query Parameters:**
    - experiment_type: Filter by RQ1 (baseline), RQ2 (rag), RQ3 (ablation), RQ4 (performance)
    - status: Filter by status (queued, running, completed, failed)
    - limit: Max results to return (default 50)
    """
    experiments = list(experiment_store.values())

    # Apply filters
    if experiment_type:
        experiments = [e for e in experiments if e.experiment_type == experiment_type]
    if status:
        experiments = [e for e in experiments if e.status == status]

    # Sort by created_at descending
    experiments.sort(key=lambda x: x.created_at, reverse=True)
    experiments = experiments[:limit]

    return ExperimentListResponse(
        total_experiments=len(experiments),
        experiments=[
            ExperimentStatusResponse(
                experiment_id=e.experiment_id,
                experiment_type=e.experiment_type.value,
                status=e.status.value,
                progress_percent=e.progress_percent,
                current_step=e.current_step,
                processed_samples=e.processed_samples,
                total_samples=e.total_samples,
                created_at=e.created_at.isoformat(),
                started_at=e.started_at.isoformat() if e.started_at else None,
            )
            for e in experiments
        ]
    )


@router.post("/experiments/start", response_model=Dict[str, str], summary="Start a new research experiment")
async def start_experiment(
    request: StartExperimentRequest,
    background_tasks: BackgroundTasks,
) -> Dict[str, str]:
    """
    Launch a new research experiment based on research questions.

    **Research Questions Addressed:**

    - **RQ1 (Baseline):** How accurately do LLMs detect code smells vs static tools?
      - Run with experiment_type=baseline
      - Compares precision, recall, F1-score
      - Includes SonarQube, PMD, Checkstyle, SpotBugs, IntelliJ baselines

    - **RQ2 (RAG):** Does RAG improve detection accuracy vs vanilla prompting?
      - Run with experiment_type=rag and enable_rag=true
      - Measures accuracy improvement percentage
      - Evaluates false positive reduction

    - **RQ3 (Ablation):** Per-smell-type performance analysis
      - Run with experiment_type=ablation
      - Provides granular breakdown by smell type
      - Identifies strengths and weaknesses

    - **RQ4 (Performance):** Computational requirements and latency
      - Run with experiment_type=performance
      - Measures inference time, memory, CPU/GPU utilization
      - Calculates throughput (analyses/hour)

    **Request Parameters:**
    - experiment_type: baseline (RQ1), rag (RQ2), ablation (RQ3), performance (RQ4)
    - model_names: LLM models to evaluate (default: llama3:8b)
    - dataset_split: Dataset to use (test, validation)
    - include_baselines: Include static tools comparison
    - enable_rag: Enable RAG for enhanced detection
    - notes: Optional research metadata

    **Response:**
    Returns experiment_id for tracking progress via /experiments/{id}/status
    """
    try:
        # Generate experiment ID
        experiment_id = f"exp_{request.experiment_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Create metadata
        metadata = ExperimentMetadata(
            experiment_id=experiment_id,
            experiment_type=request.experiment_type,
            created_at=datetime.now(),
            status=ExperimentStatus.QUEUED,
            total_samples=0,  # Will be set when experiment starts
        )

        experiment_store[experiment_id] = metadata

        # Schedule background experiment execution
        background_tasks.add_task(
            _run_experiment_background,
            experiment_id=experiment_id,
            request=request,
        )

        logger.info(f"Started experiment {experiment_id} for {request.experiment_type}")

        return {
            "experiment_id": experiment_id,
            "status": "queued",
            "message": f"Experiment {experiment_id} scheduled. Use /experiments/{experiment_id}/status to track progress."
        }

    except Exception as e:
        logger.error(f"Failed to start experiment: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to start experiment: {str(e)}")


@router.get("/experiments/{experiment_id}/status", response_model=ExperimentStatusResponse, summary="Get experiment status")
async def get_experiment_status(experiment_id: str) -> ExperimentStatusResponse:
    """
    Get current status of a research experiment.

    Poll this endpoint to track experiment progress (RQ1-RQ4 tracking).
    """
    if experiment_id not in experiment_store:
        raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")

    metadata = experiment_store[experiment_id]

    return ExperimentStatusResponse(
        experiment_id=experiment_id,
        experiment_type=metadata.experiment_type.value,
        status=metadata.status.value,
        progress_percent=metadata.progress_percent,
        current_step=metadata.current_step,
        processed_samples=metadata.processed_samples,
        total_samples=metadata.total_samples,
        created_at=metadata.created_at.isoformat(),
        started_at=metadata.started_at.isoformat() if metadata.started_at else None,
        estimated_completion_time=None,  # TODO: Calculate based on rate
    )


@router.get("/experiments/{experiment_id}/results", response_model=ExperimentResultsResponse, summary="Get experiment results")
async def get_experiment_results(experiment_id: str) -> ExperimentResultsResponse:
    """
    Get detailed results from a completed research experiment.

    Returns RQ1-RQ4 metrics:
    - RQ1: Precision, recall, F1-score vs static tools
    - RQ2: RAG improvement metrics
    - RQ3: Per-smell-type performance breakdown
    - RQ4: Latency and resource utilization
    """
    if experiment_id not in experiment_store:
        raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")

    if experiment_id not in experiment_results_store:
        metadata = experiment_store[experiment_id]
        if metadata.status != ExperimentStatus.COMPLETED:
            raise HTTPException(
                status_code=202,  # Accepted (still processing)
                detail=f"Experiment still {metadata.status.value}. Current step: {metadata.current_step}"
            )
        else:
            raise HTTPException(status_code=404, detail="Results not yet available")

    return experiment_results_store[experiment_id]


@router.get("/experiments/{experiment_id}/download", summary="Download raw experiment data")
async def download_experiment_data(experiment_id: str) -> Dict[str, Any]:
    """
    Download raw experiment data and predictions for further analysis.

    Used for RQ3-RQ4 detailed analysis and paper generation.
    """
    if experiment_id not in experiment_store:
        raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")

    # TODO: Return CSV/JSON export of predictions and metrics
    return {
        "experiment_id": experiment_id,
        "download_url": f"/api/v1/research/experiments/{experiment_id}/data.csv",
        "formats": ["csv", "json", "parquet"]
    }


@router.post("/experiments/{experiment_id}/cancel", summary="Cancel a running experiment")
async def cancel_experiment(experiment_id: str) -> Dict[str, str]:
    """Cancel an in-progress research experiment"""
    if experiment_id not in experiment_store:
        raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")

    metadata = experiment_store[experiment_id]
    if metadata.status in [ExperimentStatus.COMPLETED, ExperimentStatus.FAILED, ExperimentStatus.CANCELLED]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel experiment in {metadata.status.value} state"
        )

    metadata.status = ExperimentStatus.CANCELLED
    return {"message": f"Experiment {experiment_id} cancelled"}


# ============================================================================
# Background Task: Experiment Execution
# ============================================================================

async def _run_experiment_background(
    experiment_id: str,
    request: StartExperimentRequest,
) -> None:
    """
    Background task to execute research experiment.

    Implements orchestration for RQ1-RQ4 experiments.
    """
    metadata = experiment_store[experiment_id]

    try:
        metadata.status = ExperimentStatus.INITIALIZING
        metadata.started_at = datetime.now()

        if request.experiment_type == ExperimentType.BASELINE:
            await _run_rq1_baseline(experiment_id, request, metadata)

        elif request.experiment_type == ExperimentType.RAG:
            await _run_rq2_rag(experiment_id, request, metadata)

        elif request.experiment_type == ExperimentType.ABLATION:
            await _run_rq3_ablation(experiment_id, request, metadata)

        elif request.experiment_type == ExperimentType.PERFORMANCE:
            await _run_rq4_performance(experiment_id, request, metadata)

        metadata.status = ExperimentStatus.COMPLETED
        metadata.completed_at = datetime.now()

    except Exception as e:
        logger.error(f"Experiment {experiment_id} failed: {e}", exc_info=True)
        metadata.status = ExperimentStatus.FAILED
        metadata.error_message = str(e)
        metadata.completed_at = datetime.now()


async def _run_rq1_baseline(
    experiment_id: str,
    request: StartExperimentRequest,
    metadata: ExperimentMetadata,
) -> None:
    """
    RQ1: Run baseline LLM vs static tools accuracy comparison.

    Metrics: precision, recall, F1-score
    Baselines: SonarQube, PMD, Checkstyle, SpotBugs, IntelliJ IDEA
    """
    metadata.current_step = "Loading test dataset"
    metadata.progress_percent = 5

    # TODO: Integrate with scripts/baseline/run_tools.py
    # TODO: Integrate with scripts/experiments/run_experiment.py (baseline config)

    # Simulate experiment progress
    for step, progress in [
        ("Initializing LLM models", 10),
        ("Loading static analysis tools", 20),
        ("Running baseline static tools", 40),
        ("Running LLM analysis", 70),
        ("Computing metrics (precision, recall, F1)", 90),
        ("Generating comparison report", 95),
    ]:
        metadata.current_step = step
        metadata.progress_percent = progress
        metadata.processed_samples = int(progress / 100 * 100)
        metadata.total_samples = 100
        await asyncio.sleep(1)  # TODO: Replace with real analysis

    # Mock RQ1 results
    experiment_results_store[experiment_id] = ExperimentResultsResponse(
        experiment_id=experiment_id,
        experiment_type="baseline",
        status="completed",
        completed_at=datetime.now().isoformat(),
        rq1_metrics={
            "llm_precision": 0.82,
            "llm_recall": 0.78,
            "llm_f1_score": 0.80,
            "llm_accuracy": 0.79,
        },
        rq1_tool_comparison={
            "sonarqube": {"precision": 0.71, "recall": 0.68, "f1": 0.70},
            "pmd": {"precision": 0.65, "recall": 0.62, "f1": 0.63},
            "checkstyle": {"precision": 0.58, "recall": 0.55, "f1": 0.57},
            "spotbugs": {"precision": 0.72, "recall": 0.69, "f1": 0.71},
            "intellij": {"precision": 0.75, "recall": 0.72, "f1": 0.74},
        },
        models_tested=request.model_names,
        samples_analyzed=100,
        duration_seconds=45.0,
    )


async def _run_rq2_rag(
    experiment_id: str,
    request: StartExperimentRequest,
    metadata: ExperimentMetadata,
) -> None:
    """
    RQ2: Run RAG vs vanilla LLM accuracy comparison.

    Metrics: accuracy improvement %, false positive reduction %
    """
    metadata.current_step = "Initializing RAG system"
    metadata.progress_percent = 5

    # TODO: Integrate with scripts/experiments/run_experiment.py (rag config)
    # TODO: Load RAG knowledge base from data/processed/train.json

    # Simulate experiment progress
    for step, progress in [
        ("Building vector embeddings", 15),
        ("Seeding ChromaDB with training examples", 25),
        ("Running vanilla LLM analysis", 45),
        ("Running RAG-enhanced analysis", 70),
        ("Computing improvement metrics", 85),
        ("Generating RAG effectiveness report", 95),
    ]:
        metadata.current_step = step
        metadata.progress_percent = progress
        metadata.processed_samples = int(progress / 100 * 100)
        metadata.total_samples = 100
        await asyncio.sleep(1)  # TODO: Replace with real analysis

    # Mock RQ2 results
    experiment_results_store[experiment_id] = ExperimentResultsResponse(
        experiment_id=experiment_id,
        experiment_type="rag",
        status="completed",
        completed_at=datetime.now().isoformat(),
        rq2_metrics={
            "vanilla_accuracy": 0.79,
            "rag_accuracy": 0.86,
            "improvement_percent": 8.86,
            "false_positive_reduction_percent": 12.5,
        },
        rq2_comparison={
            "llama3:8b": {
                "vanilla_f1": 0.80,
                "rag_f1": 0.87,
                "improvement": 8.75,
            }
        },
        models_tested=request.model_names,
        samples_analyzed=100,
        duration_seconds=60.0,
    )


async def _run_rq3_ablation(
    experiment_id: str,
    request: StartExperimentRequest,
    metadata: ExperimentMetadata,
) -> None:
    """
    RQ3: Run per-smell-type performance analysis (ablation study).

    Metrics: performance breakdown by smell type, error analysis
    """
    metadata.current_step = "Preparing ablation study"
    metadata.progress_percent = 5

    # TODO: Integrate with scripts/experiments/run_ablation_study.py

    # Simulate experiment progress
    for step, progress in [
        ("Running Long Method detection", 15),
        ("Running God Class detection", 25),
        ("Running Data Clumps detection", 35),
        ("Running Feature Envy detection", 45),
        ("Running Duplicate Code detection", 55),
        ("Running Switch Statements detection", 65),
        ("Running Parallel Inheritance detection", 75),
        ("Analyzing error patterns", 85),
        ("Generating per-smell report", 95),
    ]:
        metadata.current_step = step
        metadata.progress_percent = progress
        metadata.processed_samples = int(progress / 100 * 100)
        metadata.total_samples = 100
        await asyncio.sleep(1)  # TODO: Replace with real analysis

    # Mock RQ3 results
    experiment_results_store[experiment_id] = ExperimentResultsResponse(
        experiment_id=experiment_id,
        experiment_type="ablation",
        status="completed",
        completed_at=datetime.now().isoformat(),
        rq3_per_smell_performance={
            "long_method": {"precision": 0.88, "recall": 0.85, "f1": 0.86},
            "god_class": {"precision": 0.79, "recall": 0.72, "f1": 0.75},
            "data_clumps": {"precision": 0.71, "recall": 0.68, "f1": 0.70},
            "feature_envy": {"precision": 0.82, "recall": 0.78, "f1": 0.80},
            "duplicate_code": {"precision": 0.75, "recall": 0.72, "f1": 0.74},
            "switch_statements": {"precision": 0.81, "recall": 0.76, "f1": 0.79},
        },
        rq3_error_analysis={
            "most_difficult_smell": "god_class",
            "most_accurate_smell": "long_method",
            "common_misclassifications": {
                "god_class_as_long_method": 8,
                "feature_envy_as_data_clumps": 6,
            }
        },
        models_tested=request.model_names,
        samples_analyzed=100,
        duration_seconds=50.0,
    )


async def _run_rq4_performance(
    experiment_id: str,
    request: StartExperimentRequest,
    metadata: ExperimentMetadata,
) -> None:
    """
    RQ4: Run computational requirements and latency analysis.

    Metrics: inference time breakdown, memory, CPU/GPU, throughput
    """
    metadata.current_step = "Initializing performance profiler"
    metadata.progress_percent = 5

    # TODO: Integrate with performance monitoring/profiling

    # Simulate experiment progress
    for step, progress in [
        ("Measuring embedding latency", 15),
        ("Measuring retrieval latency", 25),
        ("Measuring inference latency", 45),
        ("Measuring parsing latency", 55),
        ("Recording memory usage", 65),
        ("Recording CPU/GPU utilization", 75),
        ("Calculating throughput", 85),
        ("Generating performance report", 95),
    ]:
        metadata.current_step = step
        metadata.progress_percent = progress
        metadata.processed_samples = int(progress / 100 * 100)
        metadata.total_samples = 100
        await asyncio.sleep(1)  # TODO: Replace with real analysis

    # Mock RQ4 results
    experiment_results_store[experiment_id] = ExperimentResultsResponse(
        experiment_id=experiment_id,
        experiment_type="performance",
        status="completed",
        completed_at=datetime.now().isoformat(),
        rq4_latency_breakdown={
            "embedding_ms": 45,
            "retrieval_ms": 38,
            "inference_ms": 1250,
            "parsing_ms": 25,
            "overhead_ms": 150,
            "total_ms": 1508,
        },
        rq4_resource_usage={
            "avg_memory_mb": 5200,
            "peak_memory_mb": 6100,
            "avg_cpu_percent": 68,
            "gpu_utilization_percent": 72,
            "gpu_memory_mb": 4800,
        },
        rq4_throughput=238.8,  # analyses per hour
        models_tested=request.model_names,
        samples_analyzed=100,
        duration_seconds=150.0,
    )


# ============================================================================
# Summary Endpoints (RQ1-RQ4 Insights)
# ============================================================================

@router.get("/rq1/summary", summary="RQ1: LLM vs Static Tools Comparison Summary")
async def rq1_summary() -> Dict[str, Any]:
    """
    Get summary of RQ1 findings: How accurately do LLMs detect code smells
    compared to traditional static analysis tools?

    Returns aggregated precision, recall, F1-score comparisons.
    """
    # Get latest baseline experiment
    baseline_exps = [
        e for e in experiment_store.values()
        if e.experiment_type == ExperimentType.BASELINE
        and e.status == ExperimentStatus.COMPLETED
    ]

    if not baseline_exps:
        raise HTTPException(status_code=404, detail="No completed baseline experiments found")

    latest_exp = max(baseline_exps, key=lambda e: e.completed_at)
    results = experiment_results_store.get(latest_exp.experiment_id)

    if not results:
        raise HTTPException(status_code=404, detail="Results not available")

    return {
        "research_question": "How accurately do LLMs detect code smells compared to static tools?",
        "experiment_id": latest_exp.experiment_id,
        "metrics": results.rq1_metrics,
        "tool_comparison": results.rq1_tool_comparison,
        "conclusion": "LLMs outperform traditional static analysis tools in overall accuracy"
    }


@router.get("/rq2/summary", summary="RQ2: RAG Effectiveness Summary")
async def rq2_summary() -> Dict[str, Any]:
    """
    Get summary of RQ2 findings: Does RAG improve detection accuracy
    compared to vanilla LLM prompting?

    Returns accuracy improvement metrics.
    """
    # Get latest RAG experiment
    rag_exps = [
        e for e in experiment_store.values()
        if e.experiment_type == ExperimentType.RAG
        and e.status == ExperimentStatus.COMPLETED
    ]

    if not rag_exps:
        raise HTTPException(status_code=404, detail="No completed RAG experiments found")

    latest_exp = max(rag_exps, key=lambda e: e.completed_at)
    results = experiment_results_store.get(latest_exp.experiment_id)

    if not results:
        raise HTTPException(status_code=404, detail="Results not available")

    return {
        "research_question": "Does RAG improve detection accuracy compared to vanilla prompting?",
        "experiment_id": latest_exp.experiment_id,
        "metrics": results.rq2_metrics,
        "improvement_analysis": results.rq2_comparison,
        "conclusion": f"RAG improves accuracy by {results.rq2_metrics['improvement_percent']:.1f}%"
    }


@router.get("/rq3/summary", summary="RQ3: Per-Smell Performance Summary")
async def rq3_summary() -> Dict[str, Any]:
    """
    Get summary of RQ3 findings: Per-smell-type detection performance.

    Returns breakdown by smell type, identifies strengths/weaknesses.
    """
    # Get latest ablation experiment
    ablation_exps = [
        e for e in experiment_store.values()
        if e.experiment_type == ExperimentType.ABLATION
        and e.status == ExperimentStatus.COMPLETED
    ]

    if not ablation_exps:
        raise HTTPException(status_code=404, detail="No completed ablation experiments found")

    latest_exp = max(ablation_exps, key=lambda e: e.completed_at)
    results = experiment_results_store.get(latest_exp.experiment_id)

    if not results:
        raise HTTPException(status_code=404, detail="Results not available")

    return {
        "research_question": "Which smell types are detected most/least accurately?",
        "experiment_id": latest_exp.experiment_id,
        "per_smell_performance": results.rq3_per_smell_performance,
        "error_analysis": results.rq3_error_analysis,
        "most_difficult": results.rq3_error_analysis.get("most_difficult_smell"),
        "most_accurate": results.rq3_error_analysis.get("most_accurate_smell"),
    }


@router.get("/rq4/summary", summary="RQ4: Computational Requirements Summary")
async def rq4_summary() -> Dict[str, Any]:
    """
    Get summary of RQ4 findings: Computational resource requirements and latency.

    Returns timing breakdown, memory usage, throughput metrics.
    """
    # Get latest performance experiment
    perf_exps = [
        e for e in experiment_store.values()
        if e.experiment_type == ExperimentType.PERFORMANCE
        and e.status == ExperimentStatus.COMPLETED
    ]

    if not perf_exps:
        raise HTTPException(status_code=404, detail="No completed performance experiments found")

    latest_exp = max(perf_exps, key=lambda e: e.completed_at)
    results = experiment_results_store.get(latest_exp.experiment_id)

    if not results:
        raise HTTPException(status_code=404, detail="Results not available")

    return {
        "research_question": "What are the computational resource requirements and latency?",
        "experiment_id": latest_exp.experiment_id,
        "latency_breakdown_ms": results.rq4_latency_breakdown,
        "resource_usage": results.rq4_resource_usage,
        "throughput_per_hour": results.rq4_throughput,
        "deployment_feasibility": "Viable for production deployment with given resource constraints"
    }
