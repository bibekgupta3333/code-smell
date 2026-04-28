"""
LangGraph Workflow Graph for Multi-Agent Code Smell Detection
Orchestrates the entire analysis workflow using state machines and conditional routing.

Architecture: LangGraph-based state machine for workflow orchestration
Uses TypedDict for state management and conditional edges for routing
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional
from dataclasses import field
from datetime import datetime

from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel

from config import MODEL_SELECTION_SCORE
from src.utils.common import CodeSmellFinding, SeverityLevel
from src.utils.logger import log_workflow_step, log_agent_event
from src.analysis.code_parser import CodeParser, CodeMetrics, ProgrammingLanguage
from src.rag.rag_retriever import RAGRetriever
from src.analysis.code_smell_detector import CodeSmellDetector
from src.analysis.quality_validator import QualityValidator
from src.llm.llm_client import OllamaClient

logger = logging.getLogger(__name__)


# ============================================================================
# State Definition
# ============================================================================

class AnalysisState(BaseModel):
    """State for code smell analysis workflow with agentic model selection.

    This state is passed through all nodes in the LangGraph workflow.
    Each node can read and modify the state.

    Attributes:
        code_snippet: Original code to analyze
        file_name: Source file name for context
        language: Detected programming language
        code_metrics: Extracted code metrics
        chunks: Code chunks for parallel analysis
        current_chunk_idx: Index of chunk being processed
        rag_context: Retrieved relevant examples
        detections: Code smell findings from detectors
        validated_findings: Findings after validation
        errors: List of errors encountered
        workflow_step: Current step in workflow (for logging)
        start_time: Workflow start timestamp
        metadata: Additional metadata for tracking
        model: LLM model to use (from Ollama) for this analysis
        use_rag: Whether to use RAG context for this analysis
        available_models: List of available Ollama models
        model_reasoning: Agent reasoning for model selection
    """
    code_snippet: str
    file_name: str = "code"
    language: Optional[ProgrammingLanguage] = None
    code_metrics: Optional[CodeMetrics] = None
    chunks: List[str] = field(default_factory=list)  # pylint: disable=invalid-field-call
    current_chunk_idx: int = 0
    rag_context: Dict[str, Any] = field(default_factory=dict)  # pylint: disable=invalid-field-call
    detections: List[CodeSmellFinding] = field(default_factory=list)  # pylint: disable=invalid-field-call
    validated_findings: List[CodeSmellFinding] = field(default_factory=list)  # pylint: disable=invalid-field-call
    errors: List[str] = field(default_factory=list)  # pylint: disable=invalid-field-call
    workflow_step: str = "initialize"
    start_time: datetime = field(default_factory=datetime.now)  # pylint: disable=invalid-field-call
    metadata: Dict[str, Any] = field(default_factory=dict)  # pylint: disable=invalid-field-call
    model: Optional[str] = None  # Selected LLM model (e.g., "llama3:8b", "mistral", "codellama")
    use_rag: bool = True  # Whether to use RAG context
    available_models: List[str] = field(default_factory=list)  # pylint: disable=invalid-field-call
    model_reasoning: Optional[str] = None  # Agent reasoning for model selection

    class Config:
        arbitrary_types_allowed = True


# ============================================================================
# Progress Reporting
# ============================================================================
# Module-level registry mapping analysis_id → progress callback(step: str).
# The API layer (detection_integration → analysis route) registers a callback
# that writes the step into analysis_state[analysis_id]["workflow_step"].
# Nodes call _publish_step(state, "<step_name>") so frontend /progress shows
# real workflow progression instead of time-based guesses.

from typing import Callable  # noqa: E402

_progress_callbacks: Dict[str, Callable[[str], None]] = {}


def register_progress_callback(analysis_id: str, callback: Callable[[str], None]) -> None:
    """Register a progress callback for an analysis_id."""
    if analysis_id:
        _progress_callbacks[analysis_id] = callback


def unregister_progress_callback(analysis_id: str) -> None:
    """Remove a progress callback when the analysis finishes."""
    _progress_callbacks.pop(analysis_id, None)


def _publish_step(state: "AnalysisState", step: str) -> None:
    """Publish the current workflow step to state + any registered callback."""
    state.workflow_step = step
    analysis_id = (state.metadata or {}).get("analysis_id")
    cb = _progress_callbacks.get(analysis_id) if analysis_id else None
    if cb is None:
        return
    try:
        cb(step)
    except Exception as e:  # noqa: BLE001 - never break the workflow on a UI concern
        logger.debug("Progress callback failed for %s: %s", analysis_id, e)


# ============================================================================
# Node Functions for Workflow
# ============================================================================

async def parse_code_node(state: AnalysisState) -> AnalysisState:
    """Parse and validate code syntax, extract metrics.

    Args:
        state: Current workflow state

    Returns:
        Updated state with parsed code metrics and language
    """
    _publish_step(state, "parsing")
    log_workflow_step("parse_code", {"file": state.file_name})

    try:
        parser = CodeParser()

        # Preprocess code to remove terminal artifacts (prompts, shell output, etc.)
        cleaned_code = parser.preprocess_code(state.code_snippet)
        state.code_snippet = cleaned_code

        # Detect language
        state.language = parser.detect_language(state.code_snippet)
        logger.info("Detected language: %s", state.language)  # noqa: G201

        # Validate syntax
        is_valid, error_msg = parser.validate_python_syntax(state.code_snippet)
        if not is_valid:
            logger.warning("Syntax validation failed: %s", error_msg)  # noqa: G201
            state.errors.append(f"Syntax error: {error_msg}")

        # Extract metrics
        state.code_metrics = parser.extract_metrics(state.code_snippet)
        logger.info("Extracted metrics: %d functions, %d lines", state.code_metrics.functions, state.code_metrics.total_lines)  # noqa: G201

        state.workflow_step = "select_model"
    except ValueError as e:  # noqa: B014
        logger.error("Error in parse_code_node: %s", e, exc_info=True)  # noqa: G201
        state.errors.append(f"Parse error: {str(e)}")

    return state


async def select_model_node(state: AnalysisState) -> AnalysisState:
    """Agentic model selection based on code complexity and analysis type.

    This node implements agentic reasoning to select the best LLM model
    from available Ollama models based on code characteristics.

    Args:
        state: Current workflow state

    Returns:
        Updated state with selected model and reasoning
    """
    log_workflow_step("select_model", {"language": state.language, "code_size": state.code_metrics.total_lines if state.code_metrics else 0})
    _publish_step(state, "model_selection")

    try:
        # Get available models from Ollama
        client = OllamaClient()
        try:
            available_models = client.get_available_models()
            state.available_models = available_models
            logger.info("Available models: %s", available_models)  # noqa: G201
        except Exception as e:  # noqa: B014
            logger.warning("Could not fetch models from Ollama: %s", e)  # noqa: G201
            available_models = ["llama3:8b", "mistral:7b", "codellama:13b"]
            state.available_models = available_models

        # If the user explicitly requested a model, honour it and skip agentic selection
        if state.model:
            state.model_reasoning = f"User-specified model: {state.model}"
            logger.info("Using user-specified model: %s", state.model)  # noqa: G201
        else:
            # Agentic model selection based on code characteristics
            selected_model = _select_best_model(
                code_size=state.code_metrics.total_lines if state.code_metrics else 0,
                language=str(state.language) if state.language else "python",
                available_models=available_models
            )

            state.model = selected_model
            state.model_reasoning = f"Selected {selected_model} for code size {state.code_metrics.total_lines if state.code_metrics else 0} lines ({state.language})"

        logger.info("Model selected: %s (%s)", state.model, state.model_reasoning)  # noqa: G201
        state.workflow_step = "chunk_code"
    except ValueError as e:  # noqa: B014
        logger.error("Error in select_model_node: %s", e, exc_info=True)  # noqa: G201
        state.errors.append(f"Model selection error: {str(e)}")
        # Fallback to default model
        state.model = "llama3:8b"
        state.model_reasoning = "Fallback to default model due to error"

    return state


def _select_best_model(code_size: int, language: str, available_models: List[str]) -> str:
    """Agentic reasoning for model selection based on code characteristics.

    Implements intelligent model selection:
    - For small code (<500 lines): Use faster, smaller models
    - For large code: Use more capable, larger models
    - For specific languages: Prefer specialized models

    Args:
        code_size: Total lines of code
        language: Programming language
        available_models: List of available Ollama models

    Returns:
        Selected model name
    """
    logger.info("Agent reasoning: Selecting model for %d lines of %s code", code_size, language)  # noqa: G201

    # Prioritized model preferences (from config)
    model_priority = dict(MODEL_SELECTION_SCORE)
    model_priority.update({
        "neural-chat": 75,
        "orca-mini": 70,
    })

    # Score models based on availability and priority
    scored_models = []
    for model in available_models:
        priority = 0
        reasoning = []

        # Check for code specialization
        if "code" in model.lower():
            priority += 50
            reasoning.append("code-specialized")

        # Check for language specificity
        if language.lower() in model.lower():
            priority += 30
            reasoning.append(f"specialized-for-{language}")

        # Get base priority
        for key, score in model_priority.items():
            if key in model.lower():
                priority += score
                reasoning.append(f"base-priority-{score}")
                break

        # Adjust based on code size
        if code_size > 1000:
            if "13b" in model or "70b" in model:
                priority += 40
                reasoning.append("large-model-for-large-code")
        elif code_size < 100:
            if "7b" in model or "8b" in model or "orca" in model:
                priority += 30
                reasoning.append("small-model-for-small-code")

        scored_models.append((model, priority, " + ".join(reasoning)))

    # Sort by priority and select best
    scored_models.sort(key=lambda x: x[1], reverse=True)

    if scored_models:
        best_model, score, reasoning = scored_models[0]
        logger.info("Agent reasoning: Selected %s (score: %d, reasoning: %s)", best_model, score, reasoning)  # noqa: G201
        return best_model

    # Fallback
    logger.warning("No models found, using hardcoded default")  # noqa: G201
    return "llama3:8b"


async def chunk_code_node(state: AnalysisState) -> AnalysisState:
    """Split large code into chunks for parallel processing.

    When RAG is enabled: Analyze entire code holistically with RAG context
    When RAG is disabled: Chunk only for very large files (>1000 lines)

    Args:
        state: Current workflow state

    Returns:
        Updated state with code chunks
    """
    log_workflow_step("chunk_code", {"total_lines": state.code_metrics.total_lines if state.code_metrics else 0})
    _publish_step(state, "chunking")

    try:
        parser = CodeParser()
        total_lines = state.code_metrics.total_lines if state.code_metrics else 0

        # When using RAG: Keep entire code as single chunk for holistic analysis
        if state.use_rag:
            state.chunks = [state.code_snippet]
            logger.info("RAG mode: Analyzing entire code (%d lines) as single chunk", total_lines)  # noqa: G201
        # For small/medium code: use single chunk (threshold: 500 lines)
        elif total_lines < 500:
            state.chunks = [state.code_snippet]
            logger.info("Code size %d lines < 500: Using single chunk", total_lines)  # noqa: G201
        else:
            # Split into functions only for very large code (>500 lines)
            # split_into_functions returns List[Tuple[name, code, line_start, line_end]]
            # Extract just the code snippets (second element)
            function_chunks = parser.split_into_functions(state.code_snippet)
            state.chunks = [code_snippet for _, code_snippet, _, _ in function_chunks]
            logger.info("Large code (%d lines): Split into %d function chunks", total_lines, len(state.chunks))  # noqa: G201

        if not state.chunks:
            state.chunks = [state.code_snippet]

        logger.info("Chunk configuration: %d chunks", len(state.chunks))  # noqa: G201

        # Log chunk details
        for idx, chunk in enumerate(state.chunks):
            chunk_lines = len(chunk.split('\n'))
            chunk_chars = len(chunk)
            logger.info("[Chunk %d/%d] Size: %d lines, %d characters", idx + 1, len(state.chunks), chunk_lines, chunk_chars)  # noqa: G201

        state.workflow_step = "retrieve_context"
    except ValueError as e:  # noqa: B014
        logger.error("Error in chunk_code_node: %s", e, exc_info=True)  # noqa: G201
        state.errors.append(f"Chunking error: {str(e)}")
        state.chunks = [state.code_snippet]

    return state


async def retrieve_context_node(state: AnalysisState) -> AnalysisState:
    """Retrieve relevant examples from knowledge base using RAG.

    Args:
        state: Current workflow state

    Returns:
        Updated state with RAG context
    """
    log_workflow_step("retrieve_context", {"chunk_count": len(state.chunks)})
    _publish_step(state, "rag_retrieval")

    try:
        retriever = RAGRetriever()

        # For first chunk, retrieve context
        if state.chunks:
            context = await retriever.find_relevant_examples(
                state.chunks[0],
                smell_type="Long Method",
                top_k=5
            )

            # Convert to serializable format
            state.rag_context = {
                "examples": [str(ex) for ex in context] if context else [],
                "count": len(context) if context else 0,
                "retrieved_at": datetime.now().isoformat()
            }
            logger.info("Retrieved %d context examples", state.rag_context['count'])  # noqa: G201
    except ValueError as e:  # noqa: B014
        logger.error("Error in retrieve_context_node: %s", e, exc_info=True)  # noqa: G201
        state.errors.append(f"RAG retrieval error: {str(e)}")
        state.rag_context = {"examples": [], "count": 0}

    state.workflow_step = "detect_smells"
    return state


async def _analyze_chunk_for_all_smells(chunk: str, chunk_idx: int, total_chunks: int,
                                        rag_context: Dict[str, Any], model: str) -> tuple:
    """Analyze a single chunk for ALL code smell types in parallel.

    Args:
        chunk: Code chunk to analyze
        chunk_idx: Index of this chunk (1-based for logging)
        total_chunks: Total number of chunks
        rag_context: RAG context for this analysis
        model: Model to use for analysis

    Returns:
        Tuple of (chunk_idx, findings_list)
    """
    chunk_lines = len(chunk.split('\n'))
    chunk_chars = len(chunk)
    logger.info("[Parallel Chunk %d/%d] Size: %d lines, %d chars | Starting analysis for ALL smell types...",
                chunk_idx, total_chunks, chunk_lines, chunk_chars)  # noqa: G201

    detector = CodeSmellDetector(
        specialization=f"smell-detector-chunk-{chunk_idx}",
        rag_retriever=RAGRetriever() if rag_context.get("count", 0) > 0 else None,
        model=model
    )

    # Analyze for ALL canonical smell types (comprehensive coverage)
    findings = await detector.detect_smells(
        chunk,
        context=rag_context,
        smell_types=None  # None means analyze ALL types in catalog
    )

    logger.info("[Parallel Chunk %d/%d] Analysis complete: %d smells detected across all types",
                chunk_idx, total_chunks, len(findings))  # noqa: G201

    # Log smell type distribution for this chunk
    if findings:
        smell_types = {}
        for finding in findings:
            smell_type = finding.smell_type
            smell_types[smell_type] = smell_types.get(smell_type, 0) + 1

        types_summary = ", ".join([f"{stype}({cnt})" for stype, cnt in sorted(smell_types.items())])
        logger.info("[Parallel Chunk %d/%d] Smell types detected: %s", chunk_idx, total_chunks, types_summary)  # noqa: G201

    return (chunk_idx, findings)


async def detect_smells_node(state: AnalysisState) -> AnalysisState:
    """Detect code smells in code chunks using LLM with selected model.

    Analyzes all chunks in PARALLEL to ensure complete code coverage and comprehensive
    smell type detection. When RAG is enabled, the entire code is kept as a single chunk
    for holistic analysis.

    Features:
    - Parallel processing of multiple chunks for faster analysis
    - Comprehensive analysis of ALL code smell types per chunk
    - Detailed logging of smell type distribution
    - Aggregates findings across all chunks

    Args:
        state: Current workflow state

    Returns:
        Updated state with detected findings
    """
    log_workflow_step("detect_smells", {"chunk_count": len(state.chunks), "model": state.model, "mode": "parallel"})
    _publish_step(state, "inference")

    try:
        if not state.chunks:
            logger.warning("No chunks to analyze")
            state.workflow_step = "validate_findings"
            return state

        # Create parallel analysis tasks for all chunks
        logger.info("Starting parallel analysis of %d chunk(s) for ALL code smell types using model: %s",
                    len(state.chunks), state.model)  # noqa: G201

        analysis_tasks = []
        for chunk_idx, chunk in enumerate(state.chunks, start=1):
            task = _analyze_chunk_for_all_smells(
                chunk=chunk,
                chunk_idx=chunk_idx,
                total_chunks=len(state.chunks),
                rag_context=state.rag_context,
                model=state.model
            )
            analysis_tasks.append(task)

        # Execute all chunk analyses in parallel
        results = await asyncio.gather(*analysis_tasks, return_exceptions=True)

        # Aggregate results and track coverage
        total_findings = 0
        smell_type_coverage = {}
        chunk_results = {}

        for result in results:
            if isinstance(result, Exception):
                logger.error("Error during parallel chunk analysis: %s", result, exc_info=True)  # noqa: G201
                state.errors.append(f"Parallel detection error: {str(result)}")
                continue

            chunk_idx, findings = result
            chunk_results[chunk_idx] = len(findings)
            state.detections.extend(findings)
            total_findings += len(findings)

            # Track smell type coverage
            for finding in findings:
                smell_type = finding.smell_type
                if smell_type not in smell_type_coverage:
                    smell_type_coverage[smell_type] = 0
                smell_type_coverage[smell_type] += 1

        # Log comprehensive analysis summary
        logger.info("Parallel chunk analysis summary: %d total smells detected using %s across %d chunk(s)",
                    total_findings, state.model, len(state.chunks))  # noqa: G201

        # Log coverage by chunk
        if len(state.chunks) > 1:
            chunk_summary = ", ".join([f"Chunk {idx}: {count} smells"
                                      for idx, count in sorted(chunk_results.items())])
            logger.info("Per-chunk breakdown: %s", chunk_summary)  # noqa: G201

        # Log smell type coverage (which types were found)
        if smell_type_coverage:
            coverage_summary = ", ".join([f"{stype}({cnt})"
                                         for stype, cnt in sorted(smell_type_coverage.items())])
            logger.info("Smell types detected: %s", coverage_summary)  # noqa: G201

        logger.info("Analysis coverage: %d unique smell types identified", len(smell_type_coverage))  # noqa: G201

    except ValueError as e:  # noqa: B014
        logger.error("Error in detect_smells_node: %s", e, exc_info=True)  # noqa: G201
        state.errors.append(f"Detection error: {str(e)}")
    except Exception as e:  # noqa: B014
        logger.error("Unexpected error in detect_smells_node: %s", e, exc_info=True)  # noqa: G201
        state.errors.append(f"Unexpected detection error: {str(e)}")

    state.workflow_step = "validate_findings"
    return state


async def validate_findings_node(state: AnalysisState) -> AnalysisState:
    """Validate findings and filter false positives.

    Args:
        state: Current workflow state

    Returns:
        Updated state with validated findings
    """
    log_workflow_step("validate_findings", {"findings_count": len(state.detections)})
    _publish_step(state, "validation")

    try:
        validator = QualityValidator()

        if state.detections:
            validated = validator.validate_findings(state.detections)
            state.validated_findings = validated
            logger.info("Validated %d findings (from %d detections)", len(validated), len(state.detections))  # noqa: G201
    except ValueError as e:  # noqa: B014
        logger.error("Error in validate_findings_node: %s", e, exc_info=True)  # noqa: G201
        state.errors.append(f"Validation error: {str(e)}")
        state.validated_findings = state.detections

    state.workflow_step = "aggregate_results"
    return state


async def aggregate_results_node(state: AnalysisState) -> AnalysisState:
    """Aggregate findings into final results.

    Args:
        state: Current workflow state

    Returns:
        Updated state with aggregated results
    """
    log_workflow_step("aggregate_results", {
        "validated_findings": len(state.validated_findings),
        "errors": len(state.errors)
    })
    _publish_step(state, "aggregating")

    try:
        # Create DetectionResult from validated findings
        severity_counts = {
            "critical": sum(1 for f in state.validated_findings if f.severity == SeverityLevel.CRITICAL),
            "high": sum(1 for f in state.validated_findings if f.severity == SeverityLevel.HIGH),
            "medium": sum(1 for f in state.validated_findings if f.severity == SeverityLevel.MEDIUM),
            "low": sum(1 for f in state.validated_findings if f.severity == SeverityLevel.LOW),
        }

        logger.info("Final results: %s", severity_counts)  # noqa: G201

        state.metadata["severity_counts"] = severity_counts
        state.metadata["total_findings"] = len(state.validated_findings)
        state.metadata["execution_time"] = (datetime.now() - state.start_time).total_seconds()
    except ValueError as e:  # noqa: B014
        logger.error("Error in aggregate_results_node: %s", e, exc_info=True)  # noqa: G201
        state.errors.append(f"Aggregation error: {str(e)}")

    state.workflow_step = "complete"
    return state


# ============================================================================
# Conditional Routing
# ============================================================================

def route_based_on_chunks(state: AnalysisState) -> str:
    """Route workflow based on number of chunks.

    For simplicity, we process only the first chunk. In production,
    this would use LangGraph's Send() for parallel processing.

    Args:
        state: Current workflow state

    Returns:
        Next node name or END
    """
    if len(state.chunks) == 0 or state.current_chunk_idx >= len(state.chunks):
        return "aggregate_results"
    return "retrieve_context"


# ============================================================================
# Graph Construction
# ============================================================================

def build_workflow_graph() -> StateGraph:
    """Build LangGraph StateGraph for code smell analysis workflow with agentic model selection.

    The graph follows this structure:

    START
      ↓
    parse_code
      ↓
    select_model (AGENTIC: Chooses best LLM based on code characteristics)
      ↓
    chunk_code
      ↓
    retrieve_context (RAG)
      ↓
    detect_smells (Uses selected model)
      ↓
    validate_findings
      ↓
    aggregate_results
      ↓
    END

    Returns:
        Compiled StateGraph ready for execution
    """
    # Create graph
    graph = StateGraph(AnalysisState)

    # Add nodes
    graph.add_node("parse_code", parse_code_node)
    graph.add_node("select_model", select_model_node)
    graph.add_node("chunk_code", chunk_code_node)
    graph.add_node("retrieve_context", retrieve_context_node)
    graph.add_node("detect_smells", detect_smells_node)
    graph.add_node("validate_findings", validate_findings_node)
    graph.add_node("aggregate_results", aggregate_results_node)

    # Add edges (linear flow for now)
    graph.add_edge(START, "parse_code")
    graph.add_edge("parse_code", "select_model")
    graph.add_edge("select_model", "chunk_code")
    graph.add_edge("chunk_code", "retrieve_context")
    graph.add_edge("retrieve_context", "detect_smells")
    graph.add_edge("detect_smells", "validate_findings")
    graph.add_edge("validate_findings", "aggregate_results")
    graph.add_edge("aggregate_results", END)

    # Compile graph
    compiled_graph = graph.compile()

    logger.info("LangGraph workflow compiled successfully with agentic model selection")
    return compiled_graph


# ============================================================================
# Workflow Execution
# ============================================================================

class WorkflowExecutor:
    """Executor for code smell analysis workflow.

    Wraps the compiled LangGraph for easy invocation and result handling.
    """

    def __init__(self):
        """Initialize workflow executor."""
        self.graph = build_workflow_graph()
        logger.info("WorkflowExecutor initialized")

    async def execute(
        self,
        code_snippet: str,
        file_name: str = "code",
        model: Optional[str] = None,
        use_rag: bool = True,
        analysis_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute workflow for code analysis with optional model and RAG selection.

        Args:
            code_snippet: Code to analyze
            file_name: Source file name for context
            model: Optional specific LLM model to use (if None, agent will select)
            use_rag: Whether to use RAG context (default: True)
            analysis_id: Optional API-layer identifier so each node can publish
                progress back to the FastAPI /progress/{id} endpoint.

        Returns:
            Dictionary containing:
                - detections: List of CodeSmellFinding
                - validated_findings: List of validated CodeSmellFinding
                - metadata: Execution metadata including model_used
                - errors: List of encountered errors
        """
        log_agent_event("workflow_executor", "execution_start", {
            "file": file_name,
            "code_length": len(code_snippet),
            "model": model or "auto-select",
            "use_rag": use_rag
        })

        # Create initial state
        initial_state = AnalysisState(
            code_snippet=code_snippet,
            file_name=file_name,
            model=model,  # Pass specified model (or None for auto-selection)
            use_rag=use_rag  # Pass RAG preference
        )
        if analysis_id:
            initial_state.metadata["analysis_id"] = analysis_id

        try:
            # Execute workflow
            final_state = await self.graph.ainvoke(
                initial_state,
                config={"recursion_limit": 100}
            )
            log_agent_event(
                "workflow_executor",
                "execution_complete",
                {
                    "findings_count": len(final_state.get("validated_findings", [])),
                    "errors_count": len(final_state.get("errors", [])),
                    "execution_time": final_state.get("metadata", {}).get("execution_time", 0),
                    "model_used": final_state.get("model"),
                    "model_reasoning": final_state.get("model_reasoning")
                }
            )

            return {
                "code_snippet": final_state.get("code_snippet"),
                "file_name": final_state.get("file_name"),
                "language": final_state.get("language"),
                "detections": final_state.get("detections"),
                "validated_findings": final_state.get("validated_findings", []),
                "metadata": {
                    **final_state.get("metadata", {}),
                    "model_used": final_state.get("model"),
                    "model_reasoning": final_state.get("model_reasoning"),
                    "available_models": final_state.get("available_models"),
                    "use_rag": final_state.get("use_rag"),
                    "execution_time": (datetime.now() - initial_state.start_time).total_seconds(),
                },
                "errors": final_state.get("errors", [])
            }
        except ValueError as e:  # noqa: B014
            logger.error("Workflow execution error: %s", e, exc_info=True)  # noqa: G201
            log_agent_event("workflow_executor", "execution_error", {"error": str(e)})

            return {
                "code_snippet": code_snippet,
                "file_name": file_name,
                "language": None,
                "detections": [],
                "validated_findings": [],
                "metadata": {
                    "execution_time": (datetime.now() - initial_state.start_time).total_seconds(),
                    "model_used": model or "failed-to-select",
                    "available_models": [],
                    "use_rag": use_rag
        },
                "errors": [str(e)]
            }


# ============================================================================
# Test Function
# ============================================================================

async def test_workflow_graph():
    """Test the LangGraph workflow with sample code."""
    print("Testing LangGraph Workflow...")

    test_code = """
def process_data(items, config, cache, database, logger, metrics):
    result = []
    for i in range(len(items)):
        if items[i].is_valid():
            processed = items[i].process()
            result.append(processed)
            logger.log(f"Processed {i}")
            metrics.increment()
            cache.set(f"item_{i}", processed)
            database.save(processed)
            if len(result) > 1000:
                logger.flush()
    return result

class DataProcessor:
    def __init__(self):
        self.items = []
        self.cache = {}
        self.database = None
        self.logger = None
        self.metrics = None
        self.config = {}

    def process(self):
        pass

    def validate(self):
        pass

    def report(self):
        pass
    """

    executor = WorkflowExecutor()
    result = await executor.execute(test_code, "test.py")

    print("\n✓ Workflow executed successfully")
    print(f"  Detections: {len(result['detections'])}")
    print(f"  Validated: {len(result['validated_findings'])}")
    print(f"  Errors: {len(result['errors'])}")
    print(f"  Language: {result['language']}")
    print(f"  Execution time: {result['metadata'].get('execution_time', 0):.2f}s")


if __name__ == "__main__":
    asyncio.run(test_workflow_graph())
