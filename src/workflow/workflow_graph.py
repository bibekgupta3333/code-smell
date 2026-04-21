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
# Node Functions for Workflow
# ============================================================================

async def parse_code_node(state: AnalysisState) -> AnalysisState:
    """Parse and validate code syntax, extract metrics.

    Args:
        state: Current workflow state

    Returns:
        Updated state with parsed code metrics and language
    """
    log_workflow_step("parse_code", {"file": state.file_name})

    try:
        parser = CodeParser()

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

    # Prioritized model preferences
    model_priority = {
        "codellama": 100,  # Best for code-specific tasks
        "llama3": 85,
        "mistral": 80,
        "neural-chat": 75,
        "orca-mini": 70,
    }

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

    Args:
        state: Current workflow state

    Returns:
        Updated state with code chunks
    """
    log_workflow_step("chunk_code", {"total_lines": state.code_metrics.total_lines if state.code_metrics else 0})

    try:
        parser = CodeParser()

        # For small code, use single chunk
        if state.code_metrics and state.code_metrics.total_lines < 100:
            state.chunks = [state.code_snippet]
        else:
            # Split into functions for larger code
            # split_into_functions returns List[Tuple[name, code, line_start, line_end]]
            # Extract just the code snippets (second element)
            function_chunks = parser.split_into_functions(state.code_snippet)
            state.chunks = [code_snippet for _, code_snippet, _, _ in function_chunks]

        if not state.chunks:
            state.chunks = [state.code_snippet]

        logger.info("Split code into %d chunks", len(state.chunks))  # noqa: G201
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


async def detect_smells_node(state: AnalysisState) -> AnalysisState:
    """Detect code smells in code chunks using LLM with selected model.

    Args:
        state: Current workflow state

    Returns:
        Updated state with detected findings
    """
    log_workflow_step("detect_smells", {"chunk_count": len(state.chunks), "model": state.model})

    try:
        detector = CodeSmellDetector(
            specialization="multi-detector",
            rag_retriever=RAGRetriever() if state.use_rag and state.rag_context.get("count", 0) > 0 else None,
            model=state.model  # Pass selected model to detector
        )

        # Detect smells in first chunk (parallel processing would use LangGraph's Send)
        if state.chunks:
            findings = await detector.detect_smells(
                state.chunks[0],
                context=state.rag_context
            )
            state.detections.extend(findings)
            logger.info("Detected %d code smells using %s", len(findings), state.model)  # noqa: G201
    except ValueError as e:  # noqa: B014
        logger.error("Error in detect_smells_node: %s", e, exc_info=True)  # noqa: G201
        state.errors.append(f"Detection error: {str(e)}")

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
    ) -> Dict[str, Any]:
        """Execute workflow for code analysis with optional model and RAG selection.

        Args:
            code_snippet: Code to analyze
            file_name: Source file name for context
            model: Optional specific LLM model to use (if None, agent will select)
            use_rag: Whether to use RAG context (default: True)

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
                    "use_rag": final_state.get("use_rag")
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
