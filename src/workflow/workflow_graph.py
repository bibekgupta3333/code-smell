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

logger = logging.getLogger(__name__)


# ============================================================================
# State Definition
# ============================================================================

class AnalysisState(BaseModel):
    """State for code smell analysis workflow.

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

        state.workflow_step = "chunk_code"
    except ValueError as e:  # noqa: B014
        logger.error("Error in parse_code_node: %s", e, exc_info=True)  # noqa: G201
        state.errors.append(f"Parse error: {str(e)}")

    return state


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
            state.chunks = parser.split_into_functions(state.code_snippet)

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
    """Detect code smells in code chunks using LLM.

    Args:
        state: Current workflow state

    Returns:
        Updated state with detected findings
    """
    log_workflow_step("detect_smells", {"chunk_count": len(state.chunks)})

    try:
        detector = CodeSmellDetector(
            specialization="multi-detector",
            rag_retriever=RAGRetriever() if state.rag_context.get("count", 0) > 0 else None
        )

        # Detect smells in first chunk (parallel processing would use LangGraph's Send)
        if state.chunks:
            findings = await detector.detect_smells(
                state.chunks[0],
                context=state.rag_context
            )
            state.detections.extend(findings)
            logger.info("Detected %d code smells", len(findings))  # noqa: G201
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
            validated = await validator.validate_findings(state.detections)
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
    """Build LangGraph StateGraph for code smell analysis workflow.

    The graph follows this structure:

    START
      ↓
    parse_code
      ↓
    chunk_code
      ↓
    retrieve_context
      ↓
    detect_smells
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
    graph.add_node("chunk_code", chunk_code_node)
    graph.add_node("retrieve_context", retrieve_context_node)
    graph.add_node("detect_smells", detect_smells_node)
    graph.add_node("validate_findings", validate_findings_node)
    graph.add_node("aggregate_results", aggregate_results_node)

    # Add edges (linear flow for now)
    graph.add_edge(START, "parse_code")
    graph.add_edge("parse_code", "chunk_code")
    graph.add_edge("chunk_code", "retrieve_context")
    graph.add_edge("retrieve_context", "detect_smells")
    graph.add_edge("detect_smells", "validate_findings")
    graph.add_edge("validate_findings", "aggregate_results")
    graph.add_edge("aggregate_results", END)

    # Compile graph
    compiled_graph = graph.compile()

    logger.info("LangGraph workflow compiled successfully")
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
        file_name: str = "code"
    ) -> Dict[str, Any]:
        """Execute workflow for code analysis.

        Args:
            code_snippet: Code to analyze
            file_name: Source file name for context

        Returns:
            Dictionary containing:
                - detections: List of CodeSmellFinding
                - validated_findings: List of validated CodeSmellFinding
                - metadata: Execution metadata
                - errors: List of encountered errors
        """
        log_agent_event("workflow_executor", "execution_start", {"file": file_name, "code_length": len(code_snippet)})

        # Create initial state
        initial_state = AnalysisState(
            code_snippet=code_snippet,
            file_name=file_name
        )

        try:
            # Execute workflow
            final_state = await asyncio.to_thread(
                self.graph.invoke,
                initial_state,
                config={"recursion_limit": 100}
            )

            # Log completion
            log_agent_event(
                "workflow_executor",
                "execution_complete",
                {
                    "findings_count": len(final_state.validated_findings),
                    "errors_count": len(final_state.errors),
                    "execution_time": final_state.metadata.get("execution_time", 0)
                }
            )

            return {
                "code_snippet": final_state.code_snippet,
                "file_name": final_state.file_name,
                "language": final_state.language,
                "detections": final_state.detections,
                "validated_findings": final_state.validated_findings,
                "metadata": final_state.metadata,
                "errors": final_state.errors
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
                "metadata": {"execution_time": (datetime.now() - initial_state.start_time).total_seconds()},
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
