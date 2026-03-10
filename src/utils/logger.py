"""
Structured Logging for Multi-Agent System
Logging all agent activities, LLM interactions, and workflow steps in JSON format.

Architecture: Centralized logging infrastructure for reproducibility and debugging
"""

import logging
import json
import sys
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path

from config import RESULTS_DIR

logger = logging.getLogger(__name__)


class JSONFormatter:
    """JSON formatter for structured logging."""

    @staticmethod
    def format(record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add extra fields if present
        if hasattr(record, "extra"):
            log_data.update(record.extra)

        return json.dumps(log_data)


def setup_logging(
    log_dir: Path = RESULTS_DIR / "logs",
    log_name: str = "analysis",
) -> Path:
    """
    Setup structured logging configuration.

    Args:
        log_dir: Directory for log files
        log_name: Base name for log file

    Returns:
        Path to log file
    """
    log_dir.mkdir(parents=True, exist_ok=True)

    # Timestamp for unique log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{log_name}_{timestamp}.json"

    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Remove existing handlers
    root_logger.handlers.clear()

    # Console handler (INFO level)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File handler (DEBUG level, JSON format)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    json_formatter = JSONFormatter()
    file_handler.setFormatter(json_formatter)
    root_logger.addHandler(file_handler)

    logger.info("Logging configured, writing to %s", log_file)  # noqa: G201

    return log_file


def log_agent_event(
    agent_name: str,
    event_type: str,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Log an agent event.

    Args:
        agent_name: Name of the agent
        event_type: Type of event (e.g., "detection_start", "detection_complete")
        details: Optional event details
    """
    message = f"[{agent_name}] {event_type}"
    extra_data = {
        "agent": agent_name,
        "event_type": event_type,
    }

    if details:
        extra_data.update(details)

    logger.info(message, extra={'extra': extra_data})


def log_llm_request(
    agent_name: str,
    model: str,
    prompt_length: int,
    prompt_preview: str,
    temperature: float = 0.1,
) -> None:
    """
    Log an LLM request.

    Args:
        agent_name: Agent making the request
        model: Model name
        prompt_length: Length of prompt
        prompt_preview: First 100 chars of prompt
        temperature: Temperature parameter
    """
    logger.info(
        "[%s] LLM request",  # noqa: G201
        agent_name,
        extra={
            'extra': {
                'agent': agent_name,
                'model': model,
                'prompt_length': prompt_length,
                'prompt_preview': prompt_preview[:100],
                'temperature': temperature,
            }
        },
    )


def log_llm_response(
    agent_name: str,
    model: str,
    response_length: int,
    tokens_generated: int,
    latency_seconds: float,
    cached: bool = False,
) -> None:
    """
    Log an LLM response.

    Args:
        agent_name: Agent that made the request
        model: Model name
        response_length: Length of response
        tokens_generated: Number of tokens generated
        latency_seconds: Response latency in seconds
        cached: Whether response was from cache
    """
    logger.info(
        "[%s] LLM response %s",  # noqa: G201
        agent_name,
        "(cached)" if cached else "",
        extra={
            'extra': {
                'agent': agent_name,
                'model': model,
                'response_length': response_length,
                'tokens_generated': tokens_generated,
                'latency_seconds': latency_seconds,
                'cached': cached,
            }
        },
    )


def log_detection_result(
    agent_name: str,
    file_name: str,
    smells_found: int,
    critical: int,
    high: int,
    medium: int,
    low: int,
    processing_time: float,
) -> None:
    """
    Log detection results from an agent.

    Args:
        agent_name: Name of detecting agent
        file_name: File being analyzed
        smells_found: Total smells found
        critical: Number of critical smells
        high: Number of high smells
        medium: Number of medium smells
        low: Number of low smells
        processing_time: Processing time in seconds
    """
    logger.info(
        "[%s] Detection complete: %d smells",  # noqa: G201
        agent_name,
        smells_found,
        extra={
            'extra': {
                'agent': agent_name,
                'file': file_name,
                'total_smells': smells_found,
                'critical': critical,
                'high': high,
                'medium': medium,
                'low': low,
                'processing_time_seconds': processing_time,
            }
        },
    )


def log_rag_retrieval(
    agent_name: str,
    query: str,
    results_count: int,
    top_similarity: float,
    retrieval_time: float,
    cached: bool = False,
) -> None:
    """
    Log RAG retrieval operation.

    Args:
        agent_name: Agent performing retrieval
        query: Query used
        results_count: Number of results
        top_similarity: Top similarity score
        retrieval_time: Time taken in seconds
        cached: Whether results were cached
    """
    logger.info(
        "[%s] RAG retrieval %s: %d results",  # noqa: G201
        agent_name,
        "(cached)" if cached else "",
        results_count,
        extra={
            'extra': {
                'agent': agent_name,
                'query': query[:100],
                'results_count': results_count,
                'top_similarity': top_similarity,
                'retrieval_time_seconds': retrieval_time,
                'cached': cached,
            }
        },
    )


def log_workflow_step(
    step_name: str,
    status: str,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Log a workflow step.

    Args:
        step_name: Name of the step
        status: Status (start, complete, error)
        details: Optional details
    """
    message = f"[Workflow] {step_name}: {status}"
    extra_data = {
        'workflow_step': step_name,
        'status': status,
    }

    if details:
        extra_data.update(details)

    if status == "error":
        logger.error(message, extra={'extra': extra_data})
    else:
        logger.info(message, extra={'extra': extra_data})


def log_error(
    agent_name: str,
    error_type: str,
    error_message: str,
    context: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Log an error.

    Args:
        agent_name: Agent where error occurred
        error_type: Type of error
        error_message: Error message
        context: Optional context
    """
    extra_data = {
        'agent': agent_name,
        'error_type': error_type,
        'error_message': error_message,
    }

    if context:
        extra_data.update(context)

    logger.error(
        "[%s] %s: %s",  # noqa: G201
        agent_name,
        error_type,
        error_message,
        extra={'extra': extra_data},
    )


def log_metric(
    metric_name: str,
    metric_value: float,
    tags: Optional[Dict[str, str]] = None,
) -> None:
    """
    Log a metric.

    Args:
        metric_name: Name of metric
        metric_value: Metric value
        tags: Optional metric tags
    """
    extra_data = {
        'metric_name': metric_name,
        'metric_value': metric_value,
    }

    if tags:
        extra_data.update(tags)

    logger.info(
        "Metric: %s = %s",  # noqa: G201
        metric_name,
        metric_value,
        extra={'extra': extra_data},
    )


# Test function
def test_logging():
    """Test logging setup."""
    print("✓ Testing logging setup...")

    log_file = setup_logging(log_name="test_logging")
    print(f"  Log file: {log_file}")

    # Log some test events
    log_agent_event("detector_long_method", "detection_start")
    log_llm_request(
        "detector_long_method",
        "llama3:8b",
        200,
        "def long_method..."
    )
    log_llm_response(
        "detector_long_method",
        "llama3:8b",
        150,
        50,
        3.2
    )
    log_detection_result(
        "detector_long_method",
        "example.py",
        1,
        0,
        1,
        0,
        0,
        2.5
    )
    log_workflow_step("code_analysis", "start", {"file": "example.py"})
    log_workflow_step("code_analysis", "complete", {"smells_found": 3})

    print(f"✓ Logging test complete, check {log_file}")


if __name__ == "__main__":
    test_logging()
