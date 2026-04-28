"""
Colored logging utility for better console output.
Provides ANSI color codes and formatted logging for different event types.
"""

import logging
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """Custom formatter with ANSI color codes for different log levels."""

    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",   # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
        "INFERENCE": "\033[1;36m",  # Bright Cyan
        "SUCCESS": "\033[1;32m",  # Bright Green
        "HIGHLIGHT": "\033[1;35m",  # Bright Magenta
    }
    RESET = "\033[0m"

    def format(self, record):
        """Format log record with color codes."""
        if not hasattr(record, "color"):
            record.color = self.COLORS.get(record.levelname, "")
        if not hasattr(record, "reset"):
            record.reset = self.RESET

        # Use custom format with color
        if record.levelname in self.COLORS:
            log_fmt = (
                f"{record.color}[{record.levelname:<8}]{self.RESET} "
                f"%(message)s"
            )
            formatter = logging.Formatter(log_fmt)
            return formatter.format(record)

        return super().format(record)


def setup_colored_logging(
    logger: logging.Logger,
    level: int = logging.INFO,
    handler: Optional[logging.Handler] = None,
) -> None:
    """
    Setup colored logging for a logger.

    Args:
        logger: Logger instance to configure
        level: Logging level
        handler: Optional custom handler (defaults to StreamHandler)
    """
    if handler is None:
        handler = logging.StreamHandler()

    handler.setLevel(level)
    formatter = ColoredFormatter()
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.setLevel(level)


def get_colored_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Get a colored logger instance.

    Args:
        name: Logger name
        level: Logging level

    Returns:
        Configured logger with colored output
    """
    logger = logging.getLogger(name)
    setup_colored_logging(logger, level)
    return logger


def log_inference_start(
    logger: logging.Logger,
    analysis_id: str,
    code_lines: int,
    use_rag: bool,
    model: str,
) -> None:
    """Log inference start with color highlighting."""
    logger.info(
        f"🔍 [INFERENCE] Analyzing {code_lines} lines | "
        f"ID: {analysis_id[:8]}... | RAG: {use_rag} | Model: {model}",
    )


def log_inference_end(
    logger: logging.Logger,
    analysis_id: str,
    findings_count: int,
    f1_score: Optional[float],
    time_ms: float,
    model: str,
) -> None:
    """Log inference completion with results."""
    f1_str = f"{f1_score:.3f}" if f1_score is not None else "N/A"
    logger.info(
        f"✅ [INFERENCE] Complete | "
        f"Findings: {findings_count} | F1: {f1_str} | "
        f"Time: {time_ms:.1f}ms | Model: {model}",
    )


def log_inference_error(
    logger: logging.Logger,
    analysis_id: str,
    error_msg: str,
    time_ms: float,
) -> None:
    """Log inference error with details."""
    logger.error(
        f"❌ [INFERENCE] Failed | Error: {error_msg} | Time: {time_ms:.1f}ms | "
        f"ID: {analysis_id[:8]}...",
    )


def log_rag_retrieval(
    logger: logging.Logger,
    query_type: str,
    retrieved_count: int,
    relevance_scores: Optional[list] = None,
) -> None:
    """Log RAG retrieval details."""
    if relevance_scores:
        avg_score = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0
        logger.info(
            f"📚 [RAG] Retrieved {retrieved_count} documents | "
            f"Type: {query_type} | Avg relevance: {avg_score:.3f}",
        )
    else:
        logger.info(
            f"📚 [RAG] Retrieved {retrieved_count} documents | Type: {query_type}",
        )


def log_model_selection(logger: logging.Logger, model: str, reason: str) -> None:
    """Log LLM model selection."""
    logger.info(f"🤖 [MODEL] Selected: {model} | Reason: {reason}")
