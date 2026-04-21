"""
Dependency injection setup for FastAPI.
Provides singletons and factories for database, workflow graph, and logging.
"""

import logging
from typing import Generator, Optional
from functools import lru_cache

from src.api.config import config
from src.api.validators import RequestValidator, RateLimiter


# ============================================================================
# Logger Setup
# ============================================================================

def get_logger(name: str) -> logging.Logger:
    """
    Get or create logger with standardized configuration.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Configure if not already configured
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(getattr(logging, config.LOG_LEVEL, logging.INFO))

    return logger


# ============================================================================
# Singleton Instances
# ============================================================================

@lru_cache(maxsize=1)
def get_request_validator() -> RequestValidator:
    """
    Get request validator singleton.

    Returns:
        RequestValidator instance
    """
    logger = get_logger(__name__)
    logger.info("Initializing request validator")
    return RequestValidator()


@lru_cache(maxsize=1)
def get_rate_limiter() -> RateLimiter:
    """
    Get rate limiter singleton.

    Returns:
        RateLimiter instance
    """
    logger = get_logger(__name__)
    logger.info("Initializing rate limiter")
    return RateLimiter()


@lru_cache(maxsize=1)
def get_database_manager():
    """
    Get database manager singleton.

    Returns:
        DatabaseManager instance

    Raises:
        ImportError: If database module not available
    """
    try:
        from src.database.database_manager import DatabaseManager

        logger = get_logger(__name__)
        logger.info(f"Initializing database manager with {config.DATABASE_URL}")
        return DatabaseManager()
    except ImportError as e:
        logger = get_logger(__name__)
        logger.error(f"Failed to import DatabaseManager: {str(e)}")
        raise


@lru_cache(maxsize=1)
def get_workflow_graph():
    """
    Get workflow graph singleton.

    Returns:
        WorkflowGraph instance or None if not available
    """
    try:
        from src.workflow.workflow_graph import WorkflowGraph

        logger = get_logger(__name__)
        logger.info("Initializing workflow graph")
        return WorkflowGraph()
    except (ImportError, AttributeError) as e:
        logger = get_logger(__name__)
        logger.debug(f"WorkflowGraph not available (optional): {str(e)}")
        return None


@lru_cache(maxsize=1)
def get_rag_manager():
    """
    Get RAG manager singleton.

    Returns:
        RAGManager instance

    Raises:
        ImportError: If RAG module not available
    """
    try:
        from src.rag.rag_manager import RAGManager

        logger = get_logger(__name__)
        logger.info("Initializing RAG manager")
        return RAGManager()
    except ImportError as e:
        logger = get_logger(__name__)
        logger.warning(f"RAG manager not available: {str(e)}")
        return None


@lru_cache(maxsize=1)
def get_detector():
    """
    Get code smell detector singleton.

    Returns:
        Detector instance or None if not available
    """
    try:
        from src.core.detector import Detector

        logger = get_logger(__name__)
        logger.info("Initializing code smell detector")
        return Detector()
    except (ImportError, AttributeError) as e:
        logger = get_logger(__name__)
        logger.debug(f"Detector not available (optional): {str(e)}")
        return None


# ============================================================================
# FastAPI Dependency Injections
# ============================================================================

def get_request_validator_dep() -> RequestValidator:
    """
    FastAPI dependency for request validator.

    Returns:
        RequestValidator instance
    """
    return get_request_validator()


def get_rate_limiter_dep() -> RateLimiter:
    """
    FastAPI dependency for rate limiter.

    Returns:
        RateLimiter instance
    """
    return get_rate_limiter()


def get_logger_dep() -> logging.Logger:
    """
    FastAPI dependency for logger.

    Returns:
        Logger instance
    """
    return get_logger(__name__)


def get_database_session() -> Generator:
    """
    FastAPI dependency for database session.

    Yields:
        Database session

    Note:
        Can be used with Depends() in FastAPI routes
    """
    db_manager = get_database_manager()
    session = db_manager.get_session()
    try:
        yield session
    finally:
        session.close()


def get_workflow_graph_dep():
    """
    FastAPI dependency for workflow graph.

    Returns:
        WorkflowGraph instance
    """
    return get_workflow_graph()


def get_rag_manager_dep():
    """
    FastAPI dependency for RAG manager.

    Returns:
        RAGManager instance
    """
    return get_rag_manager()


def get_detector_dep():
    """
    FastAPI dependency for code detector.

    Returns:
        Detector instance
    """
    return get_detector()


# ============================================================================
# Initialization Utilities
# ============================================================================

def initialize_dependencies() -> dict:
    """
    Initialize all dependencies at startup.

    Returns:
        Dictionary with initialization status
    """
    logger = get_logger(__name__)
    logger.info("Initializing API dependencies...")

    status = {
        "validator": False,
        "rate_limiter": False,
        "database": False,
        "workflow_graph": False,
        "rag_manager": False,
        "detector": False,
    }

    try:
        get_request_validator()
        status["validator"] = True
        logger.info("✓ Request validator initialized")
    except Exception as e:
        logger.error(f"✗ Failed to initialize request validator: {str(e)}")

    try:
        get_rate_limiter()
        status["rate_limiter"] = True
        logger.info("✓ Rate limiter initialized")
    except Exception as e:
        logger.error(f"✗ Failed to initialize rate limiter: {str(e)}")

    try:
        get_database_manager()
        status["database"] = True
        logger.info("✓ Database manager initialized")
    except Exception as e:
        logger.warning(f"⚠ Database manager initialization failed: {str(e)}")

    try:
        wg = get_workflow_graph()
        if wg:
            status["workflow_graph"] = True
            logger.info("✓ Workflow graph initialized")
        else:
            logger.debug("⚠ Workflow graph not available (optional)")
    except Exception as e:
        logger.debug(f"Workflow graph not available (optional): {str(e)}")

    try:
        rag = get_rag_manager()
        if rag:
            status["rag_manager"] = True
            logger.info("✓ RAG manager initialized")
        else:
            logger.warning("⚠ RAG manager not available")
    except Exception as e:
        logger.warning(f"⚠ RAG manager initialization failed: {str(e)}")

    try:
        detector = get_detector()
        if detector:
            status["detector"] = True
            logger.info("✓ Code detector initialized")
        else:
            logger.debug("⚠ Code detector not available (optional)")
    except Exception as e:
        logger.debug(f"Code detector not available (optional): {str(e)}")

    return status


def cleanup_dependencies() -> None:
    """Clean up resources at shutdown."""
    logger = get_logger(__name__)
    logger.info("Cleaning up API dependencies...")

    try:
        db_manager = get_database_manager()
        if hasattr(db_manager, "close"):
            db_manager.close()
            logger.info("✓ Database connection closed")
    except Exception as e:
        logger.warning(f"Error closing database: {str(e)}")

    logger.info("✓ Dependency cleanup complete")
