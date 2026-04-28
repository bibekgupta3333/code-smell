"""
Health and status routes for code smell detection API.
Service health checks and system status monitoring.
"""

from fastapi import APIRouter, HTTPException
from datetime import datetime
import logging
import time
import psutil

from src.api.models import HealthCheckResponse, SystemStatusResponse, ServiceStatus

logger = logging.getLogger(__name__)

router = APIRouter()

# Startup time for uptime calculation
_startup_time = time.time()


def _check_ollama_availability() -> bool:
    """Check if Ollama service is available."""
    try:
        import requests

        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        return response.status_code == 200
    except Exception as e:
        logger.warning(f"Ollama health check failed: {str(e)}")
        return False


def _check_chromadb_availability() -> bool:
    """Check if ChromaDB is available by opening the real vector store."""
    try:
        from src.rag.vector_store import VectorStore

        vs = VectorStore()
        # Lightweight op: returns collection stats without running a query.
        vs.get_stats()
        return True
    except (ImportError, ModuleNotFoundError) as e:
        logger.debug(f"ChromaDB dependencies unavailable: {e}")
        return False
    except Exception as e:
        logger.warning(f"ChromaDB health check failed: {e}")
        return False


def _check_database_availability() -> bool:
    """Check if database connection is available."""
    try:
        from src.database.database_manager import DatabaseManager

        db_manager = DatabaseManager()
        # Try a simple query to verify connection
        db_manager.get_session()
        return True
    except Exception as e:
        logger.warning(f"Database health check failed: {str(e)}")
        return False


@router.get(
    "/health",
    response_model=HealthCheckResponse,
    status_code=200,
    summary="Service health check",
    description="Check the health status of the API and its dependencies.",
)
async def health_check() -> HealthCheckResponse:
    """
    Quick health check for the API and core dependencies.

    Returns:
        HealthCheckResponse with overall and per-service status

    Example:
        GET /api/v1/health
        Response:
        {
            "status": "healthy",
            "timestamp": "2026-04-20T12:34:56.789Z",
            "version": "1.0.0",
            "services": {
                "ollama": {
                    "name": "ollama",
                    "status": "healthy",
                    "message": "LLM inference engine ready"
                },
                "chromadb": {
                    "name": "chromadb",
                    "status": "healthy",
                    "message": "Vector store connected"
                },
                "database": {
                    "name": "database",
                    "status": "healthy",
                    "message": "PostgreSQL connection active"
                }
            }
        }
    """
    logger.info("Health check requested")

    # Check service availability
    ollama_ok = _check_ollama_availability()
    chromadb_ok = _check_chromadb_availability()
    database_ok = _check_database_availability()

    # Build service statuses
    services = {
        "ollama": ServiceStatus(
            name="ollama",
            status="healthy" if ollama_ok else "unhealthy",
            message="LLM inference engine ready" if ollama_ok else "LLM engine unavailable",
        ),
        "chromadb": ServiceStatus(
            name="chromadb",
            status="healthy" if chromadb_ok else "unhealthy",
            message="Vector store connected" if chromadb_ok else "Vector store unavailable",
        ),
        "database": ServiceStatus(
            name="database",
            status="healthy" if database_ok else "unhealthy",
            message="Database connection active" if database_ok else "Database unavailable",
        ),
    }

    # Determine overall status
    all_healthy = all(s.status == "healthy" for s in services.values())
    any_unhealthy = any(s.status == "unhealthy" for s in services.values())
    overall_status = "healthy" if all_healthy else "degraded" if not any_unhealthy else "unhealthy"

    response = HealthCheckResponse(
        status=overall_status,
        timestamp=datetime.utcnow(),
        services=services,
        version="1.0.0",
    )

    logger.info(f"✓ Health check complete: {overall_status}")
    return response


@router.get(
    "/status",
    response_model=SystemStatusResponse,
    status_code=200,
    summary="Detailed system status",
    description="Get detailed system status including uptime, cache statistics, and model information.",
)
async def system_status() -> SystemStatusResponse:
    """
    Get detailed system status and metrics.

    Returns:
        SystemStatusResponse with uptime, cache stats, and service information

    Example:
        GET /api/v1/status
        Response:
        {
            "status": "healthy",
            "version": "1.0.0",
            "uptime_seconds": 3654.25,
            "active_analyses": 3,
            "completed_analyses": 127,
            "cache_size_mb": 45.3,
            "services": {
                "ollama": {...},
                "chromadb": {...},
                "database": {...}
            }
        }
    """
    logger.info("System status requested")

    # Calculate uptime
    uptime_seconds = time.time() - _startup_time

    # Check service availability
    ollama_ok = _check_ollama_availability()
    chromadb_ok = _check_chromadb_availability()
    database_ok = _check_database_availability()

    # Get cache size (from chromadb or file system)
    cache_size_mb = 0.0
    try:
        import os

        cache_dir = "/Users/bibekgupta/Downloads/projects/code-smell/cache"
        if os.path.exists(cache_dir):
            total_size = sum(
                os.path.getsize(os.path.join(dirpath, filename))
                for dirpath, dirnames, filenames in os.walk(cache_dir)
                for filename in filenames
            )
            cache_size_mb = total_size / (1024 * 1024)
    except Exception as e:
        logger.warning(f"Could not calculate cache size: {str(e)}")

    # Get active and completed analyses counts (from database or memory)
    active_analyses = 0
    completed_analyses = 0
    try:
        # Import from analysis.py to get in-memory counts
        from src.api.routes.analysis import analysis_state

        active_analyses = sum(
            1 for a in analysis_state.values() if a.get("status") in ["queued", "processing"]
        )
        completed_analyses = sum(
            1 for a in analysis_state.values() if a.get("status") in ["completed", "failed"]
        )
    except Exception as e:
        logger.warning(f"Could not retrieve analysis counts: {str(e)}")

    # Build service statuses
    services = {
        "ollama": ServiceStatus(
            name="ollama",
            status="healthy" if ollama_ok else "unhealthy",
            message="Ready for inference" if ollama_ok else "Unavailable",
        ),
        "chromadb": ServiceStatus(
            name="chromadb",
            status="healthy" if chromadb_ok else "unhealthy",
            message="Vector store operational" if chromadb_ok else "Unavailable",
        ),
        "database": ServiceStatus(
            name="database",
            status="healthy" if database_ok else "unhealthy",
            message="Connected" if database_ok else "Unavailable",
        ),
    }

    # Determine overall status
    all_healthy = all(s.status == "healthy" for s in services.values())
    any_unhealthy = any(s.status == "unhealthy" for s in services.values())
    overall_status = "healthy" if all_healthy else "degraded" if not any_unhealthy else "unhealthy"

    response = SystemStatusResponse(
        status=overall_status,
        version="1.0.0",
        uptime_seconds=uptime_seconds,
        active_analyses=active_analyses,
        completed_analyses=completed_analyses,
        cache_size_mb=round(cache_size_mb, 2),
        services=services,
    )

    logger.info(
        f"✓ System status: {overall_status}, uptime: {uptime_seconds:.1f}s, "
        f"active: {active_analyses}, completed: {completed_analyses}"
    )
    return response
