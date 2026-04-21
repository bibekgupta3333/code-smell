"""
FastAPI server for Code Smell Detection.
Main application entry point with route setup and middleware configuration.
"""

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from src.api.config import config
from src.api.dependencies import get_logger, initialize_dependencies, cleanup_dependencies
from src.api.middleware import setup_middleware
from src.api.exceptions import APIException

logger = get_logger(__name__)


# ============================================================================
# Startup/Shutdown Events
# ============================================================================

async def lifespan(app: FastAPI):
    """Manage app lifecycle: startup and shutdown."""
    # Startup
    logger.info("🚀 Starting Code Smell Detection API v%s", config.API_VERSION)

    try:
        # Initialize all dependencies
        init_status = initialize_dependencies()
        logger.info(
            "✅ Dependencies initialized: %s",
            {k: "✓" if v else "✗" for k, v in init_status.items()}
        )
    except Exception as e:
        logger.error(f"❌ Initialization error: {str(e)}", exc_info=True)
        raise

    logger.info("✅ API ready to accept requests")

    yield

    # Shutdown
    logger.info("🛑 Shutting down Code Smell Detection API")

    try:
        cleanup_dependencies()
        logger.info("✅ Cleanup complete, API shutdown successfully")
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}", exc_info=True)


# ============================================================================
# Create FastAPI App
# ============================================================================

app = FastAPI(
    title=config.API_TITLE,
    description=config.API_DESCRIPTION,
    version=config.API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# ============================================================================
# CORS Configuration
# ============================================================================

if config.ENABLE_CORS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.CORS_ORIGINS,
        allow_credentials=config.CORS_ALLOW_CREDENTIALS,
        allow_methods=config.CORS_ALLOW_METHODS,
        allow_headers=config.CORS_ALLOW_HEADERS,
    )
    logger.info("✓ CORS middleware configured")

# ============================================================================
# Custom Middleware Setup
# ============================================================================

setup_middleware(app)

# ============================================================================
# Exception Handlers
# ============================================================================


@app.exception_handler(APIException)
async def api_exception_handler(request: Request, exc: APIException):
    """Handle custom API exceptions."""
    request_id = request.headers.get("x-request-id", "unknown")
    logger.warning(f"API error {exc.error_code}: {exc.message}")

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.to_dict(),
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": request_id,
        },
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle global exceptions."""
    request_id = request.headers.get("x-request-id", "unknown")
    logger.error(f"Unhandled exception: {exc}", exc_info=True)

    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "error_code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred",
                "details": str(exc) if config.DEBUG else None,
            },
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": request_id,
        },
    )


# ============================================================================
# Root Endpoint
# ============================================================================

@app.get("/", tags=["root"], include_in_schema=False)
async def root():
    """Serve the frontend index.html"""
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    index_file = os.path.join(static_dir, "index.html")

    if os.path.exists(index_file):
        with open(index_file, "r") as f:
            return HTMLResponse(content=f.read())

    return {
        "message": "Code Smell Detection API",
        "version": config.API_VERSION,
        "docs": "/docs",
        "status": "running"
    }


# ============================================================================
# Route Registration
# ============================================================================

from src.api.routes import analysis, comparison, health, research

app.include_router(analysis.router, prefix="/api/v1", tags=["analysis"])
app.include_router(comparison.router, prefix="/api/v1", tags=["comparison"])
app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(research.router, tags=["research"])

# ============================================================================
# Static Files Mount
# ============================================================================

# Serve static files (HTML, CSS, JS) from the static directory
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    logger.info(f"✓ Static files mounted at /static from {static_dir}")
else:
    logger.warning(f"⚠ Static directory not found at {static_dir}")


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=True
    )
