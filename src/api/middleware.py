"""
Custom FastAPI middleware for logging, error handling, and monitoring.
"""

import time
import logging
from typing import Callable
from datetime import datetime

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from src.api.config import config
from src.api.exceptions import APIException


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging HTTP requests and responses."""

    # Paths to skip verbose logging (health checks, status, etc.)
    QUIET_PATHS = {"/api/v1/status", "/api/v1/health", "/docs", "/redoc", "/openapi.json"}

    def __init__(self, app, logger: logging.Logger = None):
        """
        Initialize request logging middleware.

        Args:
            app: FastAPI application
            logger: Logger instance
        """
        super().__init__(app)
        self.logger = logger or logging.getLogger(__name__)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request and log details.

        Args:
            request: HTTP request
            call_next: Next middleware/route handler

        Returns:
            HTTP response
        """
        # Extract request info
        request_id = request.headers.get("x-request-id", str(time.time()))
        method = request.method
        path = request.url.path
        client_ip = request.client.host if request.client else "unknown"

        # Record start time
        start_time = time.time()

        # Process request
        try:
            response = await call_next(request)
            process_time = time.time() - start_time

            # Only log errors, warnings, and important operations (not health checks)
            if response.status_code >= 500:
                self.logger.error(
                    f"❌ {method} {path} → {response.status_code} | {process_time*1000:.1f}ms"
                )
            elif response.status_code >= 400:
                self.logger.warning(
                    f"⚠️  {method} {path} → {response.status_code} | {process_time*1000:.1f}ms"
                )
            elif path not in self.QUIET_PATHS:
                # Log analysis requests but not health checks
                if "/analyze" in path or "/results" in path:
                    self.logger.info(
                        f"✓ {method} {path} → {response.status_code} | {process_time*1000:.1f}ms"
                    )

            # Add request ID to response headers
            response.headers["x-request-id"] = request_id

            return response

        except Exception as e:
            process_time = time.time() - start_time
            self.logger.error(f"❌ {method} {path} | Error: {str(e)} | {process_time*1000:.1f}ms")
            raise


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Middleware for standardized error handling."""

    def __init__(self, app, logger: logging.Logger = None):
        """
        Initialize error handling middleware.

        Args:
            app: FastAPI application
            logger: Logger instance
        """
        super().__init__(app)
        self.logger = logger or logging.getLogger(__name__)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Handle exceptions and return standardized error responses.

        Args:
            request: HTTP request
            call_next: Next middleware/route handler

        Returns:
            HTTP response or error response
        """
        request_id = request.headers.get("x-request-id", str(time.time()))

        try:
            response = await call_next(request)
            return response

        except APIException as e:
            # Handle custom API exceptions
            self.logger.warning(
                f"API exception: {e.error_code} - {e.message}",
                extra={"request_id": request_id},
            )

            return JSONResponse(
                status_code=e.status_code,
                content={
                    "error": e.to_dict(),
                    "timestamp": datetime.utcnow().isoformat(),
                    "request_id": request_id,
                },
            )

        except Exception as e:
            # Handle unexpected exceptions
            self.logger.error(
                f"Unhandled exception: {str(e)}",
                exc_info=True,
                extra={"request_id": request_id},
            )

            return JSONResponse(
                status_code=500,
                content={
                    "error": {
                        "error_code": "INTERNAL_SERVER_ERROR",
                        "message": "An unexpected error occurred",
                        "details": {} if not config.DEBUG else {"exception": str(e)},
                    },
                    "timestamp": datetime.utcnow().isoformat(),
                    "request_id": request_id,
                },
            )


class PerformanceMonitoringMiddleware(BaseHTTPMiddleware):
    """Middleware for monitoring request performance."""

    def __init__(self, app, logger: logging.Logger = None):
        """
        Initialize performance monitoring middleware.

        Args:
            app: FastAPI application
            logger: Logger instance
        """
        super().__init__(app)
        self.logger = logger or logging.getLogger(__name__)
        self.slow_request_threshold = 1.0  # seconds

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Monitor request performance.

        Args:
            request: HTTP request
            call_next: Next middleware/route handler

        Returns:
            HTTP response
        """
        start_time = time.time()

        response = await call_next(request)

        process_time = time.time() - start_time

        # Log slow requests
        if process_time > self.slow_request_threshold:
            self.logger.warning(
                f"Slow request: {request.method} {request.url.path} "
                f"took {process_time:.2f}s"
            )

        # Add performance headers
        response.headers["x-process-time"] = str(round(process_time, 4))

        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware for adding security headers."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Add security headers to response.

        Args:
            request: HTTP request
            call_next: Next middleware/route handler

        Returns:
            HTTP response with security headers
        """
        response = await call_next(request)

        # Add security headers
        response.headers["x-content-type-options"] = "nosniff"
        response.headers["x-frame-options"] = "DENY"
        response.headers["x-xss-protection"] = "1; mode=block"
        response.headers["strict-transport-security"] = "max-age=31536000; includeSubDomains"

        return response


def setup_middleware(app):
    """
    Setup all middleware on FastAPI app.

    Args:
        app: FastAPI application
    """
    logger = logging.getLogger(__name__)

    # Add middleware in reverse order (they wrap in LIFO)
    # Security headers go first (outermost)
    app.add_middleware(SecurityHeadersMiddleware)



    # Then performance monitoring
    if config.ENABLE_METRICS:
        app.add_middleware(PerformanceMonitoringMiddleware, logger=logger)

    # Then error handling
    app.add_middleware(ErrorHandlingMiddleware, logger=logger)

    # Request logging (innermost, closest to routes)
    app.add_middleware(RequestLoggingMiddleware, logger=logger)

    logger.info("✓ All middleware configured")
