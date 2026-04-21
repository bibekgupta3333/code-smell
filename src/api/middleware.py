"""
Custom FastAPI middleware for logging, error handling, and monitoring.
"""

import time
import logging
import json
from typing import Callable
from datetime import datetime

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from src.api.config import config
from src.api.exceptions import APIException


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging HTTP requests and responses."""

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

            # Log request
            log_data = {
                "request_id": request_id,
                "timestamp": datetime.utcnow().isoformat(),
                "method": method,
                "path": path,
                "status_code": response.status_code,
                "client_ip": client_ip,
                "process_time_ms": round(process_time * 1000, 2),
            }

            # Log at appropriate level
            if response.status_code >= 500:
                self.logger.error(f"Request error: {json.dumps(log_data)}")
            elif response.status_code >= 400:
                self.logger.warning(f"Request warning: {json.dumps(log_data)}")
            else:
                self.logger.info(f"Request: {json.dumps(log_data)}")

            # Add request ID to response headers
            response.headers["x-request-id"] = request_id

            return response

        except Exception as e:
            process_time = time.time() - start_time

            log_data = {
                "request_id": request_id,
                "timestamp": datetime.utcnow().isoformat(),
                "method": method,
                "path": path,
                "error": str(e),
                "client_ip": client_ip,
                "process_time_ms": round(process_time * 1000, 2),
            }

            self.logger.error(f"Request error: {json.dumps(log_data)}")
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


class RateLimitingMiddleware(BaseHTTPMiddleware):
    """Middleware for rate limiting (basic implementation)."""

    def __init__(self, app, rate_limiter=None):
        """
        Initialize rate limiting middleware.

        Args:
            app: FastAPI application
            rate_limiter: RateLimiter instance
        """
        super().__init__(app)
        self.rate_limiter = rate_limiter
        self.logger = logging.getLogger(__name__)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Check rate limits for request.

        Args:
            request: HTTP request
            call_next: Next middleware/route handler

        Returns:
            HTTP response or 429 Too Many Requests
        """
        if not config.ENABLE_RATE_LIMITING or not self.rate_limiter:
            return await call_next(request)

        # Get client identifier (IP for now, could use API key)
        client_id = request.client.host if request.client else "unknown"

        try:
            self.rate_limiter.is_allowed(client_id)
            response = await call_next(request)

            # Add rate limit headers
            remaining = self.rate_limiter.get_remaining(client_id)
            response.headers["x-ratelimit-limit"] = str(config.RATE_LIMIT_REQUESTS)
            response.headers["x-ratelimit-remaining"] = str(remaining)
            response.headers["x-ratelimit-reset"] = str(config.RATE_LIMIT_WINDOW)

            return response

        except Exception as e:
            self.logger.warning(f"Rate limit check failed: {str(e)}")
            # Continue request even if rate limiting fails
            return await call_next(request)


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

    # Then rate limiting
    if config.ENABLE_RATE_LIMITING:
        from src.api.dependencies import get_rate_limiter

        rate_limiter = get_rate_limiter()
        app.add_middleware(RateLimitingMiddleware, rate_limiter=rate_limiter)

    # Then performance monitoring
    if config.ENABLE_METRICS:
        app.add_middleware(PerformanceMonitoringMiddleware, logger=logger)

    # Then error handling
    app.add_middleware(ErrorHandlingMiddleware, logger=logger)

    # Request logging (innermost, closest to routes)
    app.add_middleware(RequestLoggingMiddleware, logger=logger)

    logger.info("✓ All middleware configured")
