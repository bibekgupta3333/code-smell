"""
Custom API exceptions for error handling.
Domain-specific exceptions for better error categorization and handling.
"""

from typing import Optional, Dict, Any


class APIException(Exception):
    """Base exception for all API errors."""

    def __init__(
        self,
        error_code: str,
        message: str,
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize API exception.

        Args:
            error_code: Machine-readable error code
            message: Human-readable error message
            status_code: HTTP status code
            details: Additional error details
        """
        self.error_code = error_code
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for JSON response."""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details,
        }


# ============================================================================
# Analysis Exceptions
# ============================================================================


class AnalysisException(APIException):
    """Base exception for analysis-related errors."""

    def __init__(
        self,
        message: str,
        error_code: str = "ANALYSIS_ERROR",
        status_code: int = 400,
        **kwargs
    ):
        super().__init__(
            error_code=error_code,
            message=message,
            status_code=status_code,
            **kwargs
        )


class AnalysisTimeoutException(AnalysisException):
    """Raised when analysis exceeds timeout limit."""

    def __init__(
        self, analysis_id: str, timeout_seconds: int, elapsed_seconds: float
    ):
        super().__init__(
            message=f"Analysis {analysis_id} exceeded timeout of {timeout_seconds}s (elapsed: {elapsed_seconds:.1f}s)",
            error_code="ANALYSIS_TIMEOUT",
            status_code=408,
            details={
                "analysis_id": analysis_id,
                "timeout_seconds": timeout_seconds,
                "elapsed_seconds": elapsed_seconds,
            },
        )


class AnalysisFailedException(AnalysisException):
    """Raised when analysis fails during processing."""

    def __init__(self, analysis_id: str, reason: str):
        super().__init__(
            message=f"Analysis {analysis_id} failed: {reason}",
            error_code="ANALYSIS_FAILED",
            status_code=500,
            details={"analysis_id": analysis_id, "reason": reason},
        )


class AnalysisNotFoundException(AnalysisException):
    """Raised when analysis is not found."""

    def __init__(self, analysis_id: str):
        super().__init__(
            message=f"Analysis with ID {analysis_id} not found",
            error_code="ANALYSIS_NOT_FOUND",
            status_code=404,
            details={"analysis_id": analysis_id},
        )


class AnalysisAlreadyProcessingException(AnalysisException):
    """Raised when attempting to process an analysis that's already being processed."""

    def __init__(self, analysis_id: str):
        super().__init__(
            message=f"Analysis {analysis_id} is already being processed",
            error_code="ANALYSIS_ALREADY_PROCESSING",
            status_code=409,
            details={"analysis_id": analysis_id},
        )


# ============================================================================
# Validation Exceptions
# ============================================================================


class ValidationException(APIException):
    """Base exception for validation errors."""

    def __init__(
        self,
        message: str,
        error_code: str = "VALIDATION_ERROR",
        status_code: int = 400,
        **kwargs
    ):
        super().__init__(
            error_code=error_code,
            message=message,
            status_code=status_code,
            **kwargs
        )


class InvalidLanguageException(ValidationException):
    """Raised when language is not supported."""

    def __init__(self, language: str, supported_languages: list):
        super().__init__(
            message=f"Language '{language}' is not supported. Supported: {', '.join(supported_languages)}",
            error_code="INVALID_LANGUAGE",
            details={
                "provided_language": language,
                "supported_languages": supported_languages,
            },
        )


class InvalidCodeException(ValidationException):
    """Raised when code is invalid or doesn't meet requirements."""

    def __init__(self, reason: str, details: Optional[Dict] = None):
        super().__init__(
            message=f"Code validation failed: {reason}",
            error_code="INVALID_CODE",
            details=details or {"reason": reason},
        )


class CodeTooShortException(InvalidCodeException):
    """Raised when code is too short."""

    def __init__(self, min_length: int, actual_length: int):
        super().__init__(
            reason=f"Code must be at least {min_length} characters (provided: {actual_length})",
            details={
                "min_length": min_length,
                "actual_length": actual_length,
            },
        )


class CodeTooLongException(InvalidCodeException):
    """Raised when code exceeds maximum length."""

    def __init__(self, max_length: int, actual_length: int):
        super().__init__(
            reason=f"Code must not exceed {max_length} characters (provided: {actual_length})",
            details={
                "max_length": max_length,
                "actual_length": actual_length,
            },
        )


# ============================================================================
# Resource Exceptions
# ============================================================================


class ResourceException(APIException):
    """Base exception for resource-related errors."""

    def __init__(
        self,
        message: str,
        error_code: str = "RESOURCE_ERROR",
        status_code: int = 503,
        **kwargs
    ):
        super().__init__(
            error_code=error_code,
            message=message,
            status_code=status_code,
            **kwargs
        )


class InsufficientResourcesException(ResourceException):
    """Raised when insufficient resources available for analysis."""

    def __init__(self, resource_type: str, required: int, available: int):
        super().__init__(
            message=f"Insufficient {resource_type}: required {required}, available {available}",
            error_code="INSUFFICIENT_RESOURCES",
            details={
                "resource_type": resource_type,
                "required": required,
                "available": available,
            },
        )


class QueueFullException(ResourceException):
    """Raised when analysis queue is full."""

    def __init__(self, queue_size: int, max_size: int):
        super().__init__(
            message=f"Analysis queue is full ({queue_size}/{max_size}). Please try again later.",
            error_code="QUEUE_FULL",
            details={"queue_size": queue_size, "max_size": max_size},
        )


class RateLimitExceededException(ResourceException):
    """Raised when rate limit is exceeded."""

    def __init__(self, limit: int, window_seconds: int):
        super().__init__(
            message=f"Rate limit exceeded: {limit} requests per {window_seconds} seconds",
            error_code="RATE_LIMIT_EXCEEDED",
            status_code=429,
            details={"limit": limit, "window_seconds": window_seconds},
        )


# ============================================================================
# Service Exceptions
# ============================================================================


class ServiceException(APIException):
    """Base exception for service-related errors."""

    def __init__(
        self,
        message: str,
        error_code: str = "SERVICE_ERROR",
        status_code: int = 503,
        **kwargs
    ):
        super().__init__(
            error_code=error_code,
            message=message,
            status_code=status_code,
            **kwargs
        )


class DatabaseException(ServiceException):
    """Raised when database operation fails."""

    def __init__(self, operation: str, reason: str):
        super().__init__(
            message=f"Database {operation} failed: {reason}",
            error_code="DATABASE_ERROR",
            details={"operation": operation, "reason": reason},
        )


class LLMServiceException(ServiceException):
    """Raised when LLM service is unavailable."""

    def __init__(self, reason: str):
        super().__init__(
            message=f"LLM service error: {reason}",
            error_code="LLM_SERVICE_ERROR",
            details={"reason": reason},
        )


class RAGServiceException(ServiceException):
    """Raised when RAG service fails."""

    def __init__(self, operation: str, reason: str):
        super().__init__(
            message=f"RAG {operation} failed: {reason}",
            error_code="RAG_SERVICE_ERROR",
            details={"operation": operation, "reason": reason},
        )


class CacheException(ServiceException):
    """Raised when cache operation fails."""

    def __init__(self, operation: str, reason: str):
        super().__init__(
            message=f"Cache {operation} failed: {reason}",
            error_code="CACHE_ERROR",
            details={"operation": operation, "reason": reason},
        )


# ============================================================================
# Configuration Exceptions
# ============================================================================


class ConfigException(APIException):
    """Base exception for configuration errors."""

    def __init__(
        self,
        message: str,
        error_code: str = "CONFIG_ERROR",
        status_code: int = 500,
        **kwargs
    ):
        super().__init__(
            error_code=error_code,
            message=message,
            status_code=status_code,
            **kwargs
        )


class InvalidConfigException(ConfigException):
    """Raised when configuration is invalid."""

    def __init__(self, config_key: str, reason: str):
        super().__init__(
            message=f"Invalid configuration for '{config_key}': {reason}",
            error_code="INVALID_CONFIG",
            details={"config_key": config_key, "reason": reason},
        )
