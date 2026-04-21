"""
Request validators for API endpoints.
Validate code size, language support, and apply rate limiting.
"""

from typing import Optional, Tuple
from datetime import datetime, timedelta
import hashlib
from collections import defaultdict

from src.api.config import config
from src.api.exceptions import (
    CodeTooShortException,
    CodeTooLongException,
    InvalidLanguageException,
    RateLimitExceededException,
)


class CodeValidator:
    """Validates code submissions."""

    @staticmethod
    def validate_code(code: str) -> Tuple[bool, Optional[str]]:
        """
        Validate code snippet meets requirements.

        Args:
            code: Code to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if len(code) < config.MIN_CODE_LENGTH:
            raise CodeTooShortException(config.MIN_CODE_LENGTH, len(code))

        if len(code) > config.MAX_CODE_LENGTH:
            raise CodeTooLongException(config.MAX_CODE_LENGTH, len(code))

        return True, None

    @staticmethod
    def validate_language(language: str) -> Tuple[bool, Optional[str]]:
        """
        Validate programming language is supported.

        Args:
            language: Language to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if language not in config.SUPPORTED_LANGUAGES:
            raise InvalidLanguageException(language, config.SUPPORTED_LANGUAGES)

        return True, None


class RateLimiter:
    """Simple in-memory rate limiter."""

    def __init__(self):
        """Initialize rate limiter."""
        # Track requests per client: {client_id: [(timestamp, method, endpoint)]}
        self.requests: dict = defaultdict(list)
        self.cleanup_interval = 300  # seconds

    def is_allowed(self, client_id: str) -> Tuple[bool, Optional[str]]:
        """
        Check if client has exceeded rate limit.

        Args:
            client_id: Client identifier (IP, user_id, etc.)

        Returns:
            Tuple of (is_allowed, error_message)
        """
        if not config.ENABLE_RATE_LIMITING:
            return True, None

        now = datetime.utcnow()
        window_start = now - timedelta(seconds=config.RATE_LIMIT_WINDOW)

        # Clean old requests
        if client_id in self.requests:
            self.requests[client_id] = [
                req_time for req_time in self.requests[client_id]
                if req_time > window_start
            ]
        else:
            self.requests[client_id] = []

        # Check limit
        if len(self.requests[client_id]) >= config.RATE_LIMIT_REQUESTS:
            raise RateLimitExceededException(
                config.RATE_LIMIT_REQUESTS, config.RATE_LIMIT_WINDOW
            )

        # Add current request
        self.requests[client_id].append(now)
        return True, None

    def get_remaining(self, client_id: str) -> int:
        """
        Get remaining requests for client in current window.

        Args:
            client_id: Client identifier

        Returns:
            Number of remaining requests
        """
        if not config.ENABLE_RATE_LIMITING:
            return config.RATE_LIMIT_REQUESTS

        now = datetime.utcnow()
        window_start = now - timedelta(seconds=config.RATE_LIMIT_WINDOW)

        if client_id in self.requests:
            recent_requests = [
                req_time for req_time in self.requests[client_id]
                if req_time > window_start
            ]
            return max(0, config.RATE_LIMIT_REQUESTS - len(recent_requests))

        return config.RATE_LIMIT_REQUESTS

    def reset(self, client_id: str) -> None:
        """
        Reset rate limit for client.

        Args:
            client_id: Client identifier
        """
        if client_id in self.requests:
            del self.requests[client_id]


class CacheValidator:
    """Validates cache operations."""

    @staticmethod
    def get_cache_key(code: str, language: str, include_rag: bool = True) -> str:
        """
        Generate cache key for code snippet.

        Args:
            code: Code snippet
            language: Programming language
            include_rag: Whether RAG is enabled

        Returns:
            Cache key hash
        """
        cache_input = f"{code}:{language}:{include_rag}"
        return hashlib.sha256(cache_input.encode()).hexdigest()

    @staticmethod
    def is_cache_valid(cache_time: datetime, ttl_seconds: int) -> bool:
        """
        Check if cached result is still valid.

        Args:
            cache_time: Time when result was cached
            ttl_seconds: Cache TTL in seconds

        Returns:
            True if cache is still valid
        """
        age = (datetime.utcnow() - cache_time).total_seconds()
        return age < ttl_seconds

    @staticmethod
    def should_use_cache(code: str, file_name: str = None) -> bool:
        """
        Determine if caching should be used for this code.

        Args:
            code: Code snippet
            file_name: Optional file name

        Returns:
            True if caching should be enabled
        """
        if not config.ENABLE_RESULT_CACHE:
            return False

        # Don't cache test/example files
        if file_name and any(
            pattern in file_name.lower()
            for pattern in ["test", "example", "demo", "temp"]
        ):
            return False

        return True


class TimeoutValidator:
    """Validates timeout configurations."""

    @staticmethod
    def validate_timeout(timeout_seconds: int) -> int:
        """
        Validate and clamp timeout to acceptable range.

        Args:
            timeout_seconds: Requested timeout

        Returns:
            Validated timeout value
        """
        if timeout_seconds < config.MIN_TIMEOUT:
            return config.MIN_TIMEOUT

        if timeout_seconds > config.MAX_TIMEOUT:
            return config.MAX_TIMEOUT

        return timeout_seconds

    @staticmethod
    def get_default_timeout() -> int:
        """Get default analysis timeout."""
        return config.DEFAULT_ANALYSIS_TIMEOUT

    @staticmethod
    def get_max_timeout() -> int:
        """Get maximum allowed timeout."""
        return config.MAX_TIMEOUT


class RequestValidator:
    """Main request validator orchestrating all validations."""

    def __init__(self):
        """Initialize validators."""
        self.code_validator = CodeValidator()
        self.rate_limiter = RateLimiter()
        self.cache_validator = CacheValidator()
        self.timeout_validator = TimeoutValidator()

    def validate_analysis_request(
        self,
        code: str,
        language: Optional[str],
        timeout_seconds: int,
        client_id: str,
    ) -> dict:
        """
        Validate complete analysis request.

        Args:
            code: Code to analyze
            language: Programming language
            timeout_seconds: Analysis timeout
            client_id: Client identifier for rate limiting

        Returns:
            Dictionary with validation results
        """
        # Validate code
        self.code_validator.validate_code(code)

        # Validate language if provided
        if language:
            self.code_validator.validate_language(language)

        # Validate timeout
        validated_timeout = self.timeout_validator.validate_timeout(timeout_seconds)

        # Check rate limit
        self.rate_limiter.is_allowed(client_id)

        return {
            "is_valid": True,
            "timeout_seconds": validated_timeout,
            "remaining_requests": self.rate_limiter.get_remaining(client_id),
        }


# Create singleton instance
validator = RequestValidator()
