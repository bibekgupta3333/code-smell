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

        return {
            "is_valid": True,
            "timeout_seconds": validated_timeout,
        }


# Create singleton instance
validator = RequestValidator()
