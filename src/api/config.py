"""
API-specific configuration settings.
Centralized settings for server, queue, timeouts, and resource limits.
"""

import os
from typing import Optional


class APIConfig:
    """API configuration settings."""

    # ========================================================================
    # Server Configuration
    # ========================================================================

    HOST: str = os.getenv("API_HOST", "0.0.0.0")
    PORT: int = int(os.getenv("API_PORT", "8000"))
    DEBUG: bool = os.getenv("API_DEBUG", "False").lower() == "true"
    RELOAD: bool = os.getenv("API_RELOAD", "False").lower() == "true"
    WORKERS: int = int(os.getenv("API_WORKERS", "4"))

    # ========================================================================
    # API Settings
    # ========================================================================

    API_VERSION: str = "1.0.0"
    API_TITLE: str = "Code Smell Detection API"
    API_DESCRIPTION: str = "Privacy-preserving LLM-based code smell detection with RAG enhancement"

    # ========================================================================
    # Code Analysis Settings
    # ========================================================================

    MIN_CODE_LENGTH: int = 10  # Minimum characters
    MAX_CODE_LENGTH: int = 100_000  # 100KB

    SUPPORTED_LANGUAGES: list = ["java", "python", "javascript"]

    # ========================================================================
    # Timeout & Performance Settings
    # ========================================================================

    DEFAULT_ANALYSIS_TIMEOUT: int = 300  # seconds
    MIN_TIMEOUT: int = 30
    MAX_TIMEOUT: int = 600

    # Background task timeouts
    QUEUE_PROCESSING_TIMEOUT: int = 30  # seconds
    DATABASE_TIMEOUT: int = 10  # seconds

    # ========================================================================
    # Queue & Async Settings
    # ========================================================================

    MAX_QUEUE_SIZE: int = 1000
    MAX_CONCURRENT_ANALYSES: int = 10
    WORKER_POOL_SIZE: int = 5

    # Task retry configuration
    MAX_RETRIES: int = 3
    RETRY_DELAY: int = 5  # seconds
    RETRY_BACKOFF: float = 2.0  # exponential backoff multiplier

    # ========================================================================
    # Cache Settings
    # ========================================================================

    ENABLE_RESULT_CACHE: bool = True
    CACHE_TTL: int = 3600  # 1 hour in seconds
    MAX_CACHE_SIZE_MB: int = 500
    CACHE_DIR: str = "./cache"

    # ========================================================================
    # Database Settings
    # ========================================================================

    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "sqlite:///./cache/code_smell.db"
    )
    DATABASE_ECHO: bool = os.getenv("DATABASE_ECHO", "False").lower() == "true"
    DATABASE_POOL_SIZE: int = 5
    DATABASE_POOL_RECYCLE: int = 3600

    # ========================================================================
    # LLM & Inference Settings
    # ========================================================================

    OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3")
    LLM_REQUEST_TIMEOUT: int = 60  # seconds

    # ========================================================================
    # RAG Settings
    # ========================================================================

    CHROMADB_HOST: str = os.getenv("CHROMADB_HOST", "localhost")
    CHROMADB_PORT: int = int(os.getenv("CHROMADB_PORT", "8000"))
    RAG_TOP_K: int = 3  # Number of similar examples to retrieve
    ENABLE_RAG: bool = os.getenv("ENABLE_RAG", "True").lower() == "true"

    # ========================================================================
    # Rate Limiting
    # ========================================================================

    ENABLE_RATE_LIMITING: bool = True
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW: int = 60  # seconds (1 minute)

    # ========================================================================
    # Resource Limits
    # ========================================================================

    MAX_MEMORY_PER_ANALYSIS_MB: int = 512
    MAX_CPU_TIME_PER_ANALYSIS: int = 300  # seconds

    # ========================================================================
    # Logging Configuration
    # ========================================================================

    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = "json"  # json or text
    LOG_DIR: str = "./logs"

    # ========================================================================
    # CORS Configuration
    # ========================================================================

    CORS_ORIGINS: list = [
        "http://localhost:3000",
        "http://localhost:8000",
        "http://127.0.0.1:3000",
    ]
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: list = ["*"]
    CORS_ALLOW_HEADERS: list = ["*"]

    # ========================================================================
    # Security Settings
    # ========================================================================

    ENABLE_CORS: bool = True
    ENABLE_HTTPS: bool = False

    # ========================================================================
    # Monitoring & Metrics
    # ========================================================================

    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 9090
    ENABLE_PROFILING: bool = False

    # ========================================================================
    # Class Methods
    # ========================================================================

    @classmethod
    def validate_analysis_timeout(cls, timeout: int) -> int:
        """
        Validate and clamp analysis timeout to acceptable range.

        Args:
            timeout: Requested timeout in seconds

        Returns:
            Validated timeout value
        """
        if timeout < cls.MIN_TIMEOUT:
            return cls.MIN_TIMEOUT
        if timeout > cls.MAX_TIMEOUT:
            return cls.MAX_TIMEOUT
        return timeout

    @classmethod
    def validate_code_length(cls, code: str) -> tuple[bool, Optional[str]]:
        """
        Validate code length.

        Args:
            code: Code snippet to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if len(code) < cls.MIN_CODE_LENGTH:
            return False, f"Code must be at least {cls.MIN_CODE_LENGTH} characters"
        if len(code) > cls.MAX_CODE_LENGTH:
            return False, f"Code must not exceed {cls.MAX_CODE_LENGTH} characters"
        return True, None

    @classmethod
    def validate_language(cls, language: str) -> tuple[bool, Optional[str]]:
        """
        Validate programming language.

        Args:
            language: Language to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if language not in cls.SUPPORTED_LANGUAGES:
            supported = ", ".join(cls.SUPPORTED_LANGUAGES)
            return False, f"Language '{language}' not supported. Supported: {supported}"
        return True, None


# Create singleton instance
config = APIConfig()
