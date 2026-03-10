"""
Ollama LLM Client Wrapper
Handles communication with local Ollama instance with caching, retries, and streaming.

Architecture: Consistent with Architecture Section 2 (Model Selection) and Section 9 (Caching)
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Optional, AsyncGenerator, Any, Dict, List
from dataclasses import dataclass, asdict
from enum import Enum

import ollama
from config import (
    OLLAMA_BASE_URL,
    OLLAMA_TIMEOUT,
    DEFAULT_MODEL,
    AVAILABLE_MODELS,
    LLM_CONFIG,
    ENABLE_LLM_CACHE,
    CACHE_DIR,
    RETRY_CONFIG,
    get_model_for_task,
)

logger = logging.getLogger(__name__)


class ResponseType(str, Enum):
    """Response types for LLM generation."""
    COMPLETION = "completion"
    STREAMING = "streaming"
    CACHED = "cached"


@dataclass
class ModelStats:
    """Statistics for LLM model performance."""
    model: str
    tokens_generated: int = 0
    tokens_input: int = 0
    latency_ms: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    errors: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class OllamaClient:
    """
    Wrapper for Ollama API with caching, retries, and performance tracking.

    Features:
    - Connection verification and health checks
    - Single and batch prompt completion
    - Streaming response support
    - Response caching (file-based)
    - Automatic retry with exponential backoff
    - Token counting and performance tracking
    - M4 Pro optimizations (low concurrency, memory-aware)

    Example:
        >>> client = OllamaClient()
        >>> response = await client.generate(
        ...     prompt="Analyze this code for smells",
        ...     model="llama3:8b"
        ... )
        >>> print(response)
    """

    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        default_model: str = DEFAULT_MODEL,
        cache_enabled: bool = ENABLE_LLM_CACHE,
        cache_dir: Path = CACHE_DIR,
    ):
        """
        Initialize Ollama client.

        Args:
            base_url: Ollama server URL
            default_model: Default model to use
            cache_enabled: Enable response caching
            cache_dir: Cache directory path
        """
        self.base_url = base_url
        self.default_model = default_model
        self.cache_enabled = cache_enabled
        self.cache_dir = Path(cache_dir)

        # Ensure cache directory exists
        if self.cache_enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Ollama client
        self.client = ollama.Client(host=base_url)

        # Performance statistics
        self.stats: Dict[str, ModelStats] = {}

        # Connection state
        self._connection_verified = False
        self._last_error: Optional[str] = None

        logger.info(f"Ollama client initialized: {base_url}")

    async def verify_connection(self) -> bool:
        """
        Verify Ollama server is running and accessible.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Try to list available models
            models = self.client.list()
            if models and "models" in models:
                self._connection_verified = True
                logger.info(f"✓ Ollama connection verified. Available models: {len(models['models'])}")
                return True
            else:
                msg = "No models available in Ollama"
                logger.error(msg)
                self._last_error = msg
                return False
        except Exception as e:
            msg = f"Connection to Ollama failed: {e}"
            logger.error(msg)
            self._last_error = msg
            self._connection_verified = False
            return False

    async def list_available_models(self) -> List[str]:
        """
        List all available models in Ollama.

        Returns:
            List of model names
        """
        try:
            models_response = self.client.list()
            if not models_response or "models" not in models_response:
                return []

            return [model["name"] for model in models_response["models"]]
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []

    def _get_cache_path(self, prompt_hash: str) -> Path:
        """
        Get cache file path for prompt hash.

        Args:
            prompt_hash: Hash of the prompt

        Returns:
            Path to cache file
        """
        return self.cache_dir / f"{prompt_hash}.json"

    def _hash_prompt(self, prompt: str, model: str) -> str:
        """
        Hash prompt for caching.

        Args:
            prompt: Prompt text
            model: Model name

        Returns:
            Hash string
        """
        import hashlib
        combined = f"{model}:{prompt}"
        return hashlib.md5(combined.encode()).hexdigest()

    def _load_from_cache(self, prompt_hash: str) -> Optional[str]:
        """
        Load response from cache.

        Args:
            prompt_hash: Prompt hash

        Returns:
            Cached response or None
        """
        if not self.cache_enabled:
            return None

        cache_path = self._get_cache_path(prompt_hash)
        try:
            if cache_path.exists():
                with open(cache_path, 'r') as f:
                    data = json.load(f)
                    return data.get("response")
        except Exception as e:
            logger.warning(f"Failed to load from cache: {e}")

        return None

    def _save_to_cache(self, prompt_hash: str, response: str, model: str) -> None:
        """
        Save response to cache.

        Args:
            prompt_hash: Prompt hash
            response: Response text
            model: Model used
        """
        if not self.cache_enabled:
            return

        cache_path = self._get_cache_path(prompt_hash)
        try:
            with open(cache_path, 'w') as f:
                json.dump({
                    "response": response,
                    "model": model,
                    "timestamp": time.time(),
                }, f)
        except Exception as e:
            logger.warning(f"Failed to save to cache: {e}")

    async def _retry_with_backoff(
        self,
        func: callable,
        max_retries: int = RETRY_CONFIG["max_retries"],
        initial_delay: float = RETRY_CONFIG["initial_delay"],
        max_delay: float = RETRY_CONFIG["max_delay"],
        exponential_base: float = RETRY_CONFIG["exponential_base"],
    ) -> Any:
        """
        Retry function with exponential backoff.

        Args:
            func: Async function to retry
            max_retries: Maximum number of retries
            initial_delay: Initial delay in seconds
            max_delay: Maximum delay in seconds
            exponential_base: Base for exponential calculation

        Returns:
            Result from function

        Raises:
            Exception: After max retries exceeded
        """
        last_error = None
        delay = initial_delay

        for attempt in range(max_retries + 1):
            try:
                return await func()
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries + 1} failed: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    await asyncio.sleep(delay)
                    delay = min(delay * exponential_base, max_delay)
                else:
                    logger.error(f"All {max_retries + 1} attempts failed")

        raise last_error

    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = LLM_CONFIG["temperature"],
        max_tokens: int = LLM_CONFIG["max_tokens"],
        timeout: int = OLLAMA_TIMEOUT,
    ) -> str:
        """
        Generate completion for a prompt.

        Args:
            prompt: Prompt text
            model: Model to use (defaults to DEFAULT_MODEL)
            system_prompt: System prompt for context
            temperature: Sampling temperature (0=deterministic, 1=random)
            max_tokens: Maximum tokens to generate
            timeout: Timeout in seconds

        Returns:
            Generated text

        Raises:
            TimeoutError: If request times out
            Exception: If generation fails
        """
        model = model or self.default_model

        # Check cache
        prompt_hash = self._hash_prompt(prompt, model)
        cached_response = self._load_from_cache(prompt_hash)

        if model not in self.stats:
            self.stats[model] = ModelStats(model=model)

        if cached_response:
            logger.debug(f"Cache hit for prompt: {prompt[:50]}...")
            self.stats[model].cache_hits += 1
            return cached_response

        self.stats[model].cache_misses += 1

        # Prepare messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Generate with retry
        start_time = time.time()

        async def _generate_impl():
            return await asyncio.to_thread(
                self.client.chat,
                model=model,
                messages=messages,
                temperature=temperature,
                num_predict=max_tokens,
                stream=False,
            )

        try:
            response = await asyncio.wait_for(
                self._retry_with_backoff(_generate_impl),
                timeout=timeout,
            )

            # Extract text from response
            text = response["message"]["content"] if isinstance(response, dict) else str(response)

            # Update stats
            latency = (time.time() - start_time) * 1000
            self.stats[model].latency_ms = latency
            self.stats[model].tokens_generated += len(text.split())

            # Cache response
            self._save_to_cache(prompt_hash, text, model)

            logger.debug(f"Generated {len(text.split())} tokens in {latency:.0f}ms")
            return text

        except asyncio.TimeoutError:
            self.stats[model].errors += 1
            raise TimeoutError(f"LLM generation timed out after {timeout}s")
        except Exception as e:
            self.stats[model].errors += 1
            logger.error(f"LLM generation failed: {e}")
            raise

    async def generate_stream(
        self,
        prompt: str,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = LLM_CONFIG["temperature"],
        max_tokens: int = LLM_CONFIG["max_tokens"],
    ) -> AsyncGenerator[str, None]:
        """
        Generate completion with streaming.

        Args:
            prompt: Prompt text
            model: Model to use
            system_prompt: System prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens

        Yields:
            Generated text chunks
        """
        model = model or self.default_model

        if model not in self.stats:
            self.stats[model] = ModelStats(model=model)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        start_time = time.time()
        total_tokens = 0

        try:
            # Stream response using thread pool
            async def _stream_impl():
                return self.client.chat(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    num_predict=max_tokens,
                    stream=True,
                )

            response_stream = await asyncio.to_thread(_stream_impl)

            for chunk in response_stream:
                if isinstance(chunk, dict) and "message" in chunk:
                    text = chunk["message"].get("content", "")
                    if text:
                        total_tokens += len(text.split())
                        yield text

            # Update stats
            latency = (time.time() - start_time) * 1000
            self.stats[model].latency_ms = latency
            self.stats[model].tokens_generated += total_tokens

            logger.debug(f"Streamed {total_tokens} tokens in {latency:.0f}ms")

        except Exception as e:
            self.stats[model].errors += 1
            logger.error(f"Streaming generation failed: {e}")
            raise

    def get_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get performance statistics for all models.

        Returns:
            Dictionary of model statistics
        """
        return {model: stats.to_dict() for model, stats in self.stats.items()}

    def reset_stats(self) -> None:
        """Reset all statistics."""
        self.stats.clear()
        logger.info("Statistics reset")


async def main():
    """Test the OllamaClient."""
    client = OllamaClient()

    # Test connection
    connected = await client.verify_connection()
    if not connected:
        print(f"❌ Failed to connect: {client._last_error}")
        return

    # List models
    models = await client.list_available_models()
    print(f"✓ Available models: {models}")

    # Test generation
    try:
        response = await client.generate(
            prompt="Explain code smell in one sentence.",
            model="llama3:8b",
        )
        print(f"✓ Generation successful: {response[:100]}...")

        # Print stats
        stats = client.get_stats()
        print(f"\n✓ Stats: {json.dumps(stats, indent=2)}")

    except Exception as e:
        print(f"❌ Generation failed: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
