"""
Embedding Service for RAG
Handles text embeddings using HuggingFace sentence transformers.

Architecture: Implements Architecture Section 3 (Embeddings) and Section 9 (Caching)
"""

import logging
import json
from pathlib import Path
from typing import List, Optional, Dict, Any
import hashlib

import numpy as np
from sentence_transformers import SentenceTransformer

from config import (
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
    EMBEDDING_DEVICE,
    CACHE_DIR,
)

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Service for generating text embeddings with caching.

    Features:
    - Lazy loading of model for M4 Pro efficiency
    - Batch embedding with progress tracking
    - Disk-based caching of embeddings
    - Memory-efficient processing
    - Configurable embedding models

    Example:
        >>> service = EmbeddingService()
        >>> embeddings = service.embed_batch(texts)
        >>> print(embeddings.shape)  # (N, 384)
    """

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL,
        device: str = EMBEDDING_DEVICE,
        cache_dir: Path = CACHE_DIR,
    ):
        """
        Initialize embedding service.

        Args:
            model_name: HuggingFace model name
            device: Device to use (cpu, cuda, mps)
            cache_dir: Directory for caching embeddings
        """
        self.model_name = model_name
        self.device = device
        self.cache_dir = Path(cache_dir) / "embeddings"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Model loaded on first use
        self.model = None
        self._dimension = None

        logger.info(f"Embedding service initialized: {model_name} on {device}")

    def _load_model(self) -> SentenceTransformer:
        """
        Lazy load embedding model.

        Returns:
            Loaded model
        """
        if self.model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(
                self.model_name,
                device=self.device,
            )
            self._dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded. Dimension: {self._dimension}")

        return self.model

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        if self._dimension is None:
            self._load_model()
        return self._dimension

    def _get_cache_path(self, text_hash: str) -> Path:
        """
        Get cache file path for text hash.

        Args:
            text_hash: Hash of text

        Returns:
            Path to cache file
        """
        return self.cache_dir / f"{text_hash}.json"

    def _hash_text(self, text: str) -> str:
        """
        Hash text for caching.

        Args:
            text: Text to hash

        Returns:
            Hash string
        """
        return hashlib.md5(text.encode()).hexdigest()

    def _load_from_cache(self, text_hash: str) -> Optional[np.ndarray]:
        """
        Load embedding from cache.

        Args:
            text_hash: Text hash

        Returns:
            Cached embedding or None
        """
        cache_path = self._get_cache_path(text_hash)
        try:
            if cache_path.exists():
                with open(cache_path, 'r') as f:
                    data = json.load(f)
                    return np.array(data["embedding"])
        except Exception as e:
            logger.debug(f"Failed to load embedding from cache: {e}")

        return None

    def _save_to_cache(self, text_hash: str, embedding: np.ndarray) -> None:
        """
        Save embedding to cache.

        Args:
            text_hash: Text hash
            embedding: Embedding vector
        """
        cache_path = self._get_cache_path(text_hash)
        try:
            with open(cache_path, 'w') as f:
                json.dump({"embedding": embedding.tolist()}, f)
        except Exception as e:
            logger.debug(f"Failed to save embedding to cache: {e}")

    def embed_text(self, text: str, use_cache: bool = True) -> np.ndarray:
        """
        Embed a single text.

        Args:
            text: Text to embed
            use_cache: Use cached embedding if available

        Returns:
            Embedding vector (1D array)
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding")
            return np.zeros(self.dimension)

        text = text.strip()

        # Check cache
        text_hash = self._hash_text(text)
        if use_cache:
            cached = self._load_from_cache(text_hash)
            if cached is not None:
                logger.debug(f"Cache hit for embedding: {text[:30]}...")
                return cached

        # Generate embedding
        model = self._load_model()
        embedding = model.encode(text, convert_to_numpy=True)

        # Cache it
        self._save_to_cache(text_hash, embedding)

        logger.debug(f"Generated embedding: {text[:30]}... (dim={embedding.shape[0]})")
        return embedding

    def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        use_cache: bool = True,
    ) -> np.ndarray:
        """
        Embed multiple texts.

        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            use_cache: Use cached embeddings

        Returns:
            Matrix of embeddings (N x dimension)
        """
        if not texts:
            return np.zeros((0, self.dimension))

        # Filter empty texts
        valid_texts = [t.strip() for t in texts if t and t.strip()]
        if not valid_texts:
            return np.zeros((len(texts), self.dimension))

        logger.info(f"Embedding {len(valid_texts)} texts in batches of {batch_size}")

        embeddings = []
        cached_count = 0

        for i in range(0, len(valid_texts), batch_size):
            batch = valid_texts[i:i+batch_size]
            batch_embeddings = []

            for text in batch:
                # Check cache first
                text_hash = self._hash_text(text)
                if use_cache:
                    cached = self._load_from_cache(text_hash)
                    if cached is not None:
                        batch_embeddings.append(cached)
                        cached_count += 1
                        continue

                # Generate and cache
                embedding = self.embed_text(text, use_cache=False)
                if use_cache:
                    self._save_to_cache(text_hash, embedding)
                batch_embeddings.append(embedding)

            embeddings.extend(batch_embeddings)

        result = np.array(embeddings) if embeddings else np.zeros((len(valid_texts), self.dimension))
        logger.info(
            f"Embedded {len(valid_texts)} texts "
            f"({cached_count} from cache, {len(embeddings) - cached_count} generated)"
        )

        return result

    def get_stats(self) -> Dict[str, Any]:
        """
        Get embedding service statistics.

        Returns:
            Statistics dictionary
        """
        cache_files = list(self.cache_dir.glob("*.json"))
        cache_size_mb = sum(f.stat().st_size for f in cache_files) / (1024 * 1024)

        return {
            "model": self.model_name,
            "dimension": self.dimension,
            "device": self.device,
            "cached_embeddings": len(cache_files),
            "cache_size_mb": round(cache_size_mb, 2),
        }


async def test_embedding_service():
    """Test the embedding service."""
    service = EmbeddingService()

    # Test single embedding
    text = "This is a test code smell for Long Method"
    embedding = service.embed_text(text)
    print(f"✓ Single embedding: shape={embedding.shape}")

    # Test batch embedding
    texts = [
        "Long Method smell detection",
        "God Class refactoring",
        "Feature Envy pattern",
    ]
    embeddings = service.embed_batch(texts)
    print(f"✓ Batch embedding: shape={embeddings.shape}")

    # Test caching
    embedding_cached = service.embed_text(text)
    print(f"✓ Cached embedding: matches={np.allclose(embedding, embedding_cached)}")

    # Print stats
    stats = service.get_stats()
    print(f"\n✓ Statistics:")
    print(f"  Model: {stats['model']}")
    print(f"  Dimension: {stats['dimension']}")
    print(f"  Cached embeddings: {stats['cached_embeddings']}")
    print(f"  Cache size: {stats['cache_size_mb']} MB")


if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_embedding_service())
