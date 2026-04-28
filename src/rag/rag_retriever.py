"""
RAG Retriever Agent (Custodian)
Responsible for finding relevant examples from knowledge base.

Architecture: Custodian agent in multi-agent system
Finds relevant examples from MaRV dataset using ChromaDB retrieval
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from src.rag.vector_store import VectorStore
from src.rag.embedding_service import EmbeddingService
from src.utils.logger import log_rag_retrieval, log_agent_event
from config import RAG_CONFIG

logger = logging.getLogger(__name__)


class RAGRetriever:
    """
    RAG Retriever Agent (Custodian Role).

    Responsible for:
    - Finding relevant examples from knowledge base
    - Ranking examples by relevance
    - Filtering by similarity threshold
    - Augmenting context for LLM prompts
    - Tracking retrieval metrics

    Example:
        >>> retriever = RAGRetriever()
        >>> examples = await retriever.find_relevant_examples(code_snippet, "Long Method")
    """

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        embedding_service: Optional[EmbeddingService] = None,
    ):
        """
        Initialize RAG retriever.

        Args:
            vector_store: ChromaDB vector store (created if None)
            embedding_service: Embedding service (created if None)
        """
        self.agent_name = "rag_retriever_custodian"
        self.vector_store = vector_store or VectorStore()
        self.embedding_service = embedding_service or EmbeddingService()

        # Statistics
        self.total_queries = 0
        self.cache_hits = 0
        self.retrieval_cache = {}  # Simple in-memory cache for M4 Pro

        logger.info("RAG Retriever initialized: %s", self.agent_name)  # noqa: G201

    async def find_relevant_examples(
        self,
        code: str,
        smell_type: Optional[str] = None,
        top_k: int = RAG_CONFIG["top_k"],
        use_cache: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Find relevant examples from knowledge base.

        Args:
            code: Code to find examples for
            smell_type: Specific smell type to search for (optional)
            top_k: Number of examples to retrieve
            use_cache: Use retrieval cache

        Returns:
            List of relevant examples with scores
        """
        start_time = datetime.now()

        # H6: Key the retrieval cache on a SHA256 digest of the full code
        # rather than its first 50 chars — otherwise any two snippets that
        # share an identical prefix (imports, boilerplate) collide and get
        # each other's examples, silently poisoning retrieval accuracy.
        import hashlib

        code_hash = hashlib.sha256(code.encode("utf-8")).hexdigest()[:16]
        cache_key = f"{code_hash}_{smell_type}_{top_k}"

        # Check cache
        if use_cache and cache_key in self.retrieval_cache:
            self.cache_hits += 1
            latency = (datetime.now() - start_time).total_seconds()

            log_rag_retrieval(
                self.agent_name,
                f"smell:{smell_type}",
                len(self.retrieval_cache[cache_key]),
                self.retrieval_cache[cache_key][0]["similarity"],
                latency,
                cached=True,
            )

            return self.retrieval_cache[cache_key]

        try:
            # Create query
            if smell_type:
                query = f"Code example of {smell_type} code smell: {code[:200]}"
            else:
                query = code[:200]

            # Search vector store
            results = self.vector_store.search_with_mmr(
                query=query,
                top_k=top_k,
                threshold=RAG_CONFIG["similarity_threshold"],
                diversity_lambda=RAG_CONFIG["diversity_lambda"],
            )

            # Format results
            examples = [
                {
                    "code": result["document"],
                    "smell_type": result["metadata"].get("smell_type", "unknown"),
                    "similarity": result["similarity"],
                    "metadata": result["metadata"],
                }
                for result in results
            ]

            # Cache results
            if use_cache:
                self.retrieval_cache[cache_key] = examples

            latency = (datetime.now() - start_time).total_seconds()
            self.total_queries += 1

            log_rag_retrieval(
                self.agent_name,
                f"smell:{smell_type}",
                len(examples),
                examples[0]["similarity"] if examples else 0.0,
                latency,
                cached=False,
            )

            logger.info(
                "Found %d relevant examples (similarity: %.3f)",  # noqa: G201
                len(examples),
                examples[0]['similarity'] if examples else 0.0
            )

            return examples

        except ValueError as e:
            logger.error("Retrieval failed: %s", e)  # noqa: G201
            return []

    async def rank_by_relevance(
        self,
        examples: List[Dict[str, Any]],
        smell_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Rank examples by relevance to smell type.

        Args:
            examples: List of examples to rank
            smell_type: Smell type for ranking

        Returns:
            Ranked examples
        """
        if not examples:
            return []

        # Simple ranking: prefer matching smell type
        if smell_type:
            ranked = sorted(
                examples,
                key=lambda x: (
                    -1 if x.get("smell_type", "").lower() == smell_type.lower() else 0,
                    -x["similarity"],  # Then by similarity
                ),
            )
        else:
            # Just sort by similarity
            ranked = sorted(examples, key=lambda x: -x["similarity"])

        logger.debug("Ranked %d examples", len(ranked))  # noqa: G201
        return ranked

    async def augment_context(
        self,
        examples: List[Dict[str, Any]],
        max_context_lines: int = 100,
    ) -> str:
        """
        Format examples for LLM prompt augmentation.

        Args:
            examples: List of examples
            max_context_lines: Maximum lines of context

        Returns:
            Formatted context string
        """
        if not examples:
            return ""

        context_lines = []
        total_lines = 0

        for i, example in enumerate(examples, 1):
            # Get code lines
            code_lines = example["code"].split('\n')

            # Add header
            smell_type = example.get("smell_type", "unknown")
            similarity = example.get("similarity", 0.0)
            context_lines.append(
                f"## Example {i}: {smell_type} (similarity: {similarity:.2%})"
            )
            total_lines += 1

            # Add code (truncate if needed)
            remaining = max_context_lines - total_lines
            if remaining <= 0:
                break

            lines_to_add = min(len(code_lines), remaining - 2)
            context_lines.extend(code_lines[:lines_to_add])
            total_lines += lines_to_add

            context_lines.append("")  # Separator
            total_lines += 1

        context = "\n".join(context_lines)
        logger.debug("Augmented context: %d chars", len(context))  # noqa: G201

        return context

    def get_stats(self) -> Dict[str, Any]:
        """
        Get retrieval statistics.

        Returns:
            Statistics dictionary
        """
        hit_rate = (
            (self.cache_hits / self.total_queries)
            if self.total_queries > 0
            else 0.0
        )

        return {
            "agent": self.agent_name,
            "total_queries": self.total_queries,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": hit_rate,
            "vector_store_stats": self.vector_store.get_stats(),
        }

    def clear_cache(self) -> None:
        """Clear the retrieval cache."""
        self.retrieval_cache.clear()
        logger.info("Retrieval cache cleared")


async def test_rag_retriever():
    """Test RAG retriever."""
    print("✓ Testing RAG Retriever...")

    retriever = RAGRetriever()
    log_agent_event(retriever.agent_name, "initialization_complete")

    # Test code
    sample_code = """
def process_data(data):
    result = None
    if data:
        for item in data:
            for sub_item in item:
                for value in sub_item:
                    result = process_value(value)
    return result
"""

    # Find relevant examples
    print("  Searching for Long Method examples...")
    examples = await retriever.find_relevant_examples(
        sample_code,
        "Long Method",
        top_k=3,
    )
    print(f"  Found {len(examples)} examples")

    # Rank examples
    ranked = await retriever.rank_by_relevance(examples, "Long Method")
    print(f"  Ranked {len(ranked)} examples")

    # Augment context
    context = await retriever.augment_context(ranked)
    print(f"  Augmented context: {len(context)} chars")

    # Get stats
    stats = retriever.get_stats()
    print("\n✓ Statistics:")
    print(f"  Total queries: {stats['total_queries']}")
    print(f"  Cache hit rate: {stats['cache_hit_rate']:.1%}")

    log_agent_event(retriever.agent_name, "testing_complete", {"examples_found": len(examples)})


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_rag_retriever())
