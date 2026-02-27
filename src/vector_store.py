"""
Vector Store Service using ChromaDB
Handles code smell example storage and retrieval for RAG.

Architecture: Implements Architecture Section 3 (Vector Store) and Section 9 (RAG)
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import chromadb

from config import CHROMADB_DIR, RAG_CONFIG

logger = logging.getLogger(__name__)


class VectorStore:
    """
    ChromaDB-based vector store for code smell examples.

    Features:
    - Persistent storage with collection support
    - Similarity search with scoring
    - Batch document insertion
    - Metadata filtering and retrieval
    - MMR (Max Marginal Relevance) reranking

    Example:
        >>> store = VectorStore()
        >>> store.add_documents(texts, metadata)
        >>> results = store.search("Long Method", top_k=5)
    """

    def __init__(
        self,
        persist_dir: Path = CHROMADB_DIR,
        collection_name: str = "code_smell_examples",
    ):
        """
        Initialize vector store.

        Args:
            persist_dir: Directory for persistent storage
            collection_name: Name of the collection
        """
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name

        # Ensure directory exists
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB with new API
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        logger.info(
            f"Vector store initialized: {collection_name} "
            f"at {self.persist_dir}"
        )

    def add_documents(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> int:
        """
        Add documents to the vector store.

        Args:
            texts: List of text documents
            metadatas: List of metadata dicts (optional)
            ids: List of document IDs (optional)

        Returns:
            Number of documents added
        """
        if not texts:
            return 0

        # Generate IDs if not provided
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(texts))]

        # Generate default metadata if not provided
        if metadatas is None:
            metadatas = [
                {"source": "default", "index": i}
                for i in range(len(texts))
            ]

        try:
            self.collection.add(
                ids=ids,
                documents=texts,
                metadatas=metadatas,
            )
            logger.info(f"Added {len(texts)} documents to collection")
            return len(texts)
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            return 0

    def search(
        self,
        query: str,
        top_k: int = RAG_CONFIG["top_k"],
        threshold: float = RAG_CONFIG["similarity_threshold"],
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents.

        Args:
            query: Query text
            top_k: Number of results
            threshold: Minimum similarity score

        Returns:
            List of results with documents, distances, and metadata
        """
        if not query or not query.strip():
            logger.warning("Empty query provided")
            return []

        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
            )

            # Format results
            formatted_results = []
            for i in range(len(results["documents"][0])):
                document = results["documents"][0][i]
                distance = results["distances"][0][i]
                metadata = results["metadatas"][0][i]

                # Convert distance to similarity (cosine distance -> similarity)
                similarity = 1 - distance

                # Filter by threshold
                if similarity >= threshold:
                    formatted_results.append({
                        "document": document,
                        "similarity": similarity,
                        "distance": distance,
                        "metadata": metadata,
                    })

            logger.debug(
                f"Search found {len(formatted_results)} results "
                f"for: {query[:50]}..."
            )
            return formatted_results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def search_with_mmr(
        self,
        query: str,
        top_k: int = RAG_CONFIG["top_k"],
        threshold: float = RAG_CONFIG["similarity_threshold"],
        diversity_lambda: float = RAG_CONFIG["diversity_lambda"],
    ) -> List[Dict[str, Any]]:
        """
        Search with Max Marginal Relevance reranking.

        MMR balances relevance and diversity:
        - lambda=1.0: Pure relevance
        - lambda=0.5: Balance
        - lambda=0.0: Pure diversity

        Args:
            query: Query text
            top_k: Number of results
            threshold: Minimum similarity
            diversity_lambda: Diversity parameter

        Returns:
            Reranked results
        """
        # Get initial results
        results = self.search(query, top_k=top_k*2, threshold=0.0)

        if len(results) <= top_k:
            # No need to rerank
            return [r for r in results if r["similarity"] >= threshold]

        try:
            # MMR reranking (simple implementation)
            selected = []
            remaining = results.copy()

            while remaining and len(selected) < top_k:
                if not selected:
                    # First result is the most relevant
                    selected.append(remaining.pop(0))
                else:
                    # Find the result that maximizes MMR
                    best_idx = 0
                    best_mmr = float("-inf")

                    for idx, result in enumerate(remaining):
                        relevance = result["similarity"]

                        # Compute diversity (min distance to selected)
                        diversity = min(
                            1 - self._similarity_between_dicts(
                                result["metadata"],
                                selected[i]["metadata"]
                            )
                            for i in range(len(selected))
                        )

                        # MMR score
                        mmr = diversity_lambda * relevance + (1 - diversity_lambda) * diversity

                        if mmr > best_mmr:
                            best_mmr = mmr
                            best_idx = idx

                    selected.append(remaining.pop(best_idx))

            # Filter by threshold
            return [r for r in selected if r["similarity"] >= threshold]

        except Exception as e:
            logger.warning(f"MMR reranking failed, returning top results: {e}")
            return [r for r in results[:top_k] if r["similarity"] >= threshold]

    def _similarity_between_dicts(self, dict1: Dict, dict2: Dict) -> float:
        """
        Basic similarity between two metadata dicts.

        Args:
            dict1: First dict
            dict2: Second dict

        Returns:
            Similarity score (0-1)
        """
        if not dict1 or not dict2:
            return 0.0

        # Simple: check if any values match
        matches = sum(1 for k, v in dict1.items() if k in dict2 and dict2[k] == v)
        total = max(len(dict1), len(dict2))

        return matches / total if total > 0 else 0.0

    def delete_documents(self, ids: List[str]) -> int:
        """
        Delete documents by IDs.

        Args:
            ids: Document IDs to delete

        Returns:
            Number of documents deleted
        """
        try:
            self.collection.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} documents")
            return len(ids)
        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            return 0

    def clear_collection(self) -> bool:
        """
        Clear all documents in collection.

        Returns:
            True if successful
        """
        try:
            # Get all document IDs
            results = self.collection.get()
            if results and results["ids"]:
                self.collection.delete(ids=results["ids"])
                logger.info(f"Cleared {len(results['ids'])} documents")
            return True
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """
        Get collection statistics.

        Returns:
            Statistics dictionary
        """
        try:
            results = self.collection.get()
            doc_count = len(results["ids"]) if results and results["ids"] else 0

            return {
                "collection": self.collection_name,
                "document_count": doc_count,
                "persist_dir": str(self.persist_dir),
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"error": str(e)}

    def close(self) -> None:
        """Close the vector store connection."""
        logger.info("Vector store closed")


async def test_vector_store():
    """Test the vector store."""
    store = VectorStore()

    # Clear collection
    store.clear_collection()

    # Add documents
    texts = [
        "This is a long method that exceeds 50 lines",
        "The God Class has too many responsibilities",
        "Feature Envy: method uses other object's data",
    ]
    metadatas = [
        {"smell_type": "Long Method", "dataset": "examples"},
        {"smell_type": "God Class", "dataset": "examples"},
        {"smell_type": "Feature Envy", "dataset": "examples"},
    ]

    added = store.add_documents(texts, metadatas)
    print(f"✓ Added {added} documents")

    # Search
    results = store.search("long methods", top_k=2)
    print(f"✓ Found {len(results)} search results")
    for r in results:
        print(f"  - Similarity: {r['similarity']:.3f}")

    # Get stats
    stats = store.get_stats()
    print(f"\n✓ Statistics:")
    print(f"  Documents: {stats['document_count']}")


if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_vector_store())
