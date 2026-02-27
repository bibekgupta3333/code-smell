"""
RAG Pipeline Service
Retrieval-Augmented Generation for code smell detection.

Architecture: Implements Architecture Section 5 (RAG Workflow)
Integrates retriever, embedder, and LLM.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional

from llm_client import OllamaClient
from embedding_service import EmbeddingService
from vector_store import VectorStore
from code_chunker import CodeChunker, CodeChunk
from prompt_templates import create_rag_prompt, create_production_analysis_prompt
from response_parser import ResponseParser, AnalysisResult
from config import RAG_CONFIG, DEFAULT_MODEL

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Retrieval-Augmented Generation pipeline.

    Features:
    - Similarity-based context retrieval
    - Prompt augmentation with examples
    - Context compression for token budget
    - Integration with LLM and embeddings
    - Streaming response support

    Example:
        >>> pipeline = RAGPipeline()
        >>> result = await pipeline.analyze_with_rag(code, smell_type="Long Method")
    """

    def __init__(
        self,
        llm_client: Optional[OllamaClient] = None,
        embedding_service: Optional[EmbeddingService] = None,
        vector_store: Optional[VectorStore] = None,
        code_chunker: Optional[CodeChunker] = None,
    ):
        """
        Initialize RAG pipeline.

        Args:
            llm_client: Ollama client (created if None)
            embedding_service: Embedding service (created if None)
            vector_store: Vector store (created if None)
            code_chunker: Code chunker (created if None)
        """
        self.llm_client = llm_client or OllamaClient()
        self.embedding_service = embedding_service or EmbeddingService()
        self.vector_store = vector_store or VectorStore()
        self.code_chunker = code_chunker or CodeChunker()
        self.response_parser = ResponseParser()

        logger.info("RAG Pipeline initialized")

    async def analyze_with_rag(
        self,
        code: str,
        smell_type: Optional[str] = None,
        top_k: int = RAG_CONFIG["top_k"],
        use_mmr: bool = True,
    ) -> AnalysisResult:
        """
        Analyze code with RAG context.

        Args:
            code: Code to analyze
            smell_type: Specific smell type to look for (optional)
            top_k: Number of context examples
            use_mmr: Use Max Marginal Relevance for diversity

        Returns:
            Analysis result with detected smells
        """
        try:
            # Chunk code
            chunks = self._chunk_code(code)
            if not chunks:
                logger.warning("No chunks created, falling back to full code")
                chunks = [CodeChunk(
                    content=code,
                    start_line=1,
                    end_line=len(code.split("\n")),
                    language="unknown",
                    type="block",
                    metadata={},
                )]

            # Retrieve context for each chunk
            context_examples = await self._retrieve_context(
                code=code,
                smell_type=smell_type,
                top_k=top_k,
                use_mmr=use_mmr,
            )

            # Create augmented prompt
            prompt = create_rag_prompt(
                code=code,
                context_examples=context_examples,
                smell_type=smell_type,
                max_context_lines=RAG_CONFIG.get("max_context_lines", 100),
            )

            # Generate response
            response = await self.llm_client.generate(
                prompt=prompt,
                model=DEFAULT_MODEL,
            )

            # Parse response
            result = self.response_parser.parse(response)

            logger.info(
                f"RAG analysis complete: {len(result.code_smells)} smells detected"
            )
            return result

        except Exception as e:
            logger.error(f"RAG analysis failed: {e}")
            return AnalysisResult(
                code_smells=[],
                summary=f"Analysis failed: {str(e)}",
                validity=False,
                confidence=0.0,
            )

    async def analyze_with_streaming(
        self,
        code: str,
        smell_type: Optional[str] = None,
        top_k: int = RAG_CONFIG["top_k"],
        callback=None,
    ):
        """
        Analyze code with RAG and streaming response.

        Args:
            code: Code to analyze
            smell_type: Specific smell type
            top_k: Number of context examples
            callback: Callback function for stream chunks

        Yields:
            Token chunks as they arrive
        """
        try:
            # Retrieve context
            context_examples = await self._retrieve_context(
                code=code,
                smell_type=smell_type,
                top_k=top_k,
            )

            # Create prompt
            prompt = create_rag_prompt(code, context_examples, smell_type)

            # Stream response
            async for chunk in self.llm_client.generate_stream(
                prompt=prompt,
                model=DEFAULT_MODEL,
            ):
                if callback:
                    callback(chunk)
                yield chunk

        except Exception as e:
            logger.error(f"Streaming analysis failed: {e}")
            yield f"Error: {str(e)}"

    async def initialize_knowledge_base(
        self,
        example_data: List[Dict[str, str]],
    ) -> int:
        """
        Initialize vector store with example data.

        Args:
            example_data: List of dicts with 'code' and 'smell_type' keys

        Returns:
            Number of documents added
        """
        if not example_data:
            logger.warning("No example data provided")
            return 0

        try:
            # Clear existing data
            self.vector_store.clear_collection()

            # Embed texts
            texts = [item.get("code", "") for item in example_data]

            # Create metadata
            metadatas = [
                {
                    "smell_type": item.get("smell_type", "unknown"),
                    "source": item.get("source", "examples"),
                }
                for item in example_data
            ]

            # Add to vector store
            count = self.vector_store.add_documents(texts, metadatas)

            logger.info(f"Knowledge base initialized with {count} documents")
            return count

        except Exception as e:
            logger.error(f"Failed to initialize knowledge base: {e}")
            return 0

    async def _retrieve_context(
        self,
        code: str,
        smell_type: Optional[str] = None,
        top_k: int = RAG_CONFIG["top_k"],
        use_mmr: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve context examples from vector store.

        Args:
            code: Code to analyze
            smell_type: Specific smell type
            top_k: Number of examples
            use_mmr: Use MMR reranking

        Returns:
            List of context examples
        """
        try:
            # Create query
            query = f"Code with {smell_type}" if smell_type else f"Code analysis: {code[:200]}"

            # Search vector store
            if use_mmr:
                results = self.vector_store.search_with_mmr(
                    query=query,
                    top_k=top_k,
                    diversity_lambda=RAG_CONFIG.get("diversity_lambda", 0.7),
                )
            else:
                results = self.vector_store.search(query=query, top_k=top_k)

            # Format results
            context_examples = [
                {
                    "code": result["document"],
                    "smell_type": result["metadata"].get("smell_type", "unknown"),
                    "similarity": result["similarity"],
                }
                for result in results
            ]

            logger.debug(f"Retrieved {len(context_examples)} context examples")
            return context_examples

        except Exception as e:
            logger.warning(f"Failed to retrieve context: {e}")
            return []

    def _chunk_code(self, code: str) -> List[CodeChunk]:
        """
        Chunk code for processing.

        Args:
            code: Source code

        Returns:
            List of code chunks
        """
        try:
            # Detect language (simple heuristic)
            language = self._detect_language(code)

            if language == "python":
                return self.code_chunker.chunk_python(code)
            elif language == "java":
                return self.code_chunker.chunk_java(code)
            else:
                return self.code_chunker.chunk_generic(code, language)

        except Exception as e:
            logger.warning(f"Chunking failed: {e}")
            return []

    def _detect_language(self, code: str) -> str:
        """
        Detect programming language.

        Args:
            code: Source code

        Returns:
            Language identifier
        """
        code_lower = code.lower()

        # Check for language-specific patterns
        if "import java." in code_lower or "public class" in code_lower:
            return "java"
        elif "import " in code_lower and "def " in code_lower:
            return "python"
        elif code_lower.startswith("<?php"):
            return "php"
        elif "function " in code_lower and "{" in code:
            return "javascript"

        return "unknown"

    def get_stats(self) -> Dict[str, Any]:
        """
        Get pipeline statistics.

        Returns:
            Statistics dictionary
        """
        return {
            "embedding_service": self.embedding_service.get_stats(),
            "vector_store": self.vector_store.get_stats(),
            "response_parser": {
                "valid_smell_types": len(
                    self.response_parser.__class__.VALID_SMELL_TYPES
                    if hasattr(self.response_parser.__class__, "VALID_SMELL_TYPES")
                    else []
                ),
            },
        }

    def close(self):
        """Close all resources."""
        try:
            self.vector_store.close()
            logger.info("RAG Pipeline closed")
        except Exception as e:
            logger.warning(f"Error closing pipeline: {e}")


async def test_rag_pipeline():
    """Test the RAG pipeline."""
    pipeline = RAGPipeline()

    # Sample code with a long method
    sample_code = '''
def process_user_data(user_id, data):
    """Process user data with many steps."""
    # Step 1: Validate
    if not user_id:
        return None

    # Step 2: Transform
    transformed = {}
    for key, value in data.items():
        if isinstance(value, str):
            transformed[key] = value.strip().lower()
        else:
            transformed[key] = value

    # Step 3: Enrich
    enriched = {
        **transformed,
        "user_id": user_id,
        "timestamp": "now",
    }

    # Step 4: Validate again
    if len(enriched) > 10:
        enriched = enriched[:10]

    return enriched
'''

    print("Testing RAG Pipeline...")

    # Test code detection
    lang = pipeline._detect_language(sample_code)
    print(f"✓ Detected language: {lang}")

    # Test chunking
    chunks = pipeline._chunk_code(sample_code)
    print(f"✓ Created {len(chunks)} chunks")

    # Get stats
    stats = pipeline.get_stats()
    print(f"\n✓ Pipeline statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_rag_pipeline())
