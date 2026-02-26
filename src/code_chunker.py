"""
Code Chunking Service
Splits code into semantically meaningful chunks for RAG.

Architecture: Implements Architecture Section 4 (Code Parsing)
Supports Python (AST) and generic language chunking.
"""

import ast
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CodeChunk:
    """Represents a code chunk."""

    content: str
    start_line: int
    end_line: int
    language: str
    type: str  # e.g., "function", "class", "block"
    metadata: Dict[str, Any]


class CodeChunker:
    """
    AST-based code chunking for semantic preservation.

    Features:
    - Python AST parsing for structure-aware chunking
    - Line-based overlap for context preservation
    - Generic language support (line-based)
    - Metadata tracking (function names, classes, etc.)

    Example:
        >>> chunker = CodeChunker(max_chunk_tokens=512, overlap_tokens=50)
        >>> chunks = chunker.chunk_python(code_text)
    """

    # Approximate tokens per line (conservative estimate)
    TOKENS_PER_LINE = 10

    def __init__(
        self,
        max_chunk_tokens: int = 512,
        overlap_tokens: int = 50,
    ):
        """
        Initialize chunker.

        Args:
            max_chunk_tokens: Maximum tokens per chunk
            overlap_tokens: Overlap between chunks
        """
        self.max_chunk_tokens = max_chunk_tokens
        self.overlap_tokens = overlap_tokens

        # Convert token limits to lines
        self.max_lines = max_chunk_tokens // self.TOKENS_PER_LINE
        self.overlap_lines = max(1, overlap_tokens // self.TOKENS_PER_LINE)

        logger.info(
            f"CodeChunker initialized: "
            f"max_lines={self.max_lines}, overlap_lines={self.overlap_lines}"
        )

    def chunk_python(self, code: str) -> List[CodeChunk]:
        """
        Chunk Python code using AST.

        Args:
            code: Python source code

        Returns:
            List of code chunks
        """
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            logger.warning(f"Failed to parse Python code: {e}")
            return self._chunk_generic(code, "python")

        chunks = []
        lines = code.split("\n")

        try:
            # Extract top-level definitions
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    chunk = self._extract_node_chunk(
                        node, lines, "function", code
                    )
                    if chunk:
                        chunks.append(chunk)

                elif isinstance(node, ast.ClassDef):
                    chunk = self._extract_node_chunk(
                        node, lines, "class", code
                    )
                    if chunk:
                        chunks.append(chunk)

        except Exception as e:
            logger.warning(f"Error extracting AST nodes: {e}")

        # If no chunks extracted, fall back to generic
        if not chunks:
            return self._chunk_generic(code, "python")

        # Sort by line number
        chunks.sort(key=lambda c: c.start_line)

        # Add overlapping context chunks
        final_chunks = []
        for i, chunk in enumerate(chunks):
            final_chunks.append(chunk)

            # Add overlap chunk if not last
            if i < len(chunks) - 1:
                overlap_chunk = self._create_overlap_chunk(
                    chunk, chunks[i+1], lines, code
                )
                if overlap_chunk:
                    final_chunks.append(overlap_chunk)

        logger.info(f"Chunked Python code into {len(final_chunks)} chunks")
        return final_chunks

    def chunk_java(self, code: str) -> List[CodeChunk]:
        """
        Chunk Java code (basic line-based approach).

        Args:
            code: Java source code

        Returns:
            List of code chunks
        """
        # Java support would require tree-sitter
        # For now, use generic chunking
        logger.info("Java chunking using generic approach (tree-sitter not available)")
        return self._chunk_generic(code, "java")

    def chunk_generic(self, code: str, language: str = "unknown") -> List[CodeChunk]:
        """
        Generic chunking for any language.

        Args:
            code: Source code
            language: Programming language

        Returns:
            List of code chunks
        """
        return self._chunk_generic(code, language)

    def _chunk_generic(self, code: str, language: str) -> List[CodeChunk]:
        """
        Generic line-based chunking.

        Args:
            code: Source code
            language: Programming language

        Returns:
            List of code chunks
        """
        lines = code.split("\n")
        chunks = []

        i = 0
        while i < len(lines):
            # Calculate chunk size
            chunk_end = min(i + self.max_lines, len(lines))

            # Get chunk content
            chunk_lines = lines[i:chunk_end]
            chunk_content = "\n".join(chunk_lines)

            # Skip empty chunks
            if not chunk_content.strip():
                i = chunk_end
                continue

            chunk = CodeChunk(
                content=chunk_content,
                start_line=i + 1,
                end_line=chunk_end,
                language=language,
                type="block",
                metadata={"strategy": "generic"},
            )
            chunks.append(chunk)

            # Move forward with overlap
            i = chunk_end - self.overlap_lines

        logger.info(
            f"Generic chunking: {len(chunks)} chunks "
            f"from {len(lines)} lines"
        )
        return chunks

    def _extract_node_chunk(
        self,
        node: ast.AST,
        lines: List[str],
        node_type: str,
        full_code: str,
    ) -> Optional[CodeChunk]:
        """
        Extract a chunk from an AST node.

        Args:
            node: AST node
            lines: Code lines
            node_type: Node type (function/class)
            full_code: Full source code

        Returns:
            Code chunk or None
        """
        try:
            start_line = node.lineno - 1
            end_line = node.end_lineno or len(lines)

            # Skip if exceeds limits
            if end_line - start_line > self.max_lines * 2:
                return None

            chunk_lines = lines[start_line:end_line]
            content = "\n".join(chunk_lines)

            # Get node name
            node_name = getattr(node, "name", "unknown")

            return CodeChunk(
                content=content,
                start_line=start_line + 1,
                end_line=end_line,
                language="python",
                type=node_type,
                metadata={
                    "name": node_name,
                    "strategy": "ast",
                    "decorators": self._get_decorators(node),
                },
            )

        except Exception as e:
            logger.warning(f"Failed to extract chunk from {node_type}: {e}")
            return None

    def _create_overlap_chunk(
        self,
        chunk1: CodeChunk,
        chunk2: CodeChunk,
        lines: List[str],
        full_code: str,
    ) -> Optional[CodeChunk]:
        """
        Create an overlap chunk between two chunks.

        Args:
            chunk1: First chunk
            chunk2: Second chunk
            lines: Code lines
            full_code: Full source code

        Returns:
            Overlap chunk or None
        """
        # Create overlap from end of chunk1 to part of chunk2
        overlap_start = chunk1.end_line - self.overlap_lines
        overlap_end = min(chunk2.start_line + self.overlap_lines, len(lines))

        if overlap_start >= overlap_end:
            return None

        chunk_lines = lines[overlap_start:overlap_end]
        content = "\n".join(chunk_lines)

        return CodeChunk(
            content=content,
            start_line=overlap_start + 1,
            end_line=overlap_end,
            language="python",
            type="overlap",
            metadata={"strategy": "overlap"},
        )

    def _get_decorators(self, node: ast.AST) -> List[str]:
        """
        Get decorator names from a node.

        Args:
            node: AST node

        Returns:
            List of decorator names
        """
        decorators = []
        if hasattr(node, "decorator_list"):
            for dec in node.decorator_list:
                if isinstance(dec, ast.Name):
                    decorators.append(dec.id)
                elif isinstance(dec, ast.Call) and isinstance(dec.func, ast.Name):
                    decorators.append(dec.func.id)
        return decorators


def test_code_chunker():
    """Test the code chunker."""
    # Python code sample
    python_code = '''
def long_function(x, y, z):
    """A function that does many things."""
    result = x + y
    result = result * z

    if result > 100:
        result = result - 50

    return result

class MyClass:
    """A sample class."""

    def method1(self):
        pass

    def method2(self):
        pass
'''

    chunker = CodeChunker(max_chunk_tokens=256, overlap_tokens=50)

    # Test Python chunking
    chunks = chunker.chunk_python(python_code)
    print(f"✓ Python chunking: {len(chunks)} chunks")
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i+1}: {chunk.type} (lines {chunk.start_line}-{chunk.end_line})")

    # Test generic chunking
    chunks = chunker.chunk_generic(python_code, "generic")
    print(f"\n✓ Generic chunking: {len(chunks)} chunks")

    # Test with invalid code
    try:
        invalid_code = "def broken syntax here"
        chunks = chunker.chunk_python(invalid_code)
        print(f"\n✓ Invalid Python handled: {len(chunks)} chunks")
    except Exception as e:
        print(f"✓ Invalid Python error handled: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_code_chunker()
