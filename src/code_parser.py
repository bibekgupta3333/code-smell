"""
Code Parser and Analysis Utilities
Validates code syntax, detects languages, parses AST, extracts metrics.

Architecture: Supports code preprocessing for agent analysis
"""

import ast
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ProgrammingLanguage(str, Enum):
    """Supported programming languages."""
    PYTHON = "python"
    JAVA = "java"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    CSHARP = "csharp"
    GOLANG = "golang"
    RUST = "rust"
    CPP = "cpp"
    UNKNOWN = "unknown"


@dataclass
class CodeMetrics:
    """Metrics extracted from code."""

    language: ProgrammingLanguage
    total_lines: int
    code_lines: int  # Non-empty, non-comment lines
    comment_lines: int
    functions: int
    classes: int
    max_function_length: int
    average_function_length: float


@dataclass
class CodeStructure:
    """AST structures extracted from code."""

    language: ProgrammingLanguage
    classes: List[str]  # Class names
    functions: List[str]  # Top-level function names
    methods: List[Tuple[str, str]]  # (class name, method name) pairs
    imports: List[str]
    defined_identifiers: List[str]


class CodeParser:
    """
    Code parser for analysis and validation.

    Supports:
    - Syntax validation
    - Language detection
    - AST parsing (Python)
    - Code metric extraction
    - Structure analysis
    """

    # Language detection patterns
    LANGUAGE_PATTERNS = {
        ProgrammingLanguage.PYTHON: [r'def\s+\w+\s*\(', r'import\s+\w+', r'class\s+\w+'],
        ProgrammingLanguage.JAVA: [r'public\s+class\s+\w+', r'public\s+static\s+void', r'import\s+[^;]+;'],
        ProgrammingLanguage.JAVASCRIPT: [r'function\s+\w+\s*\(', r'const\s+\w+\s*=', r'import\s+'],
        ProgrammingLanguage.CSHARP: [r'public\s+class\s+\w+', r'public\s+void\s+\w+', r'using\s+\w+'],
        ProgrammingLanguage.TYPESCRIPT: [r'(interface|type)\s+\w+', r'function\s+\w+\s*\(', r'import\s+'],
    }

    def __init__(self):
        """Initialize code parser."""
        self.logger = logging.getLogger(__name__)

    def detect_language(self, code: str) -> ProgrammingLanguage:
        """
        Detect programming language from code.

        Args:
            code: Source code

        Returns:
            Detected language
        """
        if not code or not code.strip():
            return ProgrammingLanguage.UNKNOWN

        code_lower = code.lower()

        # Check for language-specific patterns
        for language, patterns in self.LANGUAGE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, code, re.IGNORECASE):
                    return language

        # Check for specific keywords
        if re.search(r'\b(def|import|class)\b', code):
            return ProgrammingLanguage.PYTHON
        elif re.search(r'\b(public|private|class|void)\b', code):
            return ProgrammingLanguage.JAVA
        elif re.search(r'\b(function|const|let|var)\b', code):
            return ProgrammingLanguage.JAVASCRIPT

        return ProgrammingLanguage.UNKNOWN

    def validate_python_syntax(self, code: str) -> Tuple[bool, Optional[str]]:
        """
        Validate Python code syntax.

        Args:
            code: Python source code

        Returns:
            (is_valid, error_message)
        """
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            error_msg = f"Syntax error at line {e.lineno}: {e.msg}"
            self.logger.warning(f"Python syntax error: {error_msg}")
            return False, error_msg
        except Exception as e:
            error_msg = f"Parse error: {str(e)}"
            self.logger.warning(f"Python parse error: {error_msg}")
            return False, error_msg

    def extract_python_structure(self, code: str) -> Optional[CodeStructure]:
        """
        Extract Python AST structure.

        Args:
            code: Python source code

        Returns:
            CodeStructure or None
        """
        try:
            tree = ast.parse(code)
        except Exception as e:
            self.logger.warning(f"Failed to parse Python code: {e}")
            return None

        classes = []
        functions = []
        methods = []
        imports = []

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes.append(node.name)
                # Extract methods
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        methods.append((node.name, item.name))

            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Only top-level functions
                if isinstance(node, ast.stmt) and not any(
                    isinstance(parent, ast.ClassDef) for parent in ast.walk(tree)
                ):
                    functions.append(node.name)

            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)

        return CodeStructure(
            language=ProgrammingLanguage.PYTHON,
            classes=classes,
            functions=functions,
            methods=methods,
            imports=imports,
            defined_identifiers=classes + functions + [m[1] for m in methods],
        )

    def extract_metrics(self, code: str) -> CodeMetrics:
        """
        Extract code metrics.

        Args:
            code: Source code

        Returns:
            CodeMetrics
        """
        lines = code.split('\n')
        total_lines = len(lines)

        # Count code and comment lines
        code_lines = 0
        comment_lines = 0
        in_multiline_comment = False

        for line in lines:
            stripped = line.strip()

            # Skip empty lines
            if not stripped:
                continue

            # Python/Java-style comments
            if stripped.startswith('#') or stripped.startswith('//'):
                comment_lines += 1
                continue

            # Multiline comments
            if '"""' in stripped or "'''" in stripped or '/*' in stripped:
                in_multiline_comment = not in_multiline_comment
                comment_lines += 1
                continue

            if in_multiline_comment:
                comment_lines += 1
                continue

            code_lines += 1

        # Count functions and classes
        language = self.detect_language(code)
        functions = 0
        classes = 0
        function_lengths = []
        current_function_start = None
        current_indent = 0

        if language == ProgrammingLanguage.PYTHON:
            for i, line in enumerate(lines):
                stripped = line.strip()
                indent = len(line) - len(line.lstrip())

                if stripped.startswith('def '):
                    if current_function_start is not None:
                        function_lengths.append(i - current_function_start)
                    current_function_start = i
                    current_indent = indent
                    functions += 1

                elif stripped.startswith('class '):
                    classes += 1

                elif current_function_start is not None:
                    # Check if we've exited the function (lower or same indent)
                    if stripped and indent <= current_indent and not stripped.startswith('@'):
                        if not stripped.startswith('def ') and not stripped.startswith('class '):
                            function_lengths.append(i - current_function_start)
                            current_function_start = None

            # Add last function
            if current_function_start is not None:
                function_lengths.append(len(lines) - current_function_start)

        # Calculate averages
        max_function_length = max(function_lengths) if function_lengths else 0
        avg_function_length = (
            sum(function_lengths) / len(function_lengths)
            if function_lengths
            else 0
        )

        return CodeMetrics(
            language=language,
            total_lines=total_lines,
            code_lines=code_lines,
            comment_lines=comment_lines,
            functions=functions,
            classes=classes,
            max_function_length=max_function_length,
            average_function_length=avg_function_length,
        )

    def split_into_functions(self, code: str) -> List[Tuple[str, str, int, int]]:
        """
        Split code into functions/methods.

        Args:
            code: Source code

        Returns:
            List of (function_name, code, start_line, end_line)
        """
        language = self.detect_language(code)

        if language == ProgrammingLanguage.PYTHON:
            return self._split_python_functions(code)

        # Fallback: return whole code as one unit
        return [("full_code", code, 1, len(code.split('\n')))]

    def _split_python_functions(self, code: str) -> List[Tuple[str, str, int, int]]:
        """Split Python code into functions."""
        try:
            tree = ast.parse(code)
        except Exception:
            return [("full_code", code, 1, len(code.split('\n')))]

        lines = code.split('\n')
        functions = []

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                start_line = node.lineno - 1
                end_line = node.end_lineno or len(lines)

                function_code = '\n'.join(lines[start_line:end_line])
                functions.append((node.name, function_code, start_line + 1, end_line))

        return functions

    def get_code_preview(self, code: str, max_lines: int = 10) -> str:
        """
        Get a preview of code (first N lines).

        Args:
            code: Source code
            max_lines: Maximum lines to show

        Returns:
            Code preview
        """
        lines = code.split('\n')[:max_lines]
        return '\n'.join(lines)


def test_code_parser():
    """Test code parser."""
    print("✓ Testing code parser...")

    parser = CodeParser()

    # Test language detection
    python_code = "def hello():\n    return 42"
    java_code = "public class Hello { }"

    py_lang = parser.detect_language(python_code)
    java_lang = parser.detect_language(java_code)

    print(f"  Python detection: {py_lang}")
    print(f"  Java detection: {java_lang}")

    # Test Python validation
    valid, error = parser.validate_python_syntax(python_code)
    print(f"  Python syntax valid: {valid}")

    invalid_py = "def broken syntax"
    invalid, error = parser.validate_python_syntax(invalid_py)
    print(f"  Invalid Python detected: {not invalid}")

    # Test metrics extraction
    metrics = parser.extract_metrics(python_code)
    print(f"  Metrics: {metrics.functions} function(s), {metrics.total_lines} lines")

    print("✓ Code parser test complete")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_code_parser()
