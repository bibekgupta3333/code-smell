"""
Code Parser and Analysis Utilities
Validates code syntax, detects languages, parses AST, extracts metrics.

Architecture: Supports code preprocessing for agent analysis
"""

import ast
import re
import logging
from typing import List, Optional, Tuple
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

    # Language detection patterns (order matters - check specific languages first)
    LANGUAGE_PATTERNS = {
        ProgrammingLanguage.JAVA: [r'public\s+class\s+\w+', r'public\s+static\s+void', r'import\s+[^;]+;'],
        ProgrammingLanguage.CSHARP: [r'public\s+class\s+\w+', r'public\s+void\s+\w+', r'using\s+\w+'],
        ProgrammingLanguage.TYPESCRIPT: [r'(interface|type)\s+\w+', r'function\s+\w+\s*\(', r'import\s+'],
        ProgrammingLanguage.JAVASCRIPT: [r'function\s+\w+\s*\(', r'const\s+\w+\s*=', r'import\s+'],
        ProgrammingLanguage.PYTHON: [r'def\s+\w+\s*\(', r'import\s+\w+', r'from\s+.+\s+import'],
    }

    def __init__(self):
        """Initialize code parser."""
        self.logger = logging.getLogger(__name__)

    def preprocess_code(self, code: str) -> str:
        """
        Clean terminal artifacts, prompts, and metadata from code submission.

        Removes:
        - Terminal prompt characters (➜, $, %, #)
        - Shell commands and command output
        - Git status indicators (git:(main), ✗, ✓)
        - Paths and directory indicators
        - ANSI color codes
        - Common shell output patterns

        Args:
            code: Raw code possibly contaminated with terminal artifacts

        Returns:
            Cleaned code ready for parsing
        """
        if not code:
            return code

        lines = code.split('\n')
        cleaned_lines = []

        for line in lines:
            # Skip empty lines (preserve them for formatting)
            if not line.strip():
                cleaned_lines.append(line)
                continue

            # Detect and skip terminal prompt lines
            # Pattern: prompt chars + optional path + optional git status + command
            # Examples:
            #   ➜  code-smell git:(main) ✗  cat test.py
            #   $ python script.py
            #   (.venv) ~/projects/code-smell:
            # Note: Match ONLY virtual env markers (parentheses containing env names, not arbitrary code)
            if re.search(r'(^[➜$%#]|git:\(|✗|✓|\(\.?v?env\S*\)\s|~?/[^ ]*:$)', line):
                # This line contains shell decorators; check if it's a command or just prompt
                if re.search(r'\b(git|cat|ls|cd|mkdir|rm|python|npm|yarn|pip|echo|grep)\b', line):
                    # Shell command detected, skip it
                    continue
                # If it ends with a colon or prompt marker with no code, skip it
                if re.search(r'(:\s*$|[➜$%#]\s*$)', line):
                    continue

            cleaned = line

            # Remove leading terminal prompt patterns
            # First, remove any leading prompt marker + space
            cleaned = re.sub(r'^[➜$%#]+\s*', '', cleaned)
            # Then remove any directory/git info that precedes actual code
            # Pattern: directory-name optional-git-status spaces optional-command-verb
            cleaned = re.sub(r'^[^\s]*\s+git:\([^)]+\)\s+[✗✓]?\s*', '', cleaned)
            cleaned = re.sub(r'^(\(.+?\)\s+)?~?/[^\s]*\s+', '', cleaned)  # Virtual env + path
            cleaned = re.sub(r'^\x1b\[[0-9;]*m', '', cleaned)  # ANSI color codes at start
            cleaned = re.sub(r'\x1b\[[0-9;]*m$', '', cleaned)  # Trailing ANSI codes

            # Skip lines that still look like shell output/commands, not code
            stripped = cleaned.strip()

            # Skip remaining shell command indicators
            if stripped.startswith(('git ', 'npm ', 'yarn ', 'python ', 'python3 ', 'pip ',
                                   'ls ', 'cd ', 'mkdir ', 'rm ', 'cat ', 'echo ', 'grep ')):
                continue

            # Skip pure logging/output lines (not code comments)
            if re.match(r'^(INFO|DEBUG|ERROR|WARNING):\s+\[', stripped):
                continue

            # Skip path-only lines
            if stripped in ('code-smell', '.venv', 'src', 'tests', 'Makefile') or \
               (stripped.startswith(('./', '~/', '/')) and not stripped.count(' ') > 1):
                continue

            cleaned_lines.append(cleaned)

        return '\n'.join(cleaned_lines)

    def detect_language(self, code: str) -> ProgrammingLanguage:
        """
        Detect programming language from code.

        Args:
            code: Source code (may contain terminal artifacts)

        Returns:
            Detected language
        """
        # Preprocess to remove terminal prompts and artifacts
        cleaned_code = self.preprocess_code(code)

        if not cleaned_code or not cleaned_code.strip():
            return ProgrammingLanguage.UNKNOWN

        # Check for language-specific patterns
        for language, patterns in self.LANGUAGE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, code, re.IGNORECASE):
                    return language

        # Fallback: check specific keywords (more specific languages first)
        # Java/C#: public/private/protected + void/class together indicates OOP language
        if re.search(r'\b(public|private|protected)\b', code) and re.search(r'\b(void|static|class)\b', code):
            return ProgrammingLanguage.JAVA
        # JavaScript: function keyword or const/let/var (strong indicators)
        elif re.search(r'\b(function|const|let|var)\b', code):
            return ProgrammingLanguage.JAVASCRIPT
        # Python: def or from...import (very specific to Python)
        elif re.search(r'(^|\s)(def|async\s+def|from\s+.+\s+import|import\s+)', code, re.MULTILINE):
            return ProgrammingLanguage.PYTHON

        return ProgrammingLanguage.UNKNOWN

    def validate_python_syntax(self, code: str) -> Tuple[bool, Optional[str]]:
        """
        Validate Python code syntax.

        Args:
            code: Python source code (may contain terminal artifacts)

        Returns:
            (is_valid, error_message)
        """
        # Preprocess to remove terminal prompts and artifacts
        cleaned_code = self.preprocess_code(code)

        try:
            ast.parse(cleaned_code)
            return True, None
        except SyntaxError as e:
            error_msg = "Syntax error at line %d: %s"
            self.logger.warning(error_msg, e.lineno, e.msg)  # noqa: G201
            return False, error_msg % (e.lineno, e.msg)
        except ValueError as e:
            error_msg = "Parse error: %s"
            self.logger.warning(error_msg, str(e))  # noqa: G201
            return False, error_msg % str(e)

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
        except ValueError as e:
            self.logger.warning("Failed to parse Python code: %s", e)  # noqa: G201
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
            code: Source code (may contain terminal artifacts)

        Returns:
            CodeMetrics
        """
        # Preprocess to remove terminal prompts and artifacts
        cleaned_code = self.preprocess_code(code)

        lines = cleaned_code.split('\n')
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
        except (SyntaxError, ValueError):
            # Return whole code if parsing fails (e.g., invalid Python or non-Python language)
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
    valid, _error = parser.validate_python_syntax(python_code)
    print(f"  Python syntax valid: {valid}")

    invalid_py = "def broken syntax"
    invalid, _error = parser.validate_python_syntax(invalid_py)
    print(f"  Invalid Python detected: {not invalid}")

    # Test metrics extraction
    metrics = parser.extract_metrics(python_code)
    print(f"  Metrics: {metrics.functions} function(s), {metrics.total_lines} lines")

    print("✓ Code parser test complete")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_code_parser()
