"""
Enhanced Code Smell Detector - Comprehensive Multi-Smell Detection
Detects all 10 types of code smells using metric-based + LLM-based analysis

Supported Smells:
  1. Long Method - Functions that do too much (>50-100 LOC)
  2. Deep Nesting - Excessive nesting depth (>3-4 levels)
  3. Duplicated Code - Identical/similar code blocks (>85% similarity)
  4. High Cyclomatic Complexity - Too many decision paths (>10-15)
  5. God Class - Classes with too many responsibilities (>15 methods)
  6. Data Clump - Variables always used together (3+ functions)
  7. Magic Numbers - Hard-coded values with no meaning
  8. Unused Variables - Dead code, unused imports
  9. Inconsistent Naming - Unclear or inconsistent names
  10. Missing Error Handling - Risky operations without exception handling
"""

import asyncio
import logging
import re
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
from time import time

try:
    from langchain_ollama import ChatOllama
except ImportError:
    from langchain_community.chat_models import ChatOllama

from langchain_core.tools import tool

from src.llm.llm_client import OllamaClient
from src.llm.response_parser import ResponseParser
from src.rag.rag_retriever import RAGRetriever
from src.utils.common import CodeSmellFinding, SeverityLevel, parse_smell_severity
from src.utils.logger import log_agent_event, log_detection_result, log_llm_request, log_llm_response
from config import DEFAULT_MODEL

logger = logging.getLogger(__name__)


# ============================================================================
# Metric Computation Functions
# ============================================================================

def _compute_cyclomatic_complexity(code: str) -> int:
    """Compute cyclomatic complexity (approximate via decision points)."""
    decisions = len(re.findall(r'\b(if|elif|else|for|while|and|or|except|case)\b', code))
    return max(1, decisions)


def _compute_max_nesting_depth(code: str) -> int:
    """Compute maximum nesting depth by tracking bracket/brace/paren depth."""
    max_depth = 0
    current_depth = 0
    for char in code:
        if char in '{([':
            current_depth += 1
            max_depth = max(max_depth, current_depth)
        elif char in '})]':
            current_depth = max(0, current_depth - 1)
    return max_depth


def _compute_lines_of_code(code: str) -> int:
    """Compute lines of code (excluding empty lines and comments)."""
    lines = [line.strip() for line in code.splitlines()]
    return len([l for l in lines if l and not l.startswith('#')])


def _find_duplicate_blocks(code: str, block_size: int = 5) -> List[Tuple[str, float]]:
    """Find duplicate code blocks using token similarity."""
    lines = code.splitlines()
    blocks = []
    duplicates = []

    for i in range(len(lines) - block_size):
        block = '\n'.join(lines[i:i+block_size])
        tokens_block = set(block.split())

        for existing in blocks:
            tokens_existing = set(existing.split())
            if tokens_block and tokens_existing:
                sim = len(tokens_block & tokens_existing) / max(len(tokens_block), len(tokens_existing))
                if sim > 0.85:
                    duplicates.append((block[:50], sim))
        blocks.append(block)

    return duplicates


def _count_magic_numbers(code: str) -> List[str]:
    """Find hard-coded numbers and strings without context."""
    magic_numbers = re.findall(r'\b(\d{2,})\b', code)
    magic_strings = re.findall(r"['\"]([a-zA-Z]{4,})['\"]", code)
    return list(set(magic_numbers + magic_strings))


def _find_unused_variables(code: str) -> List[str]:
    """Detect likely unused variables using basic SSA-like analysis."""
    # Find assignments
    assigned = re.findall(r'^[\s]*([a-zA-Z_]\w*)\s*=', code, re.MULTILINE)

    # Find usage
    all_identifiers = re.findall(r'\b([a-zA-Z_]\w*)\b', code)
    used_set = set(all_identifiers)

    # Variables assigned but not used (except common patterns)
    common_patterns = {'self', 'cls', '__name__', 'Exception', 'None', 'True', 'False'}
    unused = [v for v in set(assigned) if v not in used_set and v not in common_patterns]

    return unused


def _analyze_naming_consistency(code: str) -> Dict[str, Any]:
    """Analyze naming convention consistency (snake_case vs camelCase)."""
    snake_case_vars = len(re.findall(r'\b[a-z_][a-z0-9_]*\b', code))
    camel_case_vars = len(re.findall(r'\b[a-z][a-zA-Z0-9]*[A-Z]\w*\b', code))
    single_letter_vars = len(re.findall(r'\b(?<![a-zA-Z0-9_])[a-zA-Z](?![a-zA-Z0-9_])\b', code))

    total_vars = snake_case_vars + camel_case_vars
    inconsistency = 0.0 if total_vars == 0 else abs(snake_case_vars - camel_case_vars) / total_vars

    return {
        "snake_case": snake_case_vars,
        "camel_case": camel_case_vars,
        "single_letter": single_letter_vars,
        "inconsistency_score": inconsistency
    }


def _detect_risky_operations(code: str) -> List[str]:
    """Detect risky operations (file I/O, network, parsing) without error handling."""
    risky_ops = []
    risky_patterns = [
        (r'\bopen\s*\(', 'File I/O'),
        (r'\brequests\.(get|post|put|delete)', 'HTTP request'),
        (r'\bos\.remove|mkdir', 'File system operation'),
        (r'\bjson\.load\(', 'JSON parsing'),
        (r'\bint\s*\(|float\s*\(|dict\s*\(', 'Type conversion'),
        (r'\[\s*-?\d+\s*\]', 'List indexing'),
        (r'\bpickle\.load', 'Pickle deserialization'),
    ]

    has_error_handling = 'try' in code and 'except' in code

    for pattern, description in risky_patterns:
        if re.search(pattern, code):
            if not has_error_handling:
                risky_ops.append(description)

    return risky_ops


# ============================================================================
# Additional SonarQube-style Heuristics
# ============================================================================

def _detect_long_parameter_list(code: str, threshold: int = 5) -> List[Tuple[str, int]]:
    """Return list of (function_name, param_count) exceeding threshold."""
    hits: List[Tuple[str, int]] = []
    # Python: def name(params)
    for match in re.finditer(r"def\s+(\w+)\s*\(([^)]*)\)", code, re.DOTALL):
        name, raw_params = match.group(1), match.group(2)
        params = [p.strip() for p in raw_params.split(",") if p.strip() and p.strip() not in ("self", "cls")]
        if len(params) >= threshold:
            hits.append((name, len(params)))
    # Java/JS: methodName(params) preceded by modifier/type
    for match in re.finditer(r"(?:function\s+|public\s+|private\s+|protected\s+|static\s+)\s*\w+\s+(\w+)\s*\(([^)]*)\)", code):
        name, raw_params = match.group(1), match.group(2)
        params = [p.strip() for p in raw_params.split(",") if p.strip()]
        if len(params) >= threshold:
            hits.append((name, len(params)))
    return hits


def _detect_switch_statements(code: str, min_branches: int = 4) -> List[int]:
    """Return list of branch counts for switch/long if-elif chains."""
    results: List[int] = []
    # Java/JS/C-style switch
    for match in re.finditer(r"switch\s*\(", code):
        start = match.end()
        # Count "case" occurrences in next ~2000 chars (cheap heuristic)
        window = code[start:start + 2000]
        cases = len(re.findall(r"\bcase\s+", window))
        if cases >= min_branches:
            results.append(cases)
    # Python elif chain
    elif_chains = len(re.findall(r"\n\s*elif\s", code))
    if elif_chains >= min_branches:
        results.append(elif_chains + 1)
    return results


def _detect_empty_catch(code: str) -> int:
    """Detect empty/swallowed exception handlers (pass-only or no body)."""
    count = 0
    # Python: except ...: pass  (possibly with comment)
    count += len(re.findall(r"except[^\n:]*:\s*\n\s*pass\b", code))
    count += len(re.findall(r"except[^\n:]*:\s*\n\s*\.\.\.\s*\n", code))
    # Java/JS: catch (...) { } or catch (...) { /* comment only */ }
    count += len(re.findall(r"catch\s*\([^)]*\)\s*\{\s*\}", code))
    return count


def _detect_message_chains(code: str, min_chain: int = 4) -> int:
    """Return number of call chains with >= min_chain dotted calls."""
    # Matches patterns like a.b().c().d().e() - 4+ calls
    pattern = rf"(?:\b\w+\s*\(\s*\)\s*){{{min_chain},}}"
    chained = re.findall(rf"\w+(?:\.\w+\s*\([^)]*\)){{{min_chain - 1},}}", code)
    return len(chained)


def _detect_middle_man(code: str) -> int:
    """Detect methods that only delegate to another object (return x.method(...))."""
    # Python: def m(self, ...): return self.x.y(...)
    py_hits = len(re.findall(r"def\s+\w+\s*\([^)]*\)\s*:\s*\n\s*return\s+self\.\w+\.\w+\s*\(", code))
    # Java/JS one-liner delegators
    j_hits = len(re.findall(r"(?:public|private|protected)\s+\w+\s+\w+\s*\([^)]*\)\s*\{\s*return\s+this\.\w+\.\w+\s*\([^)]*\)\s*;\s*\}", code))
    return py_hits + j_hits


def _detect_commented_out_code(code: str) -> int:
    """Detect lines that look like commented-out code (not prose)."""
    hits = 0
    for line in code.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        # Python single-line comment with code-like tokens
        if stripped.startswith("#"):
            body = stripped.lstrip("#").strip()
            if re.search(r"^(if|for|while|def|class|return|import|from|print|\w+\s*=|\w+\s*\()", body):
                hits += 1
        # Java/JS // or /* */ single-line
        elif stripped.startswith("//"):
            body = stripped[2:].strip()
            if re.search(r"^(if|for|while|function|return|var|let|const|\w+\s*=|\w+\s*\()", body):
                hits += 1
    return hits


def _detect_primitive_obsession(code: str) -> int:
    """Heuristic: many primitive-typed parameters across functions suggests primitive obsession."""
    primitive_annotations = re.findall(r":\s*(str|int|float|bool|dict|list|tuple)\b", code)
    # Java/TS primitive parameter types
    primitive_annotations += re.findall(r"\b(String|int|long|double|boolean|float)\s+\w+\s*[,)]", code)
    return len(primitive_annotations)


def _detect_data_class(code: str) -> List[str]:
    """Detect classes that only hold data (getters/setters or fields, no real methods)."""
    hits: List[str] = []
    for cls_match in re.finditer(r"class\s+(\w+)[^:{]*[:{]([\s\S]*?)(?=\nclass\s|\Z)", code):
        name = cls_match.group(1)
        body = cls_match.group(2)
        methods = re.findall(r"def\s+(\w+)", body)
        non_trivial = [m for m in methods if m not in ("__init__", "__repr__", "__eq__", "__hash__") and not m.startswith("get_") and not m.startswith("set_")]
        if methods and not non_trivial:
            hits.append(name)
    return hits


def _detect_lazy_class(code: str, min_body_lines: int = 3) -> List[str]:
    """Detect classes with trivially small bodies (likely lazy classes)."""
    hits: List[str] = []
    for cls_match in re.finditer(r"class\s+(\w+)[^:{]*[:{]([\s\S]*?)(?=\nclass\s|\Z)", code):
        name = cls_match.group(1)
        body_lines = [ln for ln in cls_match.group(2).splitlines() if ln.strip() and not ln.strip().startswith("#")]
        if 0 < len(body_lines) <= min_body_lines:
            hits.append(name)
    return hits


def _detect_feature_envy(code: str, threshold: int = 4) -> int:
    """Heuristic: a method that accesses another object's attributes >= threshold times."""
    envy_count = 0
    for fn in re.finditer(r"def\s+\w+\s*\([^)]*\)\s*:\s*([\s\S]*?)(?=\n\s*def\s|\Z)", code):
        body = fn.group(1)
        # Count accesses like obj.field or obj.method() where obj is NOT self/cls
        accesses = re.findall(r"\b(?!self\b|cls\b)(\w+)\.\w+", body)
        if accesses:
            from collections import Counter
            top_owner, top_count = Counter(accesses).most_common(1)[0]
            self_accesses = len(re.findall(r"\bself\.\w+", body))
            if top_count >= threshold and top_count > self_accesses:
                envy_count += 1
    return envy_count


# ============================================================================
# Helper Functions (Non-Decorated)
# ============================================================================

def _analyze_code_structure_impl(code: str) -> str:
    """Analyze code structure and extract metrics."""
    from src.analysis.code_parser import CodeParser
    try:
        parser = CodeParser()
        metrics = parser.extract_metrics(code)
        analysis = f"""Code Structure Analysis:
- Functions: {metrics.functions}
- Classes: {metrics.classes}
- Total Lines: {metrics.total_lines}
- Code Lines: {metrics.code_lines}
- Comment Lines: {metrics.comment_lines}
- Avg Function Length: {metrics.average_function_length:.1f}
- Language: {parser.detect_language(code)}"""
        return analysis
    except Exception as e:
        return f"Error analyzing structure: {str(e)}"


def _classify_severity_impl(smell_type: str, code_section: str) -> str:
    """Classify severity level based on smell type and code characteristics."""
    code_length = len(code_section)
    smell_lower = smell_type.lower()

    severity_map = {
        "long method": SeverityLevel.HIGH if code_length > 100 else SeverityLevel.MEDIUM,
        "god class": SeverityLevel.HIGH if code_length > 150 else SeverityLevel.MEDIUM,
        "deep nesting": SeverityLevel.HIGH,
        "duplicated code": SeverityLevel.HIGH,
        "high cyclomatic complexity": SeverityLevel.HIGH,
        "data clump": SeverityLevel.MEDIUM,
        "magic numbers": SeverityLevel.LOW,
        "unused variables": SeverityLevel.MEDIUM,
        "inconsistent naming": SeverityLevel.LOW,
        "missing error handling": SeverityLevel.HIGH,
    }

    for smell_key, severity in severity_map.items():
        if smell_key in smell_lower:
            return str(severity.value).upper()

    return "MEDIUM"


def _retrieve_patterns_impl(code: str) -> str:
    """Retrieve similar code patterns from knowledge base."""
    try:
        retriever = RAGRetriever()
        results = retriever.retrieve_similar(code, top_k=3)
        if results:
            return f"Found {len(results)} similar patterns:\n" + "\n".join(
                [f"- {r.get('snippet', 'N/A')[:100]}..." for r in results[:3]]
            )
        return "No similar patterns found"
    except Exception as e:
        return f"Unable to retrieve patterns: {str(e)}"


def _extract_refactoring_impl(smell_type: str) -> str:
    """Extract refactoring suggestions for a specific smell type."""
    suggestions = {
        "long method": "Break into smaller methods with single responsibilities. Extract helper functions.",
        "deep nesting": "Use early returns, extract to helper functions, or apply guard clause pattern.",
        "duplicated code": "Extract common logic to shared utility functions or base classes.",
        "high cyclomatic complexity": "Simplify conditionals, use polymorphism, or employ strategy pattern.",
        "god class": "Decompose into multiple focused classes, each with one responsibility.",
        "data clump": "Create a class or dataclass to encapsulate related data.",
        "magic numbers": "Replace with named constants (MAX_RETRIES, TIMEOUT_MS, ADMIN_ROLE, etc).",
        "unused variables": "Remove unused declarations, dead code, and unreachable branches.",
        "inconsistent naming": "Standardize naming convention (snake_case or camelCase) across the codebase.",
        "missing error handling": "Wrap risky operations in try/except blocks with appropriate recovery.",
    }

    for key, suggestion in suggestions.items():
        if key in smell_type.lower():
            return suggestion

    return "Review and refactor the code structure."


# ============================================================================
# LangChain Tools
# ============================================================================

@tool
def analyze_code_structure(code: str) -> str:
    """Analyze code structure: functions, classes, LOC, complexity."""
    return _analyze_code_structure_impl(code)


@tool
def classify_severity_level(smell_type: str, code_section: str) -> str:
    """Classify severity of a detected code smell."""
    return _classify_severity_impl(smell_type, code_section)


@tool
def retrieve_similar_patterns(code: str) -> str:
    """Retrieve similar code patterns from knowledge base for context."""
    return _retrieve_patterns_impl(code)


@tool
def extract_refactoring_suggestions(smell_type: str) -> str:
    """Extract concrete refactoring suggestions for a smell type."""
    return _extract_refactoring_impl(smell_type)


# ============================================================================
# Enhanced Code Smell Detector
# ============================================================================

class CodeSmellDetectorEnhanced:
    """
    Enhanced Code Smell Detector - Comprehensive 10-Smell Detection.

    Combines:
    - Metric-based detection (LOC, complexity, nesting, etc.)
    - LLM-based semantic analysis with agentic reasoning
    - RAG context for pattern matching
    - Fallback heuristic detection with all 10 smell types
    """

    def __init__(
        self,
        specialization: Optional[str] = None,
        llm_client: Optional[OllamaClient] = None,
        rag_retriever: Optional[RAGRetriever] = None,
        model: Optional[str] = None,
    ):
        self.specialization = specialization or "Comprehensive Code Smell Detector"
        self.agent_name = f"detector_enhanced_{specialization.lower().replace(' ', '_')[:20]}" if specialization else "detector_enhanced"
        self.llm_client = llm_client or OllamaClient()
        self.rag_retriever = rag_retriever or RAGRetriever()
        self.response_parser = ResponseParser()
        self.model = model or DEFAULT_MODEL

        # Statistics
        self.detections_count = 0
        self.average_confidence = 0.0
        self.total_latency = 0.0

        # Initialize LLM
        self.llm = ChatOllama(
            model=self.model,
            base_url="http://localhost:11434",
            temperature=0.1,
        )

        log_agent_event(self.agent_name, "initialization", {
            "specialization": self.specialization,
            "model": self.model,
            "smells_supported": 10,
        })
        logger.info(f"Enhanced Code Smell Detector initialized: {self.agent_name}")

    async def detect_smells(
        self,
        code: str,
        smell_types: Optional[List[str]] = None,
        use_rag: bool = True,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[CodeSmellFinding]:
        """Detect all 10 types of code smells."""
        start_time = time()

        try:
            # Step 1: Analyze code structure
            structure_analysis = await asyncio.to_thread(_analyze_code_structure_impl, code)

            # Step 2: Retrieve RAG context
            rag_context = ""
            if use_rag:
                similar = await asyncio.to_thread(_retrieve_patterns_impl, code)
                rag_context = similar

            # Step 3: Call LLM with comprehensive prompt
            llm_input = f"""Analyze code for ALL 10 code smells:

1. LONG METHOD - Functions >50-100 LOC with high complexity
2. DEEP NESTING - Nesting depth >4 levels, hard to follow
3. DUPLICATED CODE - Identical/similar blocks (>85% similarity)
4. HIGH CYCLOMATIC COMPLEXITY - >10-15 decision paths
5. GOD CLASS - >15 methods, multiple unrelated responsibilities
6. DATA CLUMP - Same parameters repeated in 3+ functions
7. MAGIC NUMBERS - Hard-coded values without semantic meaning
8. UNUSED VARIABLES - Dead code, unreachable statements, unused imports
9. INCONSISTENT NAMING - Mixed naming conventions, unclear names
10. MISSING ERROR HANDLING - Risky operations without try/except

Code Structure:
{structure_analysis}

{f'Similar Patterns:{chr(10)}{rag_context}' if rag_context else ''}

Code:
```
{code[:3000]}
```

For EACH detected smell:
- Type (exact match from list above)
- Location (line range)
- Severity (CRITICAL/HIGH/MEDIUM/LOW)
- Explanation with specific metrics
- Refactoring suggestion
- Confidence 0.0-1.0

Output as JSON array. Include low-confidence findings for manual review.
Example: [{{"type":"Long Method", "location":"line 45-120", "severity":"HIGH", "explanation":"...", "refactoring":"...", "confidence":0.85}}]"""

            log_llm_request(self.agent_name, self.model, len(llm_input), llm_input[:80], temperature=0.1)

            llm_start = time()
            response = await asyncio.to_thread(self.llm.invoke, llm_input)
            llm_latency = time() - llm_start

            response_text = response.content if hasattr(response, 'content') else str(response)
            log_llm_response(self.agent_name, self.model, len(response_text), len(response_text.split()), llm_latency)

            # Step 4: Parse response
            findings = await self._parse_response(response_text, code)

            # Update stats
            self.detections_count += 1
            processing_time = time() - start_time
            self.total_latency += processing_time
            if findings:
                avg_conf = sum(f.confidence for f in findings) / len(findings)
                self.average_confidence = (self.average_confidence + avg_conf) / 2

            log_detection_result(
                self.agent_name, code[:50], len(findings),
                sum(1 for f in findings if f.severity == SeverityLevel.CRITICAL),
                sum(1 for f in findings if f.severity == SeverityLevel.HIGH),
                sum(1 for f in findings if f.severity == SeverityLevel.MEDIUM),
                sum(1 for f in findings if f.severity == SeverityLevel.LOW),
                processing_time
            )

            logger.info(f"Enhanced detector: {len(findings)} findings, {processing_time:.2f}s")
            return findings

        except Exception as e:
            logger.error(f"Detection error: {e}", exc_info=True)
            log_agent_event(self.agent_name, "detection_error", {"error": str(e)})
            return []

    async def _parse_response(self, response: str, code: str) -> List[CodeSmellFinding]:
        """Parse LLM response into CodeSmellFinding objects."""
        findings = []
        try:
            result = self.response_parser.parse(response)
            for smell in result.code_smells:
                finding = CodeSmellFinding(
                    smell_type=smell.type,
                    location=smell.location,
                    severity=SeverityLevel(smell.severity.value),
                    explanation=smell.explanation.strip(),
                    refactoring=smell.refactoring,
                    confidence=result.confidence,
                    agent_name=self.agent_name,
                    timestamp=datetime.now().isoformat(),
                )
                findings.append(finding)
            if findings:
                return findings
        except Exception as e:
            logger.warning(f"Parse error: {e}, using fallback")

        # Fallback: Metric-based detection for all 10 smells
        return self._detect_all_smells_metric_based(code)

    def _detect_all_smells_metric_based(self, code: str) -> List[CodeSmellFinding]:
        """Metric-based detection for all 10 smell types."""
        findings = []
        lines = [l for l in code.splitlines() if l.strip()]
        line_count = len(lines)

        # 1. Long Method
        if line_count >= 50:
            findings.append(CodeSmellFinding(
                smell_type="Long Method",
                location=f"line 1-{line_count}",
                severity=SeverityLevel.HIGH if line_count > 100 else SeverityLevel.MEDIUM,
                explanation=f"Spans {line_count} LOC - exceeds recommended 50-100 line limit.",
                refactoring="Extract into smaller helper functions with single responsibilities.",
                confidence=0.68,
                agent_name=self.agent_name,
                timestamp=datetime.now().isoformat(),
            ))

        # 2. Deep Nesting
        nesting = _compute_max_nesting_depth(code)
        if nesting > 4:
            findings.append(CodeSmellFinding(
                smell_type="Deep Nesting",
                location=f"line 1-{line_count}",
                severity=SeverityLevel.HIGH,
                explanation=f"Maximum nesting depth: {nesting} levels - reduces readability.",
                refactoring="Use early returns, extract to helper functions, guard clauses.",
                confidence=0.70,
                agent_name=self.agent_name,
                timestamp=datetime.now().isoformat(),
            ))

        # 3. Duplicated Code
        dups = _find_duplicate_blocks(code)
        if dups:
            findings.append(CodeSmellFinding(
                smell_type="Duplicated Code",
                location=f"line 1-{line_count}",
                severity=SeverityLevel.HIGH,
                explanation=f"Found {len(dups)} duplicate blocks (>85% similarity).",
                refactoring="Extract common logic to shared utility or base class.",
                confidence=0.75,
                agent_name=self.agent_name,
                timestamp=datetime.now().isoformat(),
            ))

        # 4. High Cyclomatic Complexity
        cc = _compute_cyclomatic_complexity(code)
        if cc > 10:
            findings.append(CodeSmellFinding(
                smell_type="High Cyclomatic Complexity",
                location=f"line 1-{line_count}",
                severity=SeverityLevel.HIGH if cc > 15 else SeverityLevel.MEDIUM,
                explanation=f"Cyclomatic complexity: {cc} - too many decision paths.",
                refactoring="Refactor into smaller functions, use polymorphism, simplify conditionals.",
                confidence=0.69,
                agent_name=self.agent_name,
                timestamp=datetime.now().isoformat(),
            ))

        # 5. Data Clump
        sig_match = re.search(r"def\s+\w+\((.*?)\)", code)
        if sig_match:
            params = [p.strip() for p in sig_match.group(1).split(',') if p.strip() and p.strip() not in ['self', 'cls']]
            if len(params) >= 6:
                findings.append(CodeSmellFinding(
                    smell_type="Data Clump",
                    location="line 1",
                    severity=SeverityLevel.MEDIUM,
                    explanation=f"Function has {len(params)} parameters - suggests grouped data.",
                    refactoring="Encapsulate related parameters in a class/dataclass.",
                    confidence=0.65,
                    agent_name=self.agent_name,
                    timestamp=datetime.now().isoformat(),
                ))

        # 6. Magic Numbers
        magic = _count_magic_numbers(code)
        if len(magic) >= 3:
            findings.append(CodeSmellFinding(
                smell_type="Magic Numbers",
                location=f"line 1-{line_count}",
                severity=SeverityLevel.LOW,
                explanation=f"Found {len(magic)} hard-coded values without semantic meaning.",
                refactoring="Replace with named constants (MAX_RETRIES, TIMEOUT_MS, etc).",
                confidence=0.60,
                agent_name=self.agent_name,
                timestamp=datetime.now().isoformat(),
            ))

        # 7. Unused Variables
        unused = _find_unused_variables(code)
        if unused:
            findings.append(CodeSmellFinding(
                smell_type="Unused Variables",
                location=f"line 1-{line_count}",
                severity=SeverityLevel.MEDIUM,
                explanation=f"Found {len(unused)} likely unused variables: {', '.join(unused[:3])}",
                refactoring="Remove unused declarations and dead code branches.",
                confidence=0.58,
                agent_name=self.agent_name,
                timestamp=datetime.now().isoformat(),
            ))

        # 8. Inconsistent Naming
        naming = _analyze_naming_consistency(code)
        if naming['inconsistency_score'] > 0.6 and naming['single_letter'] > 2:
            findings.append(CodeSmellFinding(
                smell_type="Inconsistent Naming",
                location=f"line 1-{line_count}",
                severity=SeverityLevel.LOW,
                explanation=f"Inconsistent naming: snake_case={naming['snake_case']}, camelCase={naming['camel_case']}, single-letter={naming['single_letter']}",
                refactoring="Standardize naming convention; replace single-letter vars.",
                confidence=0.55,
                agent_name=self.agent_name,
                timestamp=datetime.now().isoformat(),
            ))

        # 9. Missing Error Handling
        risky = _detect_risky_operations(code)
        if risky:
            findings.append(CodeSmellFinding(
                smell_type="Missing Error Handling",
                location=f"line 1-{line_count}",
                severity=SeverityLevel.HIGH,
                explanation=f"Risky operations without error handling: {', '.join(risky)}",
                refactoring="Wrap in try/except blocks with appropriate error recovery.",
                confidence=0.72,
                agent_name=self.agent_name,
                timestamp=datetime.now().isoformat(),
            ))

        # 10. God Class
        methods = len(re.findall(r"def\s+\w+\s*\(", code))
        if methods > 15 and line_count > 300:
            findings.append(CodeSmellFinding(
                smell_type="God Class",
                location=f"line 1-{line_count}",
                severity=SeverityLevel.HIGH,
                explanation=f"{methods} methods, {line_count} LOC - violates single responsibility.",
                refactoring="Decompose into multiple focused classes.",
                confidence=0.66,
                agent_name=self.agent_name,
                timestamp=datetime.now().isoformat(),
            ))

        return findings

    def get_stats(self) -> Dict[str, Any]:
        """Get detector statistics."""
        return {
            "agent_name": self.agent_name,
            "specialization": self.specialization,
            "smells_supported": 10,
            "detections_count": self.detections_count,
            "average_confidence": self.average_confidence,
            "average_latency_ms": (self.total_latency / max(1, self.detections_count)) * 1000,
        }
