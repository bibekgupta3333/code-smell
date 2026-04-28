"""
Code Smell Detector Agent (Member) - LangChain Deep Agent Integration
Detects code smells in code snippets using LLM with RAG context.

Architecture: LangChain Deep Agent using LangGraph's create_react_agent
Benefits:
  - Structured tool invocation with built-in context management
  - Optimized for short-duration analysis tasks
  - Automatic tool selection and reasoning
  - Enhanced context awareness for code smell detection
"""

import asyncio
import logging
import re
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
from time import time

try:
    # L1: langchain_ollama is the primary, supported path. The
    # langchain_community fallback stays in place for environments where the
    # dedicated package isn't installed (older images, minimal CI containers),
    # but we log which path we took so a silent drift is visible in logs.
    from langchain_ollama import ChatOllama
    _CHAT_OLLAMA_SOURCE = "langchain_ollama"
except ImportError:
    from langchain_community.chat_models import ChatOllama  # type: ignore[assignment]
    _CHAT_OLLAMA_SOURCE = "langchain_community"
    logging.getLogger(__name__).warning(
        "langchain_ollama not installed; falling back to langchain_community.chat_models.ChatOllama"
    )

from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from src.llm.llm_client import OllamaClient
from src.llm.prompt_templates import create_rag_prompt, get_system_prompt
from src.llm.response_parser import ResponseParser, AnalysisResult
from src.rag.rag_retriever import RAGRetriever
from src.utils.common import CodeSmellFinding, SeverityLevel, parse_smell_severity
from src.utils.logger import log_agent_event, log_detection_result, log_llm_request, log_llm_response
from config import DEFAULT_MODEL

# Import enhanced metric functions for all code smell types
from src.analysis.code_smell_detector_enhanced import (
    _compute_cyclomatic_complexity,
    _compute_max_nesting_depth,
    _find_duplicate_blocks,
    _count_magic_numbers,
    _find_unused_variables,
    _analyze_naming_consistency,
    _detect_risky_operations,
    _detect_long_parameter_list,
    _detect_switch_statements,
    _detect_empty_catch,
    _detect_message_chains,
    _detect_middle_man,
    _detect_commented_out_code,
    _detect_primitive_obsession,
    _detect_data_class,
    _detect_lazy_class,
    _detect_feature_envy,
    _detect_inappropriate_coupling,
    _detect_poor_naming,
    _detect_inconsistent_style,
    _detect_flag_arguments,
    _detect_generic_exceptions,
    _detect_hidden_errors,
)
from src.utils.smell_catalog import (
    CANONICAL_SMELLS,
    build_prompt_catalog_block,
    normalize_smell_type,
)

logger = logging.getLogger(__name__)



# ============================================================================
# Helper Functions (Non-Decorated, Used by AsyncIO)
# ============================================================================

def _analyze_code_structure_impl(code: str) -> str:
    """Implementation of code structure analysis (non-decorated for asyncio compatibility)."""
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


def _classify_severity_impl(smell_description: str, code_section: str) -> str:
    """Implementation of severity classification (non-decorated)."""
    code_length = len(code_section)
    description_lower = smell_description.lower()

    # Severity rules based on code size and issue type
    if "long method" in description_lower or "god class" in description_lower:
        if code_length > 100:
            return "CRITICAL: Method/Class exceeds 100 lines - high refactoring priority"
        elif code_length > 50:
            return "HIGH: Method/Class exceeds 50 lines - significant refactoring needed"
        elif code_length > 20:
            return "MEDIUM: Method/Class is moderately large - minor refactoring suggested"
        else:
            return "LOW: Method/Class size is acceptable"
    elif "feature envy" in description_lower or "data clumps" in description_lower:
        return "MEDIUM: Data structure coupling issue - consider consolidation"
    elif "code duplication" in description_lower:
        return "HIGH: Duplicate code detected - extract to shared function"
    elif "dead code" in description_lower:
        return "MEDIUM: Unused code found - cleanup recommended"
    else:
        return "MEDIUM: General code smell detected - review recommendation"


def _retrieve_patterns_impl(code: str) -> str:
    """Implementation of pattern retrieval (non-decorated)."""
    try:
        from src.rag.rag_retriever import RAGRetriever
        retriever = RAGRetriever()
        results = retriever.retrieve_similar(code, top_k=3)

        if results:
            return f"Found {len(results)} similar patterns in codebase:\n" + "\n".join(
                [f"- {r['snippet'][:100]}..." for r in results[:3]]
            )
        else:
            return "No similar patterns found in knowledge base"
    except Exception as e:
        return f"Unable to retrieve patterns: {str(e)}"


def _extract_refactoring_impl(code: str, smell_type: str) -> str:
    """Implementation of refactoring suggestions (non-decorated)."""
    suggestions = {
        "long method": "Break into smaller methods with single responsibilities. Consider helper functions.",
        "god class": "Split into multiple focused classes. Extract related functionality.",
        "feature envy": "Move methods closer to data they access. Reduce cross-class dependencies.",
        "data clumps": "Create a dedicated class/structure for grouped data.",
        "code duplication": "Extract common logic to a shared function or utility module.",
        "dead code": "Remove unused functions, variables, and imports.",
        "complex conditional": "Extract to boolean helper methods or use polymorphism.",
    }

    suggestion = suggestions.get(
        smell_type.lower(),
        "Review code structure and consider SOLID principles"
    )
    return f"{smell_type.title()}: {suggestion}"


# ============================================================================
# Deep Agent Tools for Code Smell Detection (LangChain Integration)
# ============================================================================

@tool
def analyze_code_structure(code: str) -> str:
    """Analyze and extract code structure information.

    Extracts: functions, classes, lines of code, complexity metrics.
    Used by Deep Agent for reasoning about code patterns.

    Args:
        code: Code snippet to analyze

    Returns:
        Structured analysis of code
    """
    return _analyze_code_structure_impl(code)


@tool
def classify_severity_level(smell_description: str, code_section: str) -> str:
    """Classify the severity level of a detected code smell.

    Deep Agent tool for severity classification with precise rules.

    Args:
        smell_description: Description of the code smell
        code_section: The problematic code section

    Returns:
        Severity level (LOW, MEDIUM, HIGH, CRITICAL)
    """
    return _classify_severity_impl(smell_description, code_section)


@tool
def retrieve_similar_patterns(code: str) -> str:
    """Retrieve similar code patterns from knowledge base.

    Deep Agent tool for context-aware RAG integration.

    Args:
        code: Code snippet to find similar patterns for

    Returns:
        Similar patterns and best practices
    """
    return _retrieve_patterns_impl(code)


@tool
def extract_refactoring_suggestions(code: str, smell_type: str) -> str:
    """Extract concrete refactoring suggestions for identified smell.

    Deep Agent tool for generating actionable refactoring advice.

    Args:
        code: Code section with smell
        smell_type: Type of code smell identified

    Returns:
        Specific refactoring suggestions
    """
    return _extract_refactoring_impl(code, smell_type)


# ============================================================================
# Code Smell Detector Agent - Deep Agent Implementation
# ============================================================================

class CodeSmellDetector:
    """
    Code Smell Detector - LangChain Deep Agent Implementation.

    Responsible for:
    - Analyzing code snippets for code smells using multi-tool reasoning
    - Managing RAG context for enhanced detection
    - Assigning severity levels with structured rules
    - Generating actionable refactoring suggestions
    - Supporting immediate feedback for short-running tasks

    Deep Agent Benefits:
    - Built-in tool coordination and context management
    - Optimized for focused, short-duration analysis tasks
    - Structured reasoning about code patterns

    Example:
        >>> detector = CodeSmellDetector(specialization="Long Method expert")
        >>> findings = await detector.detect_smells(code)
    """

    def __init__(
        self,
        specialization: Optional[str] = None,
        llm_client: Optional[OllamaClient] = None,
        rag_retriever: Optional[RAGRetriever] = None,
        model: Optional[str] = None,
    ):
        """
        Initialize Deep Agent detector.

        Args:
            specialization: Smell type specialization (e.g., "Long Method expert")
            llm_client: Ollama client for LLM calls
            rag_retriever: RAG retriever for context
            model: Model to use for LLM inference (default: DEFAULT_MODEL)
        """
        self.specialization = specialization or "General code smell detector"
        self.agent_name = f"detector_{specialization.lower().replace(' ', '_')[:30]}" if specialization else "detector_general"
        self.llm_client = llm_client or OllamaClient()
        self.rag_retriever = rag_retriever or RAGRetriever()
        self.response_parser = ResponseParser()
        self.model = model or DEFAULT_MODEL

        # Statistics
        self.detections_count = 0
        self.false_positives = 0
        self.average_confidence = 0.0
        self.total_latency = 0.0
        self.tools_invoked_count = 0

        # Initialize LangChain ChatOllama for Deep Agent
        self.llm = ChatOllama(
            model=self.model,
            base_url="http://localhost:11434",
            temperature=0.1,
        )

        # Deep Agent tools
        self.tools = [
            analyze_code_structure,
            classify_severity_level,
            retrieve_similar_patterns,
            extract_refactoring_suggestions,
        ]

        # Create tool registry for manual invocation (Ollama doesn't support native tool binding)
        # This implements Deep Agent pattern with explicit tool orchestration
        self.tool_registry = {
            "analyze_code_structure": analyze_code_structure,
            "classify_severity_level": classify_severity_level,
            "retrieve_similar_patterns": retrieve_similar_patterns,
            "extract_refactoring_suggestions": extract_refactoring_suggestions,
        }

        log_agent_event(self.agent_name, "initialization", {
            "specialization": self.specialization,
            "model": DEFAULT_MODEL,
            "framework": "LangChain Deep Agent (Manual Tool Orchestration)",
            "tools_count": len(self.tools),
        })
        logger.info(f"CodeSmellDetector (Deep Agent) initialized: {self.agent_name}")


    async def detect_smells(
        self,
        code: str,
        smell_types: Optional[List[str]] = None,
        use_rag: bool = True,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[CodeSmellFinding]:
        """
        Detect code smells using Deep Agent framework.

        Deep Agent orchestrates tools for:
        - Code structure analysis
        - Severity classification
        - Pattern matching via RAG
        - Refactoring suggestions

        Uses manual tool orchestration pattern optimized for short-duration tasks.

        Args:
            code: Code to analyze
            smell_types: Specific smell types to look for
            use_rag: Use RAG context for enhanced detection
            context: Additional context (RAG examples, metrics)

        Returns:
            List of detected CodeSmellFinding objects
        """
        start_time = time()
        log_agent_event(self.agent_name, "detection_start", {
            "code_length": len(code),
            "use_rag": use_rag,
            "framework": "Deep Agent",
        })

        try:
            # Deep Agent: Tool Orchestration Phase
            # Step 1: Analyze code structure
            logger.info("Deep Agent: Analyzing code structure...")
            structure_analysis = await asyncio.to_thread(
                _analyze_code_structure_impl,
                code
            )
            self.tools_invoked_count += 1

            # Step 2: Retrieve similar patterns for context
            rag_context = ""
            if use_rag:
                logger.info("Deep Agent: Retrieving similar patterns...")
                similar_patterns = await asyncio.to_thread(
                    _retrieve_patterns_impl,
                    code
                )
                rag_context = similar_patterns
                self.tools_invoked_count += 1

            # Step 3: LLM-based smell detection with context
            smell_filter = f"Focus on: {', '.join(smell_types)}" if smell_types else "Analyze ALL smell types in the catalog below"

            llm_input = f"""You are a senior static-analysis reviewer (SonarQube / PMD / Fowler's catalog).
Analyze the code and detect EVERY applicable code smell from the catalog below.

SMELL CATALOG (use these EXACT names in the "type" field):
{build_prompt_catalog_block()}

{smell_filter}

Code Structure:
{structure_analysis}

{f'Similar Patterns Context:{chr(10)}{rag_context}' if rag_context else ''}

Code to Analyze:
```
{code[:15000]}
```

INSTRUCTIONS:
- Analyze the ENTIRE code above, not just the beginning. Consider every function, class, and module-level statement.
- Inspect the code against EACH catalog entry; do not limit yourself to a few obvious ones.
- Use the EXACT canonical name from the catalog for "type" (not paraphrased).
- Report every instance you find (multiple of the same type are allowed with different locations).
- Look for cross-cutting smells: Duplicate Code across functions, Feature Envy between classes, Data Clumps across signatures, Shotgun Surgery, etc.
- Include a specific, measurable justification (LOC, complexity number, nesting depth, param count, duplicate ratio, etc.) in "explanation".
- Severity: CRITICAL | HIGH | MEDIUM | LOW.
- Confidence: 0.0-1.0 based on how strong the metric/structural signal is.

Return ONLY a JSON array with objects of this shape:
[{{"type":"<canonical name>","location":"line <start>-<end>","severity":"HIGH","explanation":"<metric-backed reason>","refactoring":"<concrete fix>","confidence":0.82}}]

If no smells apply, return [].
"""

            # Log LLM request
            log_llm_request(
                self.agent_name,
                DEFAULT_MODEL,
                len(llm_input),
                llm_input[:100],
                temperature=0.1,
            )

            # Call LLM via Deep Agent pattern
            llm_start = time()
            response = await asyncio.to_thread(
                self.llm.invoke,
                llm_input
            )
            llm_latency = time() - llm_start

            response_text = response.content if hasattr(response, 'content') else str(response)

            # Log LLM response
            log_llm_response(
                self.agent_name,
                DEFAULT_MODEL,
                len(response_text),
                len(response_text.split()) // 4,
                llm_latency,
            )

            # Step 4: Tool invocation for identified smells
            # Extract and classify each smell
            findings_raw = await self._parse_response(response_text, code)

            findings = []
            for raw_finding in findings_raw:
                # Use Deep Agent tools to enhance classification
                if raw_finding.smell_type:
                    # Canonicalize smell label (belt-and-braces with parser)
                    raw_finding.smell_type = normalize_smell_type(raw_finding.smell_type) or raw_finding.smell_type
                    severity_detail = await asyncio.to_thread(
                        _classify_severity_impl,
                        raw_finding.smell_type,
                        code[max(0, 0):max(0, 500)]
                    )
                    self.tools_invoked_count += 1

                    suggestions = await asyncio.to_thread(
                        _extract_refactoring_impl,
                        code[max(0, 0):max(0, 500)],
                        raw_finding.smell_type
                    )
                    self.tools_invoked_count += 1

                    # Enhance finding with tool results
                    raw_finding.severity = parse_smell_severity(severity_detail.split(":")[0]) or raw_finding.severity
                    raw_finding.refactoring = suggestions

                findings.append(raw_finding)

            # Update statistics
            self.detections_count += 1
            processing_time = time() - start_time
            self.total_latency += processing_time

            if findings:
                avg_conf = sum(f.confidence for f in findings) / len(findings)
                self.average_confidence = (self.average_confidence + avg_conf) / 2

            log_detection_result(
                self.agent_name,
                code[:50],
                len(findings),
                sum(1 for f in findings if f.severity == SeverityLevel.CRITICAL),
                sum(1 for f in findings if f.severity == SeverityLevel.HIGH),
                sum(1 for f in findings if f.severity == SeverityLevel.MEDIUM),
                sum(1 for f in findings if f.severity == SeverityLevel.LOW),
                processing_time
            )

            logger.info(f"Deep Agent analysis complete: {len(findings)} findings, {self.tools_invoked_count} tools invoked")
            return findings
        except Exception as e:
            logger.error(f"Deep Agent detection error: {e}", exc_info=True)
            log_agent_event(self.agent_name, "detection_error", {"error": str(e)})
            return []


    async def _parse_response(
        self,
        response: str,
        original_code: str
    ) -> List[CodeSmellFinding]:
        """Parse LLM response into CodeSmellFinding objects.

        Args:
            response: LLM response text
            original_code: Original code analyzed

        Returns:
            List of CodeSmellFinding objects
        """
        findings = []
        try:
            # Try structured parsing
            result = self.response_parser.parse(response)

            if not result.code_smells and result.confidence == 0.0:
                raise ValueError(result.notes or result.summary)

            for smell in result.code_smells:
                normalized_explanation = smell.explanation.strip()
                if len(normalized_explanation.strip('. ')) < 10:
                    continue

                finding = CodeSmellFinding(
                    smell_type=smell.type,
                    location=smell.location,
                    severity=SeverityLevel(smell.severity.value),
                    explanation=normalized_explanation,
                    refactoring=smell.refactoring,
                    confidence=result.confidence,
                    agent_name=self.agent_name,
                    timestamp=datetime.now().isoformat(),
                )
                findings.append(finding)

            if not findings and any(
                smell.lower() in response.lower() for smell in CANONICAL_SMELLS
            ):
                raise ValueError("Structured parse produced no usable findings")
        except Exception as e:
            logger.warning(f"Parse error, using fallback: {e}")
            findings = self._build_fallback_findings(response, original_code)

        return findings

    def _emit_finding(
        self,
        findings: List[CodeSmellFinding],
        seen: set,
        smell_type: str,
        location: str,
        severity: SeverityLevel,
        explanation: str,
        refactoring: str,
        confidence: float,
        now: str
    ) -> None:
        """Helper to emit deduplicated findings."""
        canonical = normalize_smell_type(smell_type) or smell_type
        key = (canonical, location)
        if key in seen:
            return
        seen.add(key)
        findings.append(CodeSmellFinding(
            smell_type=canonical,
            location=location,
            severity=severity,
            explanation=explanation,
            refactoring=refactoring,
            confidence=confidence,
            agent_name=self.agent_name,
            timestamp=now,
        ))

    def _check_keyword_smells(
        self,
        findings: List[CodeSmellFinding],
        seen: set,
        response: str,
        line_location: str,
        now: str
    ) -> None:
        """Stage 1: Detect smells mentioned in LLM response."""
        response_lower = response.lower()
        for canonical in CANONICAL_SMELLS:
            if canonical.lower() in response_lower:
                self._emit_finding(
                    findings, seen, canonical, line_location,
                    SeverityLevel.MEDIUM,
                    f"LLM identified '{canonical}' in analysis output.",
                    "Review the LLM reasoning and refactor per catalog guidance.",
                    0.60, now
                )

    def _check_size_and_structure_smells(
        self,
        findings: List[CodeSmellFinding],
        seen: set,
        original_code: str,
        line_location: str,
        line_count: int,
        now: str
    ) -> None:
        """Check size-related and structural smells: Long Method, Deep Nesting, Duplicate Code, Complexity."""
        # Long Method
        if line_count >= 50:
            severity = SeverityLevel.CRITICAL if line_count > 100 else SeverityLevel.HIGH if line_count > 70 else SeverityLevel.MEDIUM
            self._emit_finding(
                findings, seen, "Long Method", line_location, severity,
                f"Method spans {line_count} non-empty lines - exceeds recommended size.",
                "Extract into smaller helper functions with single responsibilities.",
                0.70 if line_count > 100 else 0.58, now
            )

        # Deep Nesting
        max_nesting = _compute_max_nesting_depth(original_code)
        if max_nesting > 4:
            self._emit_finding(
                findings, seen, "Deep Nesting", line_location,
                SeverityLevel.HIGH if max_nesting > 6 else SeverityLevel.MEDIUM,
                f"Maximum nesting depth is {max_nesting} levels - hurts readability/testability.",
                "Use early returns, guard clauses, or extract nested logic to helpers.",
                0.72, now
            )

        # Duplicate Code
        duplicates = _find_duplicate_blocks(original_code)
        if duplicates:
            avg_sim = sum(sim for _, sim in duplicates) / len(duplicates)
            self._emit_finding(
                findings, seen, "Duplicate Code", line_location, SeverityLevel.HIGH,
                f"Found {len(duplicates)} duplicate blocks with {avg_sim*100:.0f}% avg similarity.",
                "Extract shared logic into utility functions or base classes.",
                0.72 if avg_sim > 0.90 else 0.62, now
            )

        # High Cyclomatic Complexity
        complexity = _compute_cyclomatic_complexity(original_code)
        if complexity > 10:
            self._emit_finding(
                findings, seen, "High Cyclomatic Complexity", line_location,
                SeverityLevel.HIGH if complexity > 15 else SeverityLevel.MEDIUM,
                f"Cyclomatic complexity ~{complexity} - too many decision paths.",
                "Decompose branches, use polymorphism, or lookup tables.",
                0.68, now
            )

    def _check_parameter_and_data_smells(
        self,
        findings: List[CodeSmellFinding],
        seen: set,
        original_code: str,
        line_location: str,
        now: str
    ) -> None:
        """Check parameter/data smells: Long Params, Data Clumps, Magic Numbers, Primitives."""
        # Long Parameter List + Data Clumps
        long_params = _detect_long_parameter_list(original_code, threshold=5)
        if long_params:
            for name, count in long_params[:5]:
                self._emit_finding(
                    findings, seen, "Long Parameter List", line_location,
                    SeverityLevel.MEDIUM if count < 7 else SeverityLevel.HIGH,
                    f"Function '{name}' has {count} parameters.",
                    "Introduce a parameter object / dataclass.", 0.66, now
                )
            if len(long_params) >= 2:
                self._emit_finding(
                    findings, seen, "Data Clumps", line_location, SeverityLevel.MEDIUM,
                    f"{len(long_params)} functions share large parameter lists - likely repeated data groups.",
                    "Group related parameters into a dedicated type/dataclass.", 0.60, now
                )

        # Magic Numbers
        magic_nums = _count_magic_numbers(original_code)
        if len(magic_nums) >= 3:
            self._emit_finding(
                findings, seen, "Magic Numbers", line_location, SeverityLevel.LOW,
                f"Found {len(magic_nums)} hard-coded numeric/string literals without semantic meaning.",
                "Replace with named constants (e.g., MAX_RETRIES, DEFAULT_TIMEOUT).", 0.58, now
            )

        # Primitive Obsession
        primitives = _detect_primitive_obsession(original_code)
        if primitives >= 10:
            self._emit_finding(
                findings, seen, "Primitive Obsession", line_location, SeverityLevel.LOW,
                f"High use of primitive types ({primitives} primitive parameters/fields).",
                "Introduce value objects / dataclasses for repeated primitive groups.", 0.50, now
            )

    def _check_code_quality_smells(
        self,
        findings: List[CodeSmellFinding],
        seen: set,
        original_code: str,
        line_location: str,
        now: str
    ) -> None:
        """Check code quality smells: Dead Code, Naming, Error Handling, Empty Catch, Comments."""
        # Dead Code (unused variables)
        unused = _find_unused_variables(original_code)
        if unused:
            self._emit_finding(
                findings, seen, "Dead Code", line_location, SeverityLevel.MEDIUM,
                f"Found {len(unused)} potentially unused identifiers: {', '.join(unused[:3])}{'...' if len(unused) > 3 else ''}",
                "Remove unused variables/imports and unreachable branches.", 0.55, now
            )

        # Inconsistent Naming
        naming = _analyze_naming_consistency(original_code)
        if naming["inconsistency_score"] > 0.6 and naming["single_letter"] > 2:
            self._emit_finding(
                findings, seen, "Inconsistent Naming", line_location, SeverityLevel.LOW,
                f"Mixed conventions (snake_case={naming['snake_case']}, camelCase={naming['camel_case']}, single-letter={naming['single_letter']}).",
                "Standardize on one convention; replace single-letter vars with descriptive names.",
                0.52, now
            )

        # Missing Error Handling
        risky_ops = _detect_risky_operations(original_code)
        if risky_ops:
            self._emit_finding(
                findings, seen, "Missing Error Handling", line_location, SeverityLevel.HIGH,
                f"Risky operations without error handling: {', '.join(risky_ops)}.",
                "Wrap risky I/O/network/parsing calls in try/except with recovery.", 0.70, now
            )

        # Empty Catch Block
        empty_catches = _detect_empty_catch(original_code)
        if empty_catches:
            self._emit_finding(
                findings, seen, "Empty Catch Block", line_location, SeverityLevel.HIGH,
                f"Detected {empty_catches} empty exception handlers - errors are silently swallowed.",
                "Log the exception and/or re-raise; never pass silently.", 0.75, now
            )

        # Commented-out Code
        commented = _detect_commented_out_code(original_code)
        if commented >= 3:
            self._emit_finding(
                findings, seen, "Comments", line_location, SeverityLevel.LOW,
                f"Detected {commented} lines of commented-out code.",
                "Delete dead commented code - rely on version control for history.", 0.60, now
            )

    def _check_design_smells(
        self,
        findings: List[CodeSmellFinding],
        seen: set,
        original_code: str,
        line_location: str,
        line_count: int,
        now: str
    ) -> None:
        """Check design smells: Switches, Message Chains, Middle Man, Classes (God/Large/Data/Lazy), Feature Envy."""
        # Switch Statements
        switches = _detect_switch_statements(original_code, min_branches=4)
        if switches:
            self._emit_finding(
                findings, seen, "Switch Statements", line_location, SeverityLevel.MEDIUM,
                f"Large switch/elif chain with {max(switches)} branches detected.",
                "Replace type-code switch with polymorphism or strategy map.", 0.60, now
            )

        # Message Chains
        chains = _detect_message_chains(original_code, min_chain=4)
        if chains:
            self._emit_finding(
                findings, seen, "Message Chains", line_location, SeverityLevel.MEDIUM,
                f"Detected {chains} long dotted call chains (a.b().c().d()...).",
                "Apply 'Hide Delegate' or expose a direct method on the intermediate object.",
                0.58, now
            )

        # Middle Man
        middle_man = _detect_middle_man(original_code)
        if middle_man >= 3:
            self._emit_finding(
                findings, seen, "Middle Man", line_location, SeverityLevel.LOW,
                f"Detected {middle_man} methods that only delegate to another object.",
                "Remove the middle man or inline the delegation.", 0.55, now
            )

        # Data Class
        data_classes = _detect_data_class(original_code)
        for cls_name in data_classes[:5]:
            self._emit_finding(
                findings, seen, "Data Class", line_location, SeverityLevel.LOW,
                f"Class '{cls_name}' exposes only fields/getters/setters without behavior.",
                "Move behavior that uses this data into the class itself.", 0.55, now
            )

        # Lazy Class
        lazy = _detect_lazy_class(original_code)
        for cls_name in lazy[:5]:
            self._emit_finding(
                findings, seen, "Lazy Class", line_location, SeverityLevel.LOW,
                f"Class '{cls_name}' has a trivial body - may not earn its keep.",
                "Inline into callers or merge with a related class.", 0.50, now
            )

        # Feature Envy
        envy = _detect_feature_envy(original_code, threshold=4)
        if envy:
            self._emit_finding(
                findings, seen, "Feature Envy", line_location, SeverityLevel.MEDIUM,
                f"{envy} method(s) access another object's data more than their own.",
                "Move the method to the class it envies (Move Method refactoring).", 0.58, now
            )

        # God Class / Large Class
        method_count = len(re.findall(r"def\s+\w+\s*\(", original_code))
        if method_count > 15 and line_count > 300:
            self._emit_finding(
                findings, seen, "God Class", line_location, SeverityLevel.HIGH,
                f"Class/module has {method_count} methods and {line_count} LOC - violates SRP.",
                "Decompose into multiple focused classes along responsibility lines.", 0.66, now
            )
        elif line_count > 200 and method_count > 10:
            self._emit_finding(
                findings, seen, "Large Class", line_location, SeverityLevel.MEDIUM,
                f"Class/module has {method_count} methods and {line_count} LOC - getting unwieldy.",
                "Extract cohesive groups of methods into collaborator classes.", 0.58, now
            )

    def _check_extended_smells(
        self,
        findings: List[CodeSmellFinding],
        seen: set,
        original_code: str,
        line_location: str,
        now: str
    ) -> None:
        """Check extended smell types: Poor Naming, Inconsistent Style, Inappropriate Coupling, Flag Arguments, Generic/Hidden Exceptions."""

        # Poor Naming
        poor_naming = _detect_poor_naming(original_code)
        if poor_naming >= 3:
            self._emit_finding(
                findings, seen, "Poor Naming", line_location, SeverityLevel.LOW,
                f"Detected {poor_naming} instances of unclear naming (single letters, abbreviations).",
                "Use descriptive names that reveal intent (variable, function, and class names).", 0.55, now
            )

        # Inconsistent Style
        style_stats = _detect_inconsistent_style(original_code)
        conventions_used = sum(1 for count in [style_stats["snake_case"], style_stats["camelCase"]] if count > 0)
        indent_types = sum(1 for count in [style_stats["spaces_for_indent"], style_stats["tabs_for_indent"]] if count > 0)

        if conventions_used > 1:
            self._emit_finding(
                findings, seen, "Inconsistent Style", line_location, SeverityLevel.LOW,
                f"Mixed naming conventions: snake_case ({style_stats['snake_case']}), camelCase ({style_stats['camelCase']}).",
                "Enforce one consistent naming convention across the codebase (PEP 8 for Python, etc).", 0.52, now
            )

        if indent_types > 1:
            self._emit_finding(
                findings, seen, "Inconsistent Style", line_location, SeverityLevel.LOW,
                f"Mixed indentation: spaces ({style_stats['spaces_for_indent']}), tabs ({style_stats['tabs_for_indent']}).",
                "Use consistent indentation (4 spaces recommended for Python).", 0.60, now
            )

        # Inappropriate Coupling
        coupling = _detect_inappropriate_coupling(original_code)
        if coupling:
            self._emit_finding(
                findings, seen, "Inappropriate Coupling", line_location, SeverityLevel.MEDIUM,
                f"{', '.join(coupling)} - indicates unrelated modules are tightly linked.",
                "Reduce cross-domain dependencies; use dependency injection or service locators.", 0.58, now
            )

        # Flag Arguments
        flag_args = _detect_flag_arguments(original_code)
        if flag_args:
            self._emit_finding(
                findings, seen, "Flag Arguments", line_location, SeverityLevel.MEDIUM,
                f"Detected {len(flag_args)} function(s) with flag arguments: {', '.join(flag_args[:2])}",
                "Split into separate methods or use polymorphism instead of flag parameters.", 0.62, now
            )

        # Generic Exceptions
        generic_exc = _detect_generic_exceptions(original_code)
        if generic_exc >= 1:
            self._emit_finding(
                findings, seen, "Generic Exceptions", line_location, SeverityLevel.HIGH,
                f"Found {generic_exc} broad exception handlers - catching Exception, Throwable, or bare except.",
                "Catch specific exception types; add proper logging and recovery strategies.", 0.70, now
            )

        # Hidden Errors
        hidden_errs = _detect_hidden_errors(original_code)
        if hidden_errs >= 1:
            self._emit_finding(
                findings, seen, "Hidden Errors", line_location, SeverityLevel.HIGH,
                f"Found {hidden_errs} cases of silently swallowed errors (pass, early returns).",
                "Log errors, re-raise when appropriate, or implement proper recovery logic.", 0.68, now
            )

    def _build_fallback_findings(
        self,
        response: str,
        original_code: str,
    ) -> List[CodeSmellFinding]:
        """Build comprehensive heuristic findings when structured parsing fails.

        Delegates to stage-specific helper methods for better organization.
        Uses metric-based detection aligned with the canonical SonarQube + extended catalog.
        """
        findings: List[CodeSmellFinding] = []
        seen: set = set()
        non_empty_lines = [line for line in original_code.splitlines() if line.strip()]
        line_count = len(non_empty_lines)
        line_location = f"line 1-{line_count}" if line_count > 1 else "line 1"
        now = datetime.now().isoformat()

        # Stage 1: Keyword-based detection
        self._check_keyword_smells(findings, seen, response, line_location, now)

        # Stage 2: Metric-based checks organized by category
        self._check_size_and_structure_smells(findings, seen, original_code, line_location, line_count, now)
        self._check_parameter_and_data_smells(findings, seen, original_code, line_location, now)
        self._check_code_quality_smells(findings, seen, original_code, line_location, now)
        self._check_design_smells(findings, seen, original_code, line_location, line_count, now)
        self._check_extended_smells(findings, seen, original_code, line_location, now)

        return findings


    def get_stats(self) -> Dict[str, Any]:
        """Get Deep Agent detector statistics.

        Returns:
            Dictionary with statistics including tool invocation counts
        """
        return {
            "agent_name": self.agent_name,
            "specialization": self.specialization,
            "framework": "LangChain Deep Agent",
            "detections_count": self.detections_count,
            "false_positives": self.false_positives,
            "average_confidence": self.average_confidence,
            "average_latency": self.total_latency / max(1, self.detections_count),
            "tools_invoked_count": self.tools_invoked_count,
            "tools_available": [
                "analyze_code_structure",
                "classify_severity_level",
                "retrieve_similar_patterns",
                "extract_refactoring_suggestions"
            ],
        }

# ============================================================================
# Test Function
# ============================================================================

async def test_code_smell_detector():
    """Test the Deep Agent-integrated Code Smell Detector."""
    print("Testing LangChain Deep Agent Code Smell Detector...")

    detector = CodeSmellDetector(specialization="Long Method expert")

    test_code = """
def process_user_data(user, config, cache):
    result = []
    for item in user.items:
        if item.valid:
            processed = item.process()
            result.append(processed)
            if len(result) > 1000:
                result = []
    return result
"""

    print(f"✓ Detector initialized: {detector.agent_name}")
    stats = detector.get_stats()
    print(f"✓ Tools available: {len(stats['tools_available'])}")
    print(f"✓ Framework: {stats['framework']}")

    # Test detection with Deep Agent
    findings = await detector.detect_smells(test_code)
    stats = detector.get_stats()

    print(f"✓ Deep Agent analysis complete")
    print(f"✓ Detected {len(findings)} smells")
    if findings:
        print(f"✓ Average confidence: {stats['average_confidence']:.2f}")
        print(f"✓ Tool invocations used: {stats['tools_invoked_count']}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_code_smell_detector())

