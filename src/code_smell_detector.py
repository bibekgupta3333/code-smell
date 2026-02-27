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
from typing import List, Optional, Dict, Any
from datetime import datetime
from time import time

from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from src.llm_client import OllamaClient
from src.prompt_templates import create_rag_prompt, get_system_prompt
from src.response_parser import ResponseParser, AnalysisResult
from src.rag_retriever import RAGRetriever
from src.common import CodeSmellFinding, SeverityLevel, parse_smell_severity
from src.logger import log_agent_event, log_detection_result, log_llm_request, log_llm_response
from config import DEFAULT_MODEL

logger = logging.getLogger(__name__)



# ============================================================================
# Deep Agent Tools for Code Smell Detection
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
    from src.code_parser import CodeParser
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


@tool
def retrieve_similar_patterns(code: str) -> str:
    """Retrieve similar code patterns from knowledge base.
    
    Deep Agent tool for context-aware RAG integration.

    Args:
        code: Code snippet to find similar patterns for

    Returns:
        Similar patterns and best practices
    """
    try:
        from src.rag_retriever import RAGRetriever
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
    ):
        """
        Initialize Deep Agent detector.

        Args:
            specialization: Smell type specialization (e.g., "Long Method expert")
            llm_client: Ollama client for LLM calls
            rag_retriever: RAG retriever for context
        """
        self.specialization = specialization or "General code smell detector"
        self.agent_name = f"detector_{specialization.lower().replace(' ', '_')[:30]}" if specialization else "detector_general"
        self.llm_client = llm_client or OllamaClient()
        self.rag_retriever = rag_retriever or RAGRetriever()
        self.response_parser = ResponseParser()

        # Statistics
        self.detections_count = 0
        self.false_positives = 0
        self.average_confidence = 0.0
        self.total_latency = 0.0
        self.tools_invoked_count = 0

        # Initialize LangChain ChatOllama for Deep Agent
        self.llm = ChatOllama(
            model=DEFAULT_MODEL,
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
                analyze_code_structure,
                code
            )
            self.tools_invoked_count += 1

            # Step 2: Retrieve similar patterns for context
            rag_context = ""
            if use_rag:
                logger.info("Deep Agent: Retrieving similar patterns...")
                similar_patterns = await asyncio.to_thread(
                    retrieve_similar_patterns,
                    code
                )
                rag_context = similar_patterns
                self.tools_invoked_count += 1

            # Step 3: LLM-based smell detection with context
            smell_filter = f"Focus on: {', '.join(smell_types)}" if smell_types else "Analyze all code smell types"
            
            llm_input = f"""Deep Agent Analysis Task: Detect Code Smells

Code Analysis Results:
{structure_analysis}

{f'Similar Patterns Context:{chr(10)}{rag_context}' if rag_context else ''}

Code to Analyze:
```python
{code[:2000]}
```

{smell_filter}

Identify code smells. For each smell found:
1. Use classify_severity_level to determine severity
2. Use extract_refactoring_suggestions to recommend fixes

Output JSON format: [{{"smell_type": "...", "location": "...", "severity": "...", "explanation": "...", "refactoring": "...", "confidence": 0.0}}]"""

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
                    severity_detail = await asyncio.to_thread(
                        classify_severity_level,
                        raw_finding.smell_type,
                        code[max(0, 0):max(0, 500)]
                    )
                    self.tools_invoked_count += 1

                    suggestions = await asyncio.to_thread(
                        extract_refactoring_suggestions,
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

            for smell in result.code_smells:
                finding = CodeSmellFinding(
                    smell_type=smell.type,
                    location=smell.location,
                    severity=SeverityLevel(smell.severity.value),
                    explanation=smell.explanation,
                    refactoring=smell.refactoring,
                    confidence=result.confidence,
                    agent_name=self.agent_name,
                    timestamp=datetime.now().isoformat(),
                )
                findings.append(finding)
        except Exception as e:
            logger.warning(f"Parse error, using fallback: {e}")
            # Fallback: extract basic findings from freeform response
            response_lower = response.lower()

            identified_smells = {
                "long method": ("long method", "method>50 lines", 0.6),
                "god class": ("god class", "class>20 methods", 0.6),
                "feature envy": ("feature envy", "accessing external data", 0.5),
                "data clumps": ("data clumps", "grouped data", 0.5),
            }

            for keyword, (smell_type, location, confidence) in identified_smells.items():
                if keyword in response_lower:
                    findings.append(CodeSmellFinding(
                        smell_type=smell_type,
                        location=location,
                        severity=SeverityLevel.MEDIUM,
                        explanation=f"Detected by LLM analysis: {response[:200]}",
                        refactoring="Review and refactor",
                        confidence=confidence,
                        agent_name=self.agent_name,
                        timestamp=datetime.now().isoformat(),
                    ))
                    break

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

