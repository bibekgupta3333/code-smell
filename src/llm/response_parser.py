"""
Response Parser for LLM Code Smell Detection
Parses and validates LLM JSON responses with error handling and retry logic.

Architecture: Handles response validation as per Architecture Section 10.1
"""

import json
import logging
import re
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class Severity(str, Enum):
    """Severity levels for code smells."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class CodeSmell:
    """Represents a detected code smell."""
    type: str
    location: str
    severity: Severity
    explanation: str
    refactoring: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.type,
            "location": self.location,
            "severity": self.severity.value,
            "explanation": self.explanation,
            "refactoring": self.refactoring,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CodeSmell":
        """Create from dictionary."""
        return cls(
            type=data.get("type", "Unknown"),
            location=data.get("location", ""),
            severity=Severity(data.get("severity", "LOW")),
            explanation=data.get("explanation", ""),
            refactoring=data.get("refactoring"),
        )


@dataclass
class AnalysisResult:
    """Result of code smell analysis."""
    code_smells: List[CodeSmell]
    summary: str
    is_valid_code: bool
    notes: Optional[str] = None
    confidence: float = 1.0
    raw_response: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "code_smells": [smell.to_dict() for smell in self.code_smells],
            "summary": self.summary,
            "is_valid_code": self.is_valid_code,
            "notes": self.notes,
            "confidence": self.confidence,
            "smell_count": len(self.code_smells),
        }


class ResponseParser:
    """
    Parse and validate LLM responses for code smell detection.

    Features:
    - JSON extraction from LLM output
    - Malformed response repair
    - Validation against schema
    - Confidence scoring
    - Detailed error reporting

    Example:
        >>> parser = ResponseParser()
        >>> result = parser.parse('{"code_smells": [...]}')
        >>> print(result.code_smells)
    """

    # Valid smell types
    VALID_SMELL_TYPES = {
        "Long Method",
        "God Class",
        "Large Class",
        "Long Parameter List",
        "Primitive Obsession",
        "Feature Envy",
        "Inappropriate Intimacy",
        "Message Chains",
        "Divergent Change",
        "Shotgun Surgery",
        "Duplicate Code",
        "Lazy Class",
        "Data Class",
        "Dead Code",
        "Switch Statements",
        "Refused Bequest",
        "Parallel Inheritance Hierarchies",
    }

    def __init__(self, strict_mode: bool = False):
        """
        Initialize parser.

        Args:
            strict_mode: If True, reject malformed responses
        """
        self.strict_mode = strict_mode
        self.parse_attempts = 0

    def parse(
        self,
        response: str,
        allow_repair: bool = True,
    ) -> AnalysisResult:
        """
        Parse LLM response.

        Args:
            response: Raw LLM response
            allow_repair: Attempt to repair malformed JSON

        Returns:
            Parsed analysis result
        """
        self.parse_attempts += 1

        # Try to extract JSON
        json_obj = self._extract_json(response)

        if json_obj is None:
            if allow_repair and not self.strict_mode:
                logger.warning("JSON extraction failed, attempting repair")
                json_obj = self._repair_json(response)

            if json_obj is None:
                logger.error("Failed to parse response as JSON")
                return self._create_error_result(response)

        # Validate and extract fields
        try:
            return self._validate_and_extract(json_obj, response)
        except ValueError as e:
            logger.error(f"Validation error: {e}")
            return self._create_error_result(response)

    def _extract_json(self, response: str) -> Optional[Dict[str, Any]]:
        """
        Extract JSON object from response.

        Args:
            response: Raw response text

        Returns:
            Parsed JSON object or None
        """
        # Try direct JSON parsing first
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Try to find JSON in text (markdown code blocks, etc)
        json_patterns = [
            r'```(?:json)?\s*([\s\S]*?)```',  # Markdown code blocks
            r'({[\s\S]*})',  # JSON object
        ]

        for pattern in json_patterns:
            matches = re.findall(pattern, response)
            for match in matches:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue

        return None

    def _repair_json(self, response: str) -> Optional[Dict[str, Any]]:
        """
        Attempt to repair malformed JSON.

        Args:
            response: Malformed response

        Returns:
            Repaired JSON or None
        """
        try:
            # Remove common issues
            cleaned = response.strip()

            # Remove markdown formatting
            cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
            cleaned = re.sub(r'\s*```$', '', cleaned)

            # Try single quotes to double quotes
            # (be careful with apostrophes in text)
            if "'" in cleaned and '"' not in cleaned:
                cleaned = cleaned.replace("'", '"')

            # Try to find and extract just the JSON part
            start = cleaned.find('{')
            end = cleaned.rfind('}')

            if start != -1 and end != -1 and end > start:
                cleaned = cleaned[start:end+1]
                return json.loads(cleaned)

        except Exception as e:
            logger.debug(f"JSON repair failed: {e}")

        return None

    def _validate_and_extract(
        self,
        json_obj: Dict[str, Any],
        raw_response: str,
    ) -> AnalysisResult:
        """
        Validate JSON structure and extract data.

        Args:
            json_obj: Parsed JSON object
            raw_response: Original response text

        Returns:
            Validated AnalysisResult

        Raises:
            ValueError: If validation fails (in strict mode)
        """
        # Extract code smells
        code_smells_data = json_obj.get("code_smells", [])
        if not isinstance(code_smells_data, list):
            msg = f"code_smells must be a list, got {type(code_smells_data)}"
            if self.strict_mode:
                raise ValueError(msg)
            logger.warning(msg)
            code_smells_data = []

        code_smells = []
        for smell_data in code_smells_data:
            try:
                smell = self._parse_code_smell(smell_data)
                code_smells.append(smell)
            except ValueError as e:
                logger.warning(f"Skipping invalid smell: {e}")
                continue

        # Extract other fields
        summary = json_obj.get("summary", f"Found {len(code_smells)} code smell(s)")
        is_valid_code = json_obj.get("is_valid_code", True)
        notes = json_obj.get("notes")

        # Calculate confidence
        confidence = self._calculate_confidence(json_obj, code_smells)

        return AnalysisResult(
            code_smells=code_smells,
            summary=summary,
            is_valid_code=is_valid_code,
            notes=notes,
            confidence=confidence,
            raw_response=raw_response,
        )

    def _parse_code_smell(self, data: Dict[str, Any]) -> CodeSmell:
        """
        Parse individual code smell entry.

        Args:
            data: Code smell data

        Returns:
            CodeSmell object

        Raises:
            ValueError: If required fields are missing
        """
        if not data:
            raise ValueError("Empty code smell data")

        # Validate required fields
        smell_type = data.get("type", "").strip()
        if not smell_type:
            raise ValueError("Missing or empty 'type' field")

        # Try to match to valid type (case-insensitive)
        matched_type = None
        for valid_type in self.VALID_SMELL_TYPES:
            if valid_type.lower() == smell_type.lower():
                matched_type = valid_type
                break

        if matched_type is None and self.strict_mode:
            raise ValueError(f"Unknown code smell type: {smell_type}")

        matched_type = matched_type or smell_type

        location = data.get("location", "Unknown").strip()
        if not location:
            raise ValueError("Missing or empty 'location' field")

        explanation = data.get("explanation", "").strip()
        if not explanation:
            raise ValueError("Missing or empty 'explanation' field")

        # Parse severity
        severity_str = data.get("severity", "LOW").upper().strip()
        try:
            severity = Severity(severity_str)
        except ValueError:
            logger.warning(f"Invalid severity '{severity_str}', using LOW")
            severity = Severity.LOW

        refactoring = data.get("refactoring")
        if refactoring:
            refactoring = refactoring.strip()

        return CodeSmell(
            type=matched_type,
            location=location,
            severity=severity,
            explanation=explanation,
            refactoring=refactoring,
        )

    def _calculate_confidence(
        self,
        json_obj: Dict[str, Any],
        code_smells: List[CodeSmell],
    ) -> float:
        """
        Calculate confidence score for the analysis.

        Args:
            json_obj: Parsed JSON
            code_smells: Extracted code smells

        Returns:
            Confidence score (0.0 to 1.0)
        """
        confidence = 1.0

        # Reduce confidence if fields are missing
        required_fields = {"code_smells", "summary", "is_valid_code"}
        missing_fields = required_fields - set(json_obj.keys())
        if missing_fields:
            confidence *= 0.9

        # Reduce confidence if some smells were invalid
        smells_in_json = len(json_obj.get("code_smells", []))
        if smells_in_json > 0 and len(code_smells) < smells_in_json:
            confidence *= (len(code_smells) / smells_in_json)

        # Reduce confidence if many smells are unknown types
        unknown_count = 0
        for smell_data in json_obj.get("code_smells", []):
            if isinstance(smell_data, dict):
                smell_type = smell_data.get("type", "").strip()
                if smell_type and smell_type not in self.VALID_SMELL_TYPES:
                    unknown_count += 1

        if unknown_count > 0 and len(code_smells) > 0:
            confidence *= (1 - (unknown_count / len(code_smells)) * 0.2)

        return max(0.0, min(1.0, confidence))

    def _create_error_result(self, response: str) -> AnalysisResult:
        """
        Create error result when parsing fails.

        Args:
            response: Original response

        Returns:
            AnalysisResult with error information
        """
        return AnalysisResult(
            code_smells=[],
            summary="Failed to parse LLM response",
            is_valid_code=False,
            notes=f"Response parse error: {response[:100]}...",
            confidence=0.0,
            raw_response=response,
        )


async def test_parser():
    """Test the response parser."""
    parser = ResponseParser()

    # Test valid response
    valid_response = json.dumps({
        "code_smells": [
            {
                "type": "Long Method",
                "location": "process_user, lines 1-50",
                "severity": "HIGH",
                "explanation": "Method is over 50 lines",
                "refactoring": "Extract helper methods",
            }
        ],
        "summary": "Found 1 code smell",
        "is_valid_code": True,
    })

    result = parser.parse(valid_response)
    print(f"✓ Valid response parsed: {len(result.code_smells)} smell(s)")
    print(f"  Confidence: {result.confidence:.2f}")

    # Test malformed response
    malformed_response = """
    ```json
    {
        "code_smells": [{"type": "God Class"}],
        "summary": "Found 1 smell"
    }
    ```
    """

    result = parser.parse(malformed_response)
    print(f"✓ Malformed response repaired: {len(result.code_smells)} smell(s)")


if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_parser())
