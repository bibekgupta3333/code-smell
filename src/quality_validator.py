"""
Quality Validator Agent (CodeReviewer)
Reviews and validates detection results for false positives and quality.

Architecture: CodeReviewer agent in multi-agent system
Validates findings and assigns confidence/quality scores
"""

import logging
from typing import List, Dict, Any
from datetime import datetime

from src.common import CodeSmellFinding, SeverityLevel
from src.logger import log_agent_event

logger = logging.getLogger(__name__)


class QualityValidator:
    """
    Quality Validator Agent (CodeReviewer Role).

    Responsible for:
    - Validating detection results
    - Assigning confidence scores (0-3 scale)
    - Filtering false positives
    - Cross-checking with domain knowledge
    - Suggesting improvements

    Example:
        >>> validator = QualityValidator()
        >>> validated = validator.validate_findings(findings, code)
    """

    def __init__(self):
        """Initialize quality validator."""
        self.agent_name = "validator_code_reviewer"

        # Validation statistics
        self.validations_performed = 0
        self.false_positives_filtered = 0
        self.average_confidence = 0.0

        # Domain knowledge rules for validation
        self.validation_rules = {
            "Long Method": {
                "min_lines": 30,
                "confidence_boost": 0.1,
                "false_positive_patterns": ["decorators", "generated"],
            },
            "God Class": {
                "min_methods": 10,
                "confidence_boost": 0.15,
                "false_positive_patterns": ["interface", "abstract"],
            },
            "Feature Envy": {
                "min_references": 5,
                "confidence_boost": 0.05,
                "false_positive_patterns": ["domain model", "utility"],
            },
        }

        logger.info("Quality Validator initialized: %s", self.agent_name)  # noqa: G201

    def validate_findings(self, findings: List[CodeSmellFinding]) -> List[CodeSmellFinding]:
        """
        Validate detection findings.

        Args:
            findings: List of findings to validate

        Returns:
            List of validated findings
        """
        validated = []
        log_agent_event(
            self.agent_name,
            "validation_start",
            {"findings_count": len(findings)},
        )

        for finding in findings:
            is_valid, confidence_adjustment = self._validate_finding(finding)

            if is_valid:
                # Adjust confidence based on validation
                adjusted_confidence = min(
                    1.0,
                    finding.confidence + confidence_adjustment,
                )

                validated_finding = CodeSmellFinding(
                    smell_type=finding.smell_type,
                    location=finding.location,
                    severity=finding.severity,
                    explanation=finding.explanation,
                    refactoring=finding.refactoring,
                    confidence=adjusted_confidence,
                    agent_name=f"{finding.agent_name}→{self.agent_name}",
                    timestamp=datetime.now().isoformat(),
                )
                validated.append(validated_finding)
            else:
                self.false_positives_filtered += 1
                logger.warning(
                    "Filtered likely false positive: %s at %s",  # noqa: G201
                    finding.smell_type,
                    finding.location
                )

        self.validations_performed += 1
        if validated:
            self.average_confidence = sum(f.confidence for f in validated) / len(validated)

        log_agent_event(
            self.agent_name,
            "validation_complete",
            {
                "valid_findings": len(validated),
                "false_positives": len(findings) - len(validated),
                "average_confidence": self.average_confidence,
            },
        )

        return validated

    def _validate_finding(
        self,
        finding: CodeSmellFinding,
    ) -> tuple:
        """
        Validate a single finding.

        Args:
            finding: Finding to validate

        Returns:
            (is_valid, confidence_adjustment)
        """
        # Basic validation: check if smell type is known
        known_types = set(self.validation_rules.keys()) | {
            "Data Clumps",
            "Shotgun Surgery",
            "Parallel Inheritance",
            "Lazy Class",
            "Speculative Generality",
            "Temporary Field",
            "Message Chains",
            "Middle Man",
            "Alternative Classes",
            "Data Classes",
            "Comments",
            "Duplicate Code",
            "Switch Statements",
            "Primitive Obsession",
        }

        if finding.smell_type not in known_types:
            logger.warning("Unknown smell type: %s", finding.smell_type)  # noqa: G201
            return False, 0.0

        # Apply smell-specific validation rules
        confidence_adjustment = 0.0

        if finding.smell_type in self.validation_rules:
            rules = self.validation_rules[finding.smell_type]

            # Check false positive patterns
            for pattern in rules.get("false_positive_patterns", []):
                if pattern.lower() in finding.explanation.lower():
                    logger.info(
                        "Detected false positive pattern '%s' in %s",  # noqa: G201
                        pattern,
                        finding.smell_type
                    )
                    return False, 0.0

            # Apply confidence boost for known types
            confidence_adjustment = rules.get("confidence_boost", 0.0)

        # Validation checks
        if not finding.location:
            return False, 0.0

        if not finding.explanation or len(finding.explanation) < 10:
            return False, 0.0

        if finding.confidence < 0.3:
            logger.info("Confidence too low for %s: %f", finding.smell_type, finding.confidence)  # noqa: G201
            return False, 0.0

        # Check severity alignment
        if finding.severity == SeverityLevel.CRITICAL and finding.confidence < 0.7:
            logger.warning(
                "Critical severity without high confidence: %s (%.2f)",  # noqa: G201
                finding.smell_type,
                finding.confidence
            )
            return True, -0.1  # Reduce confidence

        return True, confidence_adjustment

    def suggest_improvements(
        self,
        findings: List[CodeSmellFinding],
    ) -> Dict[str, Any]:
        """
        Suggest improvements for the set of findings.

        Args:
            findings: List of findings

        Returns:
            Improvement suggestions
        """
        suggestions = {
            "overall": [],
            "per_smell": {},
        }

        # Overall suggestions
        if not findings:
            suggestions["overall"].append("No code smells detected - good code quality!")

        elif len(findings) > 10:
            suggestions["overall"].append(
                f"High number of smells ({len(findings)}). "
                "Consider refactoring in phases."
            )

        critical_count = sum(
            1 for f in findings if f.severity == SeverityLevel.CRITICAL
        )
        if critical_count > 0:
            suggestions["overall"].append(
                f"Address {critical_count} critical issue(s) first."
            )

        # Per-smell suggestions
        smell_types = {}
        for finding in findings:
            if finding.smell_type not in smell_types:
                smell_types[finding.smell_type] = []
            smell_types[finding.smell_type].append(finding)

        for smell_type, type_findings in smell_types.items():
            count = len(type_findings)

            if smell_type == "Long Method" and count > 0:
                suggestions["per_smell"][smell_type] = (
                    "Extract smaller methods. Focus on single responsibility."
                )

            elif smell_type == "God Class" and count > 0:
                suggestions["per_smell"][smell_type] = (
                    "Split into multiple focused classes."
                )

            elif smell_type == "Feature Envy" and count > 0:
                suggestions["per_smell"][smell_type] = (
                    "Move methods closer to data they access."
                )

            elif count > 1:
                suggestions["per_smell"][smell_type] = (
                    f"Multiple instances of {smell_type} detected. "
                    "Establish refactoring patterns."
                )

        return suggestions

    def assign_confidence_score(
        self,
        finding: CodeSmellFinding,
    ) -> float:
        """
        Assign confidence score (0.0-1.0).

        Args:
            finding: Finding to score

        Returns:
            Confidence score
        """
        score = finding.confidence

        # Adjust based on severity
        if finding.severity == SeverityLevel.CRITICAL:
            score *= 1.1
        elif finding.severity == SeverityLevel.LOW:
            score *= 0.9

        # Adjust based on explanation quality
        explanation_length = len(finding.explanation)
        if explanation_length < 20:
            score *= 0.8
        elif explanation_length > 100:
            score *= 1.05

        return min(1.0, score)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get validator statistics.

        Returns:
            Statistics dictionary
        """
        return {
            "agent": self.agent_name,
            "validations_performed": self.validations_performed,
            "false_positives_filtered": self.false_positives_filtered,
            "average_confidence": self.average_confidence,
        }


def test_quality_validator():
    """Test quality validator."""
    print("✓ Testing Quality Validator...")

    validator = QualityValidator()
    log_agent_event(validator.agent_name, "initialization_complete")

    # Create test findings
    findings = [
        CodeSmellFinding(
            smell_type="Long Method",
            location="line 10-45",
            severity=SeverityLevel.HIGH,
            explanation="Method exceeds 35 lines and performs multiple tasks",
            refactoring="Extract smaller methods, apply single responsibility",
            confidence=0.85,
            agent_name="detector_test",
            timestamp=datetime.now().isoformat(),
        ),
        CodeSmellFinding(
            smell_type="Feature Envy",
            location="line 50-55",
            severity=SeverityLevel.MEDIUM,
            explanation="This is a comment about something",  # Low quality
            refactoring="Move to appropriate class",
            confidence=0.5,
            agent_name="detector_test",
            timestamp=datetime.now().isoformat(),
        ),
    ]

    # Validate findings
    print("  Validating findings...")
    validated = validator.validate_findings(findings)
    print(f"  Validated: {len(validated)} out of {len(findings)}")

    # Get suggestions
    suggestions = validator.suggest_improvements(validated)
    print("\n  Improvement suggestions:")
    for suggestion in suggestions["overall"]:
        print(f"    - {suggestion}")

    # Get stats
    stats = validator.get_stats()
    print("\n✓ Statistics:")
    print(f"  False positives filtered: {stats['false_positives_filtered']}")
    print(f"  Average confidence: {stats['average_confidence']:.2f}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_quality_validator()
