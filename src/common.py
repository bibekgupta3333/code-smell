"""
Common Utility Functions for Multi-Agent System
Shared utilities for agent operations, result merging, and naming conventions.

Architecture: Supports all agent modules with common operations
"""

import re
import json
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class SeverityLevel(str, Enum):
    """Code smell severity levels."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class CodeSmellFinding:
    """Represents a single code smell finding."""

    smell_type: str
    location: str  # e.g., "line 10-45" or "class MyClass:method_name"
    severity: SeverityLevel
    explanation: str
    refactoring: str
    confidence: float  # 0.0-1.0
    agent_name: str  # Which agent detected this
    timestamp: str


@dataclass
class DetectionResult:
    """Aggregated detection results from all agents."""

    findings: List[CodeSmellFinding]
    summary: str
    total_smells: int
    critical_count: int
    high_count: int
    medium_count: int
    low_count: int
    analysis_time_seconds: float
    timestamp: str


def get_agent_name(agent_type: str, specialization: Optional[str] = None) -> str:
    """
    Generate a unique agent name from type and specialization.

    Args:
        agent_type: Type of agent (coordinator, detector, retriever, validator)
        specialization: Optional specialization (e.g., "Long Method expert")

    Returns:
        Agent name (e.g., "detector_long_method_expert")
    """
    base_name = to_safe_name(agent_type)

    if specialization:
        spec_name = to_safe_name(specialization)
        return f"{base_name}_{spec_name}"

    return base_name


def to_safe_name(text: str) -> str:
    """
    Convert text to safe filesystem/identifier name.

    Args:
        text: Text to sanitize

    Returns:
        Sanitized name (lowercase, alphanumeric + underscore)
    """
    # Convert to lowercase
    text = text.lower()

    # Replace spaces and hyphens with underscores
    text = re.sub(r'[\s\-]+', '_', text)

    # Remove special characters, keep only alphanumeric and underscore
    text = re.sub(r'[^a-z0-9_]', '', text)

    # Remove leading/trailing underscores
    text = text.strip('_')

    # Replace multiple underscores with single
    text = re.sub(r'_+', '_', text)

    return text


def merge_results(
    results_list: List[DetectionResult],
    strategy: str = "union",
) -> DetectionResult:
    """
    Merge multiple detection results from different agents.

    Args:
        results_list: List of DetectionResult from agents
        strategy: Merge strategy ("union" = all findings, "intersection" = common findings)

    Returns:
        Merged DetectionResult
    """
    if not results_list:
        return DetectionResult(
            findings=[],
            summary="No results to merge",
            total_smells=0,
            critical_count=0,
            high_count=0,
            medium_count=0,
            low_count=0,
            analysis_time_seconds=0.0,
            timestamp=datetime.now().isoformat(),
        )

    if strategy == "union":
        # Combine all findings, deduplicate by (type, location, severity)
        merged_findings = _deduplicate_findings(
            [f for r in results_list for f in r.findings]
        )

    elif strategy == "intersection":
        # Keep only findings detected by multiple agents
        merged_findings = _find_common_findings(results_list)

    else:
        logger.warning("Unknown merge strategy: %s, using union", strategy)
        merged_findings = _deduplicate_findings(
            [f for r in results_list for f in r.findings]
        )

    # Count by severity
    severity_counts = {
        SeverityLevel.CRITICAL: 0,
        SeverityLevel.HIGH: 0,
        SeverityLevel.MEDIUM: 0,
        SeverityLevel.LOW: 0,
    }

    for finding in merged_findings:
        severity_counts[finding.severity] += 1

    # Calculate total analysis time
    total_time = sum(r.analysis_time_seconds for r in results_list)

    # Create summary
    summary = f"Found {len(merged_findings)} code smells: "
    summary += f"{severity_counts[SeverityLevel.CRITICAL]} critical, "
    summary += f"{severity_counts[SeverityLevel.HIGH]} high, "
    summary += f"{severity_counts[SeverityLevel.MEDIUM]} medium, "
    summary += f"{severity_counts[SeverityLevel.LOW]} low"

    return DetectionResult(
        findings=merged_findings,
        summary=summary,
        total_smells=len(merged_findings),
        critical_count=severity_counts[SeverityLevel.CRITICAL],
        high_count=severity_counts[SeverityLevel.HIGH],
        medium_count=severity_counts[SeverityLevel.MEDIUM],
        low_count=severity_counts[SeverityLevel.LOW],
        analysis_time_seconds=total_time,
        timestamp=datetime.now().isoformat(),
    )


def _deduplicate_findings(findings: List[CodeSmellFinding]) -> List[CodeSmellFinding]:
    """
    Deduplicate findings by (type, location, severity).
    When duplicates exist, keep the one with highest confidence.

    Args:
        findings: List of findings

    Returns:
        Deduplicated list
    """
    seen = {}  # Key: (type, location, severity) -> Finding

    for finding in findings:
        key = (finding.smell_type, finding.location, finding.severity)

        if key not in seen or finding.confidence > seen[key].confidence:
            seen[key] = finding

    return list(seen.values())


def _find_common_findings(results_list: List[DetectionResult]) -> List[CodeSmellFinding]:
    """
    Find findings that appear in most results.
    Keep findings detected by 2+ agents.

    Args:
        results_list: List of DetectionResult

    Returns:
        Common findings
    """
    if not results_list:
        return []

    # Count findings by (type, location, severity)
    finding_counts = {}

    for result in results_list:
        for finding in result.findings:
            key = (finding.smell_type, finding.location, finding.severity)
            if key not in finding_counts:
                finding_counts[key] = {"count": 0, "finding": finding}
            finding_counts[key]["count"] += 1

    # Keep findings detected by 2+ agents
    min_detections = max(2, len(results_list) // 2)
    common_findings = [
        item["finding"]
        for item in finding_counts.values()
        if item["count"] >= min_detections
    ]

    logger.info("Found %d common findings across agents", len(common_findings))
    return common_findings


def serialize_finding(finding: CodeSmellFinding) -> Dict[str, Any]:
    """
    Serialize a finding to JSON-compatible dict.

    Args:
        finding: CodeSmellFinding

    Returns:
        Serializable dict
    """
    data = asdict(finding)
    data["severity"] = finding.severity.value
    return data


def serialize_result(result: DetectionResult) -> Dict[str, Any]:
    """
    Serialize a DetectionResult to JSON-compatible dict.

    Args:
        result: DetectionResult

    Returns:
        Serializable dict
    """
    return {
        "findings": [serialize_finding(f) for f in result.findings],
        "summary": result.summary,
        "total_smells": result.total_smells,
        "critical_count": result.critical_count,
        "high_count": result.high_count,
        "medium_count": result.medium_count,
        "low_count": result.low_count,
        "analysis_time_seconds": result.analysis_time_seconds,
        "timestamp": result.timestamp,
    }


def deserialize_finding(data: Dict[str, Any]) -> CodeSmellFinding:
    """
    Deserialize a finding from dict.

    Args:
        data: Dict from JSON

    Returns:
        CodeSmellFinding
    """
    return CodeSmellFinding(
        smell_type=data["smell_type"],
        location=data["location"],
        severity=SeverityLevel(data["severity"]),
        explanation=data["explanation"],
        refactoring=data["refactoring"],
        confidence=data["confidence"],
        agent_name=data["agent_name"],
        timestamp=data["timestamp"],
    )


def format_finding_for_display(finding: CodeSmellFinding) -> str:
    """
    Format a finding for display/logging.

    Args:
        finding: CodeSmellFinding

    Returns:
        Formatted string
    """
    return (
        f"[{finding.severity.value}] {finding.smell_type} at {finding.location}\n"
        f"  Explanation: {finding.explanation}\n"
        f"  Refactoring: {finding.refactoring}\n"
        f"  Confidence: {finding.confidence:.2f}\n"
        f"  Detected by: {finding.agent_name}"
    )


def format_result_for_display(result: DetectionResult) -> str:
    """
    Format a DetectionResult for display.

    Args:
        result: DetectionResult

    Returns:
        Formatted string
    """
    lines = [
        "=" * 60,
        "CODE SMELL ANALYSIS RESULTS",
        "=" * 60,
        result.summary,
        f"\nAnalysis Time: {result.analysis_time_seconds:.2f}s",
        f"Timestamp: {result.timestamp}",
        "\nFindings:",
        "-" * 60,
    ]

    if not result.findings:
        lines.append("No code smells detected.")
    else:
        for i, finding in enumerate(result.findings, 1):
            lines.append(f"\n{i}. {format_finding_for_display(finding)}")

    lines.append("=" * 60)
    return "\n".join(lines)


def parse_smell_severity(severity_str: str) -> SeverityLevel:
    """
    Parse severity string to SeverityLevel enum.

    Args:
        severity_str: String representation of severity

    Returns:
        SeverityLevel enum value
    """
    mapping = {
        "low": SeverityLevel.LOW,
        "medium": SeverityLevel.MEDIUM,
        "high": SeverityLevel.HIGH,
        "critical": SeverityLevel.CRITICAL,
    }

    normalized = severity_str.lower().strip()
    return mapping.get(normalized, SeverityLevel.MEDIUM)


# Test function
def test_common():
    """Test common utility functions."""
    print("✓ Testing common utilities...")

    # Test naming
    name = get_agent_name("detector", "Long Method")
    print(f"  Agent name: {name}")

    # Test safe naming
    safe = to_safe_name("Long Method Expert!!!")
    print(f"  Safe name: {safe}")

    # Test finding creation
    finding = CodeSmellFinding(
        smell_type="Long Method",
        location="line 10-45",
        severity=SeverityLevel.HIGH,
        explanation="Method exceeds 30 lines",
        refactoring="Extract smaller methods",
        confidence=0.95,
        agent_name="detector_long_method",
        timestamp=datetime.now().isoformat(),
    )
    print(f"  Finding created: {finding.smell_type}")

    # Test result creation
    result = DetectionResult(
        findings=[finding],
        summary="Found 1 code smell",
        total_smells=1,
        critical_count=0,
        high_count=1,
        medium_count=0,
        low_count=0,
        analysis_time_seconds=2.5,
        timestamp=datetime.now().isoformat(),
    )
    print(f"  Result created with {result.total_smells} smell(s)")

    # Test serialization
    serialized = serialize_result(result)
    print(f"  Result serialized: {json.dumps(serialized, indent=2)[:100]}...")

    print("✓ All common utilities working")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_common()
