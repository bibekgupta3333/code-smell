"""
Analysis Coordinator Agent (Manager)
Orchestrates code smell analysis workflow with other agents.

Architecture: Manager agent in multi-agent system
Coordinates analysis, splits code, assigns tasks, aggregates results
"""

import asyncio
import logging
from typing import List, Dict, Any
from datetime import datetime
from time import time

from src.code_chunker import CodeChunker
from src.code_smell_detector import CodeSmellDetector
from src.quality_validator import QualityValidator
from src.common import CodeSmellFinding, DetectionResult
from src.logger import log_agent_event, log_workflow_step
from src.code_parser import CodeParser

logger = logging.getLogger(__name__)


class AnalysisCoordinator:
    """
    Analysis Coordinator Agent (Manager Role).

    Responsible for:
    - Orchestrating the analysis workflow
    - Splitting code into chunks
    - Assigning detection tasks to detectors
    - Aggregating results from multiple agents
    - Tracking analysis in database

    Example:
        >>> coordinator = AnalysisCoordinator()
        >>> result = await coordinator.coordinate_analysis(code, file_name="app.py")
    """

    def __init__(
        self,
        chunk_size: int = 512,
        max_concurrent_detectors: int = 2,  # M4 Pro optimization
    ):
        """
        Initialize analysis coordinator.

        Args:
            chunk_size: Max tokens per chunk
            max_concurrent_detectors: Max parallel detector agents
        """
        self.agent_name = "coordinator_manager"
        self.code_chunker = CodeChunker(max_chunk_tokens=chunk_size)
        self.code_parser = CodeParser()
        self.max_concurrent_detectors = max_concurrent_detectors

        # Detector pool (specialize some detectors)
        self.detectors = [
            CodeSmellDetector(specialization="Long Method expert"),
            CodeSmellDetector(specialization="God Class expert"),
        ]
        self.validator = QualityValidator()

        # Statistics
        self.analyses_completed = 0

        logger.info("Analysis Coordinator initialized: %s", self.agent_name)  # noqa: G201
        logger.info("Detectors: %d, Max concurrent: %d", len(self.detectors), max_concurrent_detectors)  # noqa: G201

    async def coordinate_analysis(
        self,
        code: str,
        file_name: str = "code.py",
    ) -> DetectionResult:
        """
        Coordinate end-to-end analysis.

        Args:
            code: Code to analyze
            file_name: File name for context

        Returns:
            Final analysis result
        """
        start_time = time()

        log_workflow_step(
            f"Analysis: {file_name}",
            "start",
            {"code_length": len(code)},
        )

        try:
            # Step 1: Analyze code
            log_workflow_step(f"Analysis: {file_name}", "parse_code")
            _ = self.code_parser.detect_language(code)
            _ = self.code_parser.extract_metrics(code)

            # Step 2: Split into chunks
            log_workflow_step(f"Analysis: {file_name}", "chunk_code")
            chunks = self._split_code_into_chunks(code)
            logger.info("Split into %d chunks", len(chunks))  # noqa: G201

            # Step 3: Detect smells in parallel (with concurrency limit)
            log_workflow_step(f"Analysis: {file_name}", "detect_smells")
            all_findings = await self._detect_with_agents(chunks)
            logger.info("Detected %d findings across all chunks", len(all_findings))  # noqa: G201

            # Step 4: Validate findings
            log_workflow_step(f"Analysis: {file_name}", "validate_findings")
            validated_findings = self.validator.validate_findings(all_findings)
            logger.info("Validated to %d findings", len(validated_findings))  # noqa: G201

            # Step 5: Aggregate results
            log_workflow_step(f"Analysis: {file_name}", "aggregate_results")
            result = self._create_result(validated_findings, start_time)

            self.analyses_completed += 1
            log_workflow_step(f"Analysis: {file_name}", "complete", {"result": "success"})

            return result

        except ValueError as e:  # noqa: B014
            logger.error("Analysis failed: %s", e)  # noqa: G201
            log_workflow_step(
                f"Analysis: {file_name}",
                "error",
                {"error": str(e)},
            )

            # Return empty result on error
            return DetectionResult(
                findings=[],
                summary=f"Analysis failed: {str(e)}",
                total_smells=0,
                critical_count=0,
                high_count=0,
                medium_count=0,
                low_count=0,
                analysis_time_seconds=time() - start_time,
                timestamp=datetime.now().isoformat(),
            )

    def _split_code_into_chunks(self, code: str) -> List[str]:
        """
        Split code into analysis chunks.

        Args:
            code: Code to split

        Returns:
            List of code chunks
        """
        # Check code size
        if len(code) < 512:
            # Small code, analyze as one chunk
            return [code]

        # Use code chunker for larger code
        chunks_obj = self.code_chunker.chunk_python(code)
        chunks = [chunk.content for chunk in chunks_obj]

        if not chunks:
            # Fallback to whole code
            return [code]

        return chunks

    async def _detect_with_agents(
        self,
        chunks: List[str],
    ) -> List[CodeSmellFinding]:
        """
        Run detections across chunks with multiple agents.

        Args:
            chunks: Code chunks to analyze

        Returns:
            All findings from all chunks
        """
        all_findings = []

        # Process chunks with concurrent limit
        tasks = []
        for i, chunk in enumerate(chunks[:5]):  # Limit chunks for M4 Pro
            # Distribute chunks among detectors
            detector = self.detectors[i % len(self.detectors)]
            task = detector.detect_smells(chunk, use_rag=True)
            tasks.append(task)

        # Run detection tasks with concurrency limit
        for i in range(0, len(tasks), self.max_concurrent_detectors):
            batch = tasks[i:i + self.max_concurrent_detectors]
            batch_results = await asyncio.gather(*batch, return_exceptions=True)

            for result in batch_results:
                if isinstance(result, list):
                    all_findings.extend(result)
                elif isinstance(result, Exception):
                    logger.warning("Detection error in batch: %s", result)  # noqa: G201

        return all_findings

    def _create_result(
        self,
        findings: List[CodeSmellFinding],
        start_time: float,
    ) -> DetectionResult:
        """
        Create final result from findings.

        Args:
            findings: Validated findings
            start_time: Analysis start time

        Returns:
            DetectionResult
        """
        # Count by severity
        severity_counts = {
            "CRITICAL": 0,
            "HIGH": 0,
            "MEDIUM": 0,
            "LOW": 0,
        }

        for finding in findings:
            severity_counts[finding.severity.value] += 1

        # Create summary
        summary = f"Found {len(findings)} code smell(s): "
        summary += f"{severity_counts['CRITICAL']} critical, "
        summary += f"{severity_counts['HIGH']} high, "
        summary += f"{severity_counts['MEDIUM']} medium, "
        summary += f"{severity_counts['LOW']} low"

        return DetectionResult(
            findings=findings,
            summary=summary,
            total_smells=len(findings),
            critical_count=severity_counts["CRITICAL"],
            high_count=severity_counts["HIGH"],
            medium_count=severity_counts["MEDIUM"],
            low_count=severity_counts["LOW"],
            analysis_time_seconds=time() - start_time,
            timestamp=datetime.now().isoformat(),
        )

    def get_stats(self) -> Dict[str, Any]:
        """
        Get coordinator statistics.

        Returns:
            Statistics dictionary
        """
        return {
            "agent": self.agent_name,
            "analyses_completed": self.analyses_completed,
            "detector_count": len(self.detectors),
            "detector_stats": [d.get_stats() for d in self.detectors],
            "validator_stats": self.validator.get_stats(),
        }


async def test_analysis_coordinator():
    """Test analysis coordinator."""
    print("✓ Testing Analysis Coordinator...")

    coordinator = AnalysisCoordinator()
    log_agent_event(coordinator.agent_name, "initialization_complete")

    # Test code
    sample_code = """
def long_processing_function(data):
    result = None
    for item in data:
        for subitem in item:
            for value in subitem:
                result = process(value)
                store(result)
                log_event()
    return result

class DataHandler:
    def __init__(self):
        self.cache = {}

    def process(self, data):
        pass
"""

    # Run analysis
    print("  Running analysis...")
    result = await coordinator.coordinate_analysis(sample_code, "test.py")

    print(f"  Result: {result.summary}")
    print(f"  Findings: {len(result.findings)}")
    print(f"  Time: {result.analysis_time_seconds:.2f}s")

    # Get stats
    stats = coordinator.get_stats()
    print("\n✓ Statistics:")
    print(f"  Analyses completed: {stats['analyses_completed']}")
    print(f"  Detectors: {stats['detector_count']}")

    log_agent_event(coordinator.agent_name, "testing_complete", {"findings": result.total_smells})


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_analysis_coordinator())
