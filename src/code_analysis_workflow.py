"""
Code Analysis Workflow
Main orchestration workflow for multi-agent code smell detection.

Architecture: Main entry point for code analysis
Combines all agents in coordinated workflow
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime
import json

from src.analysis_coordinator import AnalysisCoordinator
from src.common import DetectionResult, format_result_for_display, serialize_result
from src.logger import setup_logging, log_workflow_step
from config import RESULTS_DIR

logger = logging.getLogger(__name__)


class CodeAnalysisWorkflow:
    """
    Main code analysis workflow.

    Orchestrates:
    - Initialization of logging and agents
    - Submission of code for analysis
    - Coordination of analysis
    - Reporting of results
    - Persistence of results

    Example:
        >>> workflow = CodeAnalysisWorkflow()
        >>> result = await workflow.analyze_code(code, "example.py")
        >>> workflow.save_results(result)
    """

    def __init__(self, log_dir: Path = RESULTS_DIR / "logs"):
        """
        Initialize analysis workflow.

        Args:
            log_dir: Directory for logs
        """
        # Setup logging
        self.log_file = setup_logging(log_dir=log_dir, log_name="code_analysis")

        # Initialize coordinator (which manages all agents)
        self.coordinator = AnalysisCoordinator(
            chunk_size=512,
            max_concurrent_detectors=2,  # M4 Pro optimization
        )

        # Statistics
        self.analyses_count = 0
        self.total_findings = 0
        self.analysis_times = []

        logger.info("Code Analysis Workflow initialized")  # noqa: G201
        logger.info("Log file: %s", self.log_file)  # noqa: G201

    async def analyze_code(
        self,
        code: str,
        file_name: str = "code.py",
        _language: Optional[str] = None,
    ) -> DetectionResult:
        """
        Analyze code for code smells.

        Args:
            code: Source code to analyze
            file_name: File name for context
            language: Programming language (detected if None)

        Returns:
            DetectionResult with all findings
        """
        log_workflow_step("analyze_code", "start", {"file": file_name, "code_length": len(code)})

        # Validate input
        if not code or not code.strip():
            logger.warning("Empty code provided")
            return DetectionResult(
                findings=[],
                summary="No code provided",
                total_smells=0,
                critical_count=0,
                high_count=0,
                medium_count=0,
                low_count=0,
                analysis_time_seconds=0.0,
                timestamp=datetime.now().isoformat(),
            )

        try:
            # Coordinate analysis through manager agent
            result = await self.coordinator.coordinate_analysis(code, file_name)

            # Update statistics
            self.analyses_count += 1
            self.total_findings += result.total_smells
            self.analysis_times.append(result.analysis_time_seconds)

            log_workflow_step("analyze_code", "complete", {
                "file": file_name,
                "findings": result.total_smells,
                "time": result.analysis_time_seconds,
            })

            return result

        except ValueError as e:  # noqa: B014
            logger.error("Analysis error: %s", e)  # noqa: G201
            log_workflow_step("analyze_code", "error", {"error": str(e)})

            return DetectionResult(
                findings=[],
                summary=f"Analysis failed: {str(e)}",
                total_smells=0,
                critical_count=0,
                high_count=0,
                medium_count=0,
                low_count=0,
                analysis_time_seconds=0.0,
                timestamp=datetime.now().isoformat(),
            )

    async def analyze_files(
        self,
        file_paths: List[Path],
    ) -> Dict[str, DetectionResult]:
        """
        Analyze multiple files.

        Args:
            file_paths: Paths to files to analyze

        Returns:
            Dictionary of file_path -> DetectionResult
        """
        results = {}

        for file_path in file_paths:
            try:
                # Read file
                if not file_path.exists():
                    logger.warning("File not found: %s", file_path)  # noqa: G201
                    continue

                code = file_path.read_text()

                # Analyze
                result = await self.analyze_code(code, str(file_path))
                results[str(file_path)] = result

            except ValueError as e:  # noqa: B014
                logger.error("Failed to analyze %s: %s", file_path, e)  # noqa: G201

        return results

    def save_results(
        self,
        result: DetectionResult,
        file_name: str = "analysis_result",
    ) -> Path:
        """
        Save analysis results to file.

        Args:
            result: DetectionResult to save
            file_name: Base name for result file

        Returns:
            Path to saved file
        """
        # Create results directory
        results_dir = RESULTS_DIR / "detections"
        results_dir.mkdir(parents=True, exist_ok=True)

        # Create timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = results_dir / f"{file_name}_{timestamp}.json"

        try:
            # Serialize and save
            serialized = serialize_result(result)
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(serialized, f, indent=2)

            logger.info("Results saved to %s", output_file)  # noqa: G201
            return output_file

        except ValueError as e:  # noqa: B014
            logger.error("Failed to save results: %s", e)  # noqa: G201
            return output_file

    def print_results(self, result: DetectionResult) -> None:
        """
        Print results to console.

        Args:
            result: DetectionResult to print
        """
        print(format_result_for_display(result))

    def get_workflow_stats(self) -> Dict[str, Any]:
        """
        Get workflow statistics.

        Returns:
            Statistics dictionary
        """
        avg_time = (
            sum(self.analysis_times) / len(self.analysis_times)
            if self.analysis_times
            else 0.0
        )

        return {
            "analyses_count": self.analyses_count,
            "total_findings": self.total_findings,
            "average_findings_per_analysis": (
                self.total_findings / self.analyses_count
                if self.analyses_count > 0
                else 0
            ),
            "average_analysis_time_seconds": avg_time,
            "log_file": str(self.log_file),
            "coordinator_stats": self.coordinator.get_stats(),
        }

    async def close(self) -> None:
        """Close workflow and cleanup resources."""
        logger.info("Closing workflow")
        # Could add cleanup here if needed


async def run_example_analysis():
    """Run example code analysis."""
    print("=" * 60)
    print("CODE SMELL DETECTION - MULTI-AGENT WORKFLOW")
    print("=" * 60)

    # Initialize workflow
    workflow = CodeAnalysisWorkflow()

    # Example code with multiple smells
    example_code = '''
def process_user_data(user_id, data, config=None):
    """Process user data with multiple steps."""
    if not user_id:
        return None

    # Validation
    if not data:
        return None

    # Transform
    transformed = {}
    for key, value in data.items():
        if isinstance(value, str):
            transformed[key] = value.strip().lower()
        elif isinstance(value, list):
            transformed[key] = [v.strip() if isinstance(v, str) else v for v in value]
        else:
            transformed[key] = value

    # Enrich
    enriched = {
        **transformed,
        "user_id": user_id,
        "timestamp": datetime.now().isoformat(),
    }

    # Store
    store_to_db(enriched)

    # Log
    log_event("user_data_processed", enriched)

    # Notify
    notify_user(user_id)

    return enriched

class UserManager:
    def __init__(self):
        self.users = {}
        self.cache = {}

    def add_user(self, user):
        pass

    def get_user(self, user_id):
        return self.users.get(user_id)

    def process_user(self, user):
        pass
'''

    print("\nAnalyzing code...")
    print("-" * 60)

    # Run analysis
    result = await workflow.analyze_code(example_code, "example.py")

    # Print results
    workflow.print_results(result)

    # Save results
    output_file = workflow.save_results(result, "example")
    print(f"\nResults saved to: {output_file}")

    # Print workflow statistics
    stats = workflow.get_workflow_stats()
    print("\nWorkflow Statistics:")
    print(f"  Analyses completed: {stats['analyses_count']}")
    print(f"  Total findings: {stats['total_findings']}")
    print(f"  Avg time per analysis: {stats['average_analysis_time_seconds']:.2f}s")

    await workflow.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    asyncio.run(run_example_analysis())
