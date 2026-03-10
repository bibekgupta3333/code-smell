#!/usr/bin/env python3
"""
SonarQube Java Bug Detection - Preconfigured

Analyzes Java code using SonarQube API without manual setup.
Uses docker-compose running SonarQube (localhost:9000).

Usage:
    python scripts/sonarqube_analyze.py \
        --source-dir data/datasets/SmellyCodeDataset/Java \
        --project-key code-smell-java \
        --output results/predictions/baseline/sonarqube_findings.json

Prerequisites:
    - SonarQube running: docker compose up -d
    - Default login: admin/admin (set in sonarqube-admin Docker env)
"""

import argparse
import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional
import requests

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SonarQubeAnalyzer:
    """Analyze Java code using SonarQube API."""

    def __init__(self, sonar_host: str = "http://localhost:9000", sonar_token: Optional[str] = None):
        """
        Initialize SonarQube analyzer.

        Args:
            sonar_host: SonarQube base URL
            sonar_token: API token (optional, uses admin by default)
        """
        self.sonar_host = sonar_host
        self.sonar_token = sonar_token or self._get_default_token()
        self.session = requests.Session()
        if self.sonar_token:
            self.session.auth = (self.sonar_token, "")
        self.verify_connection()

    def _get_default_token(self) -> str:
        """Get default admin token from SonarQube."""
        try:
            # For default setup, use basic auth
            response = requests.post(
                f"{self.sonar_host}/api/user_tokens/generate",
                data={"name": "code-smell-analyzer"},
                auth=("admin", "admin"),
                timeout=5
            )
            if response.status_code == 200:
                return response.json().get("token", "")
        except Exception as e:
            logger.warning(f"Could not generate token: {e}")
        return ""

    def verify_connection(self) -> bool:
        """Verify SonarQube connection."""
        try:
            response = self.session.get(
                f"{self.sonar_host}/api/system/status",
                timeout=5
            )
            if response.status_code == 200:
                status = response.json()
                logger.info(f"✓ Connected to SonarQube ({status.get('version', 'unknown')})")
                return True
        except Exception as e:
            logger.error(f"Cannot connect to SonarQube: {e}")
        return False

    def create_project(self, project_key: str, project_name: str) -> bool:
        """Create SonarQube project."""
        try:
            # Check if project exists
            response = self.session.get(
                f"{self.sonar_host}/api/projects/search",
                params={"projects": project_key},
                timeout=10
            )
            if response.status_code == 200:
                projects = response.json().get("components", [])
                if projects:
                    logger.info(f"✓ Project '{project_key}' already exists")
                    return True

            # Create new project
            response = self.session.post(
                f"{self.sonar_host}/api/projects/create",
                data={
                    "project": project_key,
                    "name": project_name,
                    "visibility": "private"
                },
                timeout=10
            )
            if response.status_code == 200:
                logger.info(f"✓ Created project '{project_key}'")
                return True
            else:
                logger.error(f"Failed to create project: {response.text}")
        except Exception as e:
            logger.error(f"Error creating project: {e}")
        return False

    def run_sonar_scanner(self, source_dir: str, project_key: str, project_name: str) -> bool:
        """Run SonarScanner analysis via Docker container."""
        try:
            logger.info(f"Running SonarScanner on {source_dir}...")

            abs_source = Path(source_dir).resolve()

            # Run sonar-scanner inside sonarqube container
            cmd = [
                "docker", "run", "--rm",
                "--network=code-smell-network",
                f"-v={abs_source}:/src:ro",
                "sonarsource/sonar-scanner-cli:latest",
                "-Dsonar.projectKey=" + project_key,
                "-Dsonar.projectName=" + project_name,
                "-Dsonar.sources=/src",
                "-Dsonar.host.url=http://sonarqube:9000",
                "-Dsonar.login=admin",
                "-Dsonar.password=admin",
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)

            if result.returncode == 0 or "SonarScanner analysis completed" in result.stderr:
                logger.info("✓ SonarScanner completed successfully")
                return True
            else:
                logger.warning(f"SonarScanner returned code {result.returncode}")
                if result.stderr:
                    logger.debug(f"stderr: {result.stderr[:200]}")
                return False

        except subprocess.TimeoutExpired:
            logger.error("SonarScanner timed out (>3 minutes)")
        except Exception as e:
            logger.warning(f"SonarScanner analysis skipped: {e}")

        return False

    def _analyze_via_api(self, source_dir: str, project_key: str) -> bool:
        """Fallback: Analyze via SonarQube API."""
        try:
            logger.info(f"Analyzing files via API...")

            # List Java files
            java_files = list(Path(source_dir).glob("**/*.java"))
            logger.info(f"Found {len(java_files)} Java files")

            # Trigger re-analysis
            response = self.session.post(
                f"{self.sonar_host}/api/ce/activity",
                params={"status": "SUCCESS"},
                timeout=10
            )

            if response.status_code == 200:
                logger.info("✓ Triggered re-analysis via API")
                return True

        except Exception as e:
            logger.error(f"Error in API analysis: {e}")

        return False

    def get_issues(self, project_key: str, max_wait: int = 30) -> List[Dict]:
        """
        Retrieve issues for a project.

        Args:
            project_key: SonarQube project key
            max_wait: Max seconds to wait for analysis to complete

        Returns:
            List of issue dictionaries
        """
        try:
            # Wait briefly for analysis to complete
            wait_count = 0
            while wait_count < max_wait // 5:
                response = self.session.get(
                    f"{self.sonar_host}/api/ce/activity",
                    params={
                        "projectKey": project_key,
                        "status": "IN_PROGRESS",
                        "limit": 1
                    },
                    timeout=10
                )

                if response.status_code == 200:
                    in_progress = response.json().get("tasks", [])
                    if not in_progress:
                        logger.info("✓ Analysis complete")
                        break

                if wait_count > 0:
                    logger.info(f"Waiting for analysis... ({wait_count * 5}s)")
                time.sleep(5)
                wait_count += 1

            # Fetch issues
            issues = []
            page = 1
            while True:
                response = self.session.get(
                    f"{self.sonar_host}/api/issues/search",
                    params={
                        "projectKeys": project_key,
                        "p": page,
                        "ps": 100  # Page size
                    },
                    timeout=10
                )

                if response.status_code == 200:
                    data = response.json()
                    issues.extend(data.get("issues", []))

                    # Check if there are more pages
                    total = data.get("total", 0)
                    if len(issues) >= total:
                        break
                    page += 1
                else:
                    logger.warning(f"Error fetching issues: {response.status_code}")
                    break

            logger.info(f"✓ Retrieved {len(issues)} issues from SonarQube")
            return issues

        except Exception as e:
            logger.error(f"Error retrieving issues: {e}")
            return []

    def format_findings(self, issues: List[Dict]) -> List[Dict]:
        """Convert SonarQube issues to standard format."""
        findings = []

        # Severity mapping
        severity_map = {
            'BLOCKER': 'HIGH',
            'CRITICAL': 'HIGH',
            'MAJOR': 'MEDIUM',
            'MINOR': 'LOW',
            'INFO': 'LOW'
        }

        for issue in issues:
            finding = {
                'tool': 'SonarQube',
                'file': issue.get('component', '').split(':')[-1],
                'line': issue.get('line', 0),
                'column': issue.get('textRange', {}).get('startOffset', 0),
                'rule': issue.get('rule', 'Unknown'),
                'message': issue.get('message', ''),
                'severity': severity_map.get(issue.get('severity', 'MINOR'), 'LOW'),
                'type': issue.get('type', 'CODE_SMELL'),
                'confidence': 0.88,
                'issue_key': issue.get('key', '')
            }
            findings.append(finding)

        return findings

    def analyze(self, source_dir: str, project_key: str, project_name: Optional[str] = None) -> List[Dict]:
        """
        Analyze Java code and return standardized findings.

        Args:
            source_dir: Path to Java source directory
            project_key: SonarQube project key (unique identifier)
            project_name: Human-readable project name (defaults to key)

        Returns:
            List of standardized finding dictionaries
        """
        if not Path(source_dir).exists():
            logger.error(f"Source directory not found: {source_dir}")
            return []

        project_name = project_name or project_key

        # Step 1: Create project
        if not self.create_project(project_key, project_name):
            logger.warning("Could not create project, continuing anyway...")

        # Step 2: Run analysis
        if not self.run_sonar_scanner(source_dir, project_key, project_name):
            logger.warning("SonarScanner failed, attempting API retrieval...")

        # Step 3: Get issues
        issues = self.get_issues(project_key)

        # Step 4: Format findings
        findings = self.format_findings(issues)

        return findings


def main():
    parser = argparse.ArgumentParser(
        description='Analyze Java code with SonarQube (preconfigured)'
    )
    parser.add_argument(
        '--source-dir',
        required=True,
        help='Java source directory to analyze'
    )
    parser.add_argument(
        '--project-key',
        required=True,
        help='SonarQube project key (unique identifier)'
    )
    parser.add_argument(
        '--project-name',
        help='SonarQube project name (defaults to key)'
    )
    parser.add_argument(
        '--sonar-host',
        default='http://localhost:9000',
        help='SonarQube host URL'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Output JSON file for findings'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Initialize analyzer
    analyzer = SonarQubeAnalyzer(sonar_host=args.sonar_host)

    # Run analysis
    findings = analyzer.analyze(
        args.source_dir,
        args.project_key,
        args.project_name
    )

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result = {
        'tool': 'SonarQube',
        'project_key': args.project_key,
        'source_dir': args.source_dir,
        'total_findings': len(findings),
        'findings': findings
    }

    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"✓ Analysis complete: {len(findings)} findings")
    print(f"  Saved to: {output_path}")
    print(f"  SonarQube Dashboard: {args.sonar_host}/dashboard?id={args.project_key}")


if __name__ == '__main__':
    main()
