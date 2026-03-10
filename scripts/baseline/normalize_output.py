#!/usr/bin/env python3
"""
Baseline Tool Output Normalizer

Converts output from different baseline tools (PMD, Checkstyle, SpotBugs, SonarQube, IntelliJ)
to a common JSON schema for comparison.

Common Schema:
{
    "tool": "PMD|Checkstyle|SpotBugs|SonarQube|IntelliJ",
    "file": "path/to/File.java",
    "line": 42,
    "column": 5,
    "rule": "LongMethod",
    "message": "Method is too long",
    "severity": "HIGH|MEDIUM|LOW",
    "confidence": 0.85
}

Usage:
    python scripts/normalize_baseline_output.py \
        --input results/predictions/baseline/java_pmd.json \
        --tool pmd \
        --output results/predictions/baseline/normalized_pmd.json
"""

import argparse
import json
import logging
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BaselineNormalizer:
    """Normalizes baseline tool outputs to common schema."""

    # Severity mappings
    SEVERITY_MAP = {
        'HIGH': 'HIGH',
        'CRITICAL': 'HIGH',
        'ERROR': 'HIGH',
        'MAJOR': 'MEDIUM',
        'MEDIUM': 'MEDIUM',
        'MINOR': 'LOW',
        'LOW': 'LOW',
        'INFO': 'LOW',
        'WARNING': 'MEDIUM',
    }

    def __init__(self):
        self.tool = None
        self.findings = []

    def parse_pmd_json(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse PMD JSON output."""
        findings = []
        for file_entry in data.get('files', []):
            filename = file_entry.get('filename', 'Unknown')
            for violation in file_entry.get('violations', []):
                finding = {
                    'tool': 'PMD',
                    'file': filename,
                    'line': violation.get('beginline', 0),
                    'column': violation.get('begincolumn', 0),
                    'rule': violation.get('rule', 'Unknown'),
                    'message': violation.get('description', ''),
                    'severity': self.SEVERITY_MAP.get(
                        str(violation.get('priority', 3)).upper(), 'LOW'
                    ),
                    'confidence': 0.9,
                    'external_url': violation.get('externalInfoUrl', '')
                }
                findings.append(finding)
        return findings

    def parse_checkstyle_xml(self, xml_file: str) -> List[Dict[str, Any]]:
        """Parse Checkstyle XML output."""
        findings = []
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()

            for file_elem in root.findall('file'):
                filename = file_elem.get('name', 'Unknown')
                for error in file_elem.findall('error'):
                    # Checkstyle severity: error, warning, info
                    severity = error.get('severity', 'warning').upper()
                    finding = {
                        'tool': 'Checkstyle',
                        'file': filename,
                        'line': int(error.get('line', 0)),
                        'column': int(error.get('column', 0)),
                        'rule': error.get('source', 'Unknown'),
                        'message': error.get('message', ''),
                        'severity': self.SEVERITY_MAP.get(severity, 'MEDIUM'),
                        'confidence': 0.95,
                    }
                    findings.append(finding)
        except Exception as e:
            logger.error(f"Error parsing Checkstyle XML: {e}")
        return findings

    def parse_spotbugs_xml(self, xml_file: str) -> List[Dict[str, Any]]:
        """Parse SpotBugs XML output."""
        findings = []
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()

            for bug in root.findall('.//BugInstance'):
                bug_type = bug.get('type', 'Unknown')
                abbrev = bug.get('abbrev', 'X')
                category = bug.get('category', 'Unknown')

                # Get source file and line info
                source_line = bug.find('.//SourceLine')
                if source_line is None:
                    continue

                filename = source_line.get('sourcefile', 'Unknown')
                line = int(source_line.get('start', '0'))

                # Severity based on priority
                priority = int(bug.get('priority', 3))
                severity_map_spotbugs = {1: 'HIGH', 2: 'MEDIUM', 3: 'LOW'}
                severity = severity_map_spotbugs.get(priority, 'LOW')

                finding = {
                    'tool': 'SpotBugs',
                    'file': filename,
                    'line': line,
                    'column': 0,
                    'rule': f"{abbrev}:{bug_type}",
                    'message': bug.get('instanceHash', '') or bug_type,
                    'severity': severity,
                    'confidence': 0.92,
                    'category': category,
                }
                findings.append(finding)
        except Exception as e:
            logger.error(f"Error parsing SpotBugs XML: {e}")
        return findings

    def parse_sonarqube_json(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse SonarQube JSON output."""
        findings = []
        for issue in data.get('issues', []):
            # SonarQube severity: BLOCKER, CRITICAL, MAJOR, MINOR, INFO
            severity = self.SEVERITY_MAP.get(
                issue.get('severity', 'MINOR').upper(), 'LOW'
            )

            finding = {
                'tool': 'SonarQube',
                'file': issue.get('component', 'Unknown').split(':')[-1],
                'line': issue.get('line', 0),
                'column': issue.get('textRange', {}).get('startOffset', 0),
                'rule': issue.get('rule', 'Unknown'),
                'message': issue.get('message', ''),
                'severity': severity,
                'confidence': 0.88,
                'issue_type': issue.get('type', 'Unknown'),
            }
            findings.append(finding)
        return findings

    def normalize(self, tool: str, input_file: str) -> List[Dict[str, Any]]:
        """
        Normalize baseline tool output to common schema.

        Args:
            tool: Tool name (pmd, checkstyle, spotbugs, sonarqube)
            input_file: Path to tool output file

        Returns:
            List of normalized findings
        """
        tool = tool.lower()
        self.tool = tool

        try:
            if tool == 'pmd':
                with open(input_file) as f:
                    data = json.load(f)
                self.findings = self.parse_pmd_json(data)
            elif tool == 'checkstyle':
                self.findings = self.parse_checkstyle_xml(input_file)
            elif tool == 'spotbugs':
                self.findings = self.parse_spotbugs_xml(input_file)
            elif tool == 'sonarqube':
                with open(input_file) as f:
                    data = json.load(f)
                self.findings = self.parse_sonarqube_json(data)
            else:
                raise ValueError(f"Unknown tool: {tool}")

            logger.info(f"Parsed {len(self.findings)} findings from {tool}")
            return self.findings
        except Exception as e:
            logger.error(f"Error normalizing {tool} output: {e}")
            return []

    def save(self, output_file: str) -> None:
        """Save normalized findings to JSON file."""
        output = {
            'tool': self.tool,
            'timestamp': datetime.now().isoformat(),
            'total_findings': len(self.findings),
            'findings': self.findings
        }

        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
        logger.info(f"Saved normalized output to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Normalize baseline tool outputs to common schema'
    )
    parser.add_argument(
        '--tool',
        required=True,
        choices=['pmd', 'checkstyle', 'spotbugs', 'sonarqube', 'intellij'],
        help='Baseline tool name'
    )
    parser.add_argument(
        '--input',
        required=True,
        help='Input file from baseline tool'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Output normalized JSON file'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    if not Path(args.input).exists():
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)

    normalizer = BaselineNormalizer()
    findings = normalizer.normalize(args.tool, args.input)

    if findings:
        normalizer.save(args.output)
        print(f"✓ Normalized {len(findings)} findings from {args.tool}")
        print(f"  Saved to: {args.output}")
    else:
        print(f"✗ No findings to normalize")
        sys.exit(1)


if __name__ == '__main__':
    main()
