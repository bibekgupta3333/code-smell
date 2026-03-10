"""
Baseline Tool Runner for Code Smell Detection Benchmarking

Executes static analysis tools on Java, Python, and JavaScript source files,
parses their output, and normalizes all findings to a common JSON schema.

Java tools (5): PMD, Checkstyle, SpotBugs, SonarQube, IntelliJ IDEA
Python tools: pylint, flake8
JavaScript tools: ESLint

Usage (examples):
    # Run all tools on Java dataset
    python scripts/run_baseline_tools.py --input data/datasets/marv/ --language java --tool all

    # Run specific tools on Python file
    python scripts/run_baseline_tools.py --input sample.py --language python --tool pylint flake8

    # Run JavaScript analysis
    python scripts/run_baseline_tools.py --input src/ --language javascript --tool eslint

    # Run with timeout
    python scripts/run_baseline_tools.py --input sample.java --tool pmd checkstyle --timeout 120

    # Dry run (print plan without executing)
    python scripts/run_baseline_tools.py --input data/datasets/marv/ --dry-run

Output schema (per finding):
    {
        "tool": "PMD",
        "file": "Example.java",
        "line": 42,
        "smell_type": "Long Method",
        "severity": "HIGH",
        "confidence": 0.85,
        "explanation": "Method exceeds 30 lines...",
        "analysis_time_ms": 245.5,
        "language": "java"
    }

Architecture: Benchmarking Strategy Section 2 — Baseline Execution (Multi-language)
Resource Constraint: M4 Pro optimization (sequential execution, 60s timeout per file)
"""

import argparse
import json
import logging
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path for config imports
# scripts/baseline/ → project root is two levels up
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    PERFORMANCE_DIR,
    PREDICTIONS_DIR,
    RESULTS_DIR,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool paths (set by setup_baseline_tools.sh)
# ---------------------------------------------------------------------------
TOOLS_DIR = PROJECT_ROOT / "tools" / "baseline"
# PMD may be installed as pmd-7.0.0 or pmd-bin-7.0.0 depending on the release archive
PMD_BIN = TOOLS_DIR / "pmd-bin-7.0.0" / "bin" / "pmd"
if not PMD_BIN.exists():
    PMD_BIN = TOOLS_DIR / "pmd-7.0.0" / "bin" / "pmd"
CHECKSTYLE_JAR = TOOLS_DIR / "checkstyle-10.14.0-all.jar"
SPOTBUGS_BIN = TOOLS_DIR / "spotbugs-4.8.3" / "bin" / "spotbugs"
SONAR_SCANNER_BIN = TOOLS_DIR / "sonar-scanner-5.0.1.3006" / "bin" / "sonar-scanner"
INTELLIJ_PATH_FILE = TOOLS_DIR / "intellij_path.txt"

# Default analysis timeout per invocation
DEFAULT_TIMEOUT = 60

# ---------------------------------------------------------------------------
# Severity Normalization
# ---------------------------------------------------------------------------
SEVERITY_MAP = {
    # PMD priorities (1=highest, 5=lowest)
    "1": "CRITICAL", "2": "HIGH", "3": "MEDIUM", "4": "LOW", "5": "LOW",
    # Checkstyle
    "error": "HIGH", "warning": "MEDIUM", "info": "LOW",
    # SpotBugs priorities (1=highest)
    # SonarQube
    "BLOCKER": "CRITICAL", "CRITICAL": "CRITICAL",
    "MAJOR": "HIGH", "MINOR": "MEDIUM", "INFO": "LOW",
    # IntelliJ
    "ERROR": "HIGH", "WARNING": "MEDIUM", "WEAK WARNING": "LOW",
    "INFORMATION": "LOW",
}

# ---------------------------------------------------------------------------
# Rule → Code Smell Type Mappings
# These map tool-specific rule names to our unified CODE_SMELL_TYPES.
# Rules not in these mappings are tagged as "Other" and preserved for review.
# ---------------------------------------------------------------------------

PMD_SMELL_MAP: Dict[str, str] = {
    # Bloaters
    "ExcessiveMethodLength": "Long Method",
    "NcssMethodCount": "Long Method",
    "CyclomaticComplexity": "Long Method",
    "NPathComplexity": "Long Method",
    "CognitiveComplexity": "Long Method",
    "GodClass": "God Class",
    "TooManyFields": "God Class",
    "TooManyMethods": "God Class",
    "ExcessiveClassLength": "Large Class",
    "NcssTypeCount": "Large Class",
    "ExcessiveParameterList": "Long Parameter List",
    "ExcessivePublicCount": "Large Class",
    # Couplers
    "CouplingBetweenObjects": "Inappropriate Intimacy",
    "LawOfDemeter": "Message Chains",
    "LooseCoupling": "Feature Envy",
    # Dispensables
    "DataClass": "Data Class",
    "EmptyMethodInAbstractClassShouldBeAbstract": "Lazy Class",
    "UnusedPrivateField": "Dead Code",
    "UnusedPrivateMethod": "Dead Code",
    "UnusedLocalVariable": "Dead Code",
    # Change Preventers
    "AbstractClassWithoutAbstractMethod": "Refused Bequest",
    # OO Abusers
    "SwitchDensity": "Switch Statements",
    "MissingBreakInSwitch": "Switch Statements",
    "AvoidDeeplyNestedIfStmts": "Long Method",
}

CHECKSTYLE_SMELL_MAP: Dict[str, str] = {
    "MethodLength": "Long Method",
    "MethodCount": "God Class",
    "FileLength": "Large Class",
    "AnonInnerLength": "Long Method",
    "ParameterNumber": "Long Parameter List",
    "ClassFanOutComplexity": "God Class",
    "ClassDataAbstractionCoupling": "Feature Envy",
    "CyclomaticComplexity": "Long Method",
    "BooleanExpressionComplexity": "Long Method",
    "NPathComplexity": "Long Method",
    "JavaNCSS": "Long Method",
    "ExecutableStatementCount": "Long Method",
    "OuterTypeNumber": "Large Class",
    "MissingSwitchDefault": "Switch Statements",
    "InnerTypeLast": "Large Class",
}

SPOTBUGS_SMELL_MAP: Dict[str, str] = {
    # SpotBugs focuses on bugs; limited code smell coverage
    "SIC_INNER_SHOULD_BE_STATIC": "Large Class",
    "SIC_INNER_SHOULD_BE_STATIC_ANON": "Large Class",
    "SBSC_USE_STRINGBUFFER_CONCATENATION": "Long Method",
    "URF_UNREAD_FIELD": "Dead Code",
    "UUF_UNUSED_FIELD": "Dead Code",
    "UPM_UNCALLED_PRIVATE_METHOD": "Dead Code",
    "DLS_DEAD_LOCAL_STORE": "Dead Code",
}

SONARQUBE_SMELL_MAP: Dict[str, str] = {
    # Key rules for code smell detection
    "java:S138": "Long Method",      # Too many lines of code
    "java:S3776": "Long Method",     # Cognitive complexity
    "java:S1448": "God Class",       # Too many methods
    "java:S1200": "God Class",       # Too many dependencies
    "java:S107": "Long Parameter List",
    "java:S1133": "Dead Code",       # Deprecated code
    "java:S1068": "Dead Code",       # Unused private field
    "java:S1144": "Dead Code",       # Unused private method
    "java:S1151": "Switch Statements",  # Switch too many cases
    "java:S131": "Switch Statements",   # Missing switch default
    "java:S1121": "Long Method",     # Assignments in sub-expressions
    "java:S1186": "Lazy Class",      # Empty methods
    "java:S2160": "Refused Bequest", # Subclass overrides equals
}

INTELLIJ_SMELL_MAP: Dict[str, str] = {
    "MethodLength": "Long Method",
    "GodClass": "God Class",
    "ClassTooLarge": "Large Class",
    "LongParameterList": "Long Parameter List",
    "FeatureEnvy": "Feature Envy",
    "DataClass": "Data Class",
    "DuplicatedCode": "Duplicate Code",
    "SwitchStatement": "Switch Statements",
    "UnusedDeclaration": "Dead Code",
    "RefusedBequest": "Refused Bequest",
}

PYLINT_SMELL_MAP: Dict[str, str] = {
    # Pylint codes to code smell types
    "C0103": "Duplicate Code",  # invalid-name
    "C0111": "Data Class",      # missing-docstring
    "C0301": "Long Method",     # line-too-long
    "C0302": "Large Class",     # too-many-lines
    "R0902": "Large Class",     # too-many-instance-attributes
    "R0913": "Long Parameter List",  # too-many-arguments
    "R0914": "Long Method",     # too-many-locals
    "R0915": "Long Method",     # too-many-statements
    "W0612": "Dead Code",       # unused-variable
    "W0613": "Dead Code",       # unused-argument
    "F0401": "Dead Code",       # import-error
}

FLAKE8_SMELL_MAP: Dict[str, str] = {
    "E701": "Long Method",      # multiple statements on one line
    "E702": "Long Method",      # multiple statements with colon
    "W503": "Long Method",      # line break before binary operator
    "C901": "Long Method",      # McCabe complexity
}

ESLINT_SMELL_MAP: Dict[str, str] = {
    # ESLint rules to code smell types
    "no-unused-vars": "Dead Code",
    "no-undef": "Dead Code",
    "no-dupe-keys": "Duplicate Code",
    "complexity": "Long Method",
    "max-lines-per-function": "Long Method",
    "max-params": "Long Parameter List",
    "max-nested-callbacks": "Long Method",
    "no-duplicate-imports": "Duplicate Code",
    "max-classes-per-file": "Large Class",
}

JSLINT_SMELL_MAP: Dict[str, str] = {
    # JSLint/JSHint rules to code smell types
    "too_many_errors": "Long Method",
    "line_too_long": "Long Method",
    "too_many_arguments": "Long Parameter List",
    "unused": "Dead Code",
    "undef": "Dead Code",
    "complexity": "Long Method",
    "duplicate_code": "Duplicate Code",
}


# ---------------------------------------------------------------------------
# Language Detection
# ---------------------------------------------------------------------------
def detect_language(input_path: Path) -> str:
    """Detect programming language from file extensions.

    Args:
        input_path: Java, Python, or JavaScript file/directory.

    Returns:
        Language identifier: 'java', 'python', 'javascript'.
    """
    if input_path.is_file():
        suffix = input_path.suffix.lower()
        if suffix == ".java":
            return "java"
        elif suffix == ".py":
            return "python"
        elif suffix in {".js", ".jsx", ".ts", ".tsx"}:
            return "javascript"
    elif input_path.is_dir():
        # Check file extensions in directory
        java_files = list(input_path.glob("**/*.java"))
        py_files = list(input_path.glob("**/*.py"))
        js_files = list(input_path.glob("**/*.js")) + list(input_path.glob("**/*.jsx"))

        if java_files:
            return "java"
        elif py_files:
            return "python"
        elif js_files:
            return "javascript"

    return "java"  # Default


# ---------------------------------------------------------------------------
# Finding normalization
# ---------------------------------------------------------------------------
def _normalize_finding(
    tool: str,
    file: str,
    line: int,
    rule: str,
    severity_raw: str,
    message: str,
    smell_map: Dict[str, str],
    confidence: float = 0.8,
) -> Dict[str, Any]:
    """Normalize a single finding to the common JSON schema.

    Args:
        tool: Tool name.
        file: Source file path.
        line: Line number.
        rule: Tool-specific rule/check name.
        severity_raw: Raw severity string from tool.
        message: Description from tool.
        smell_map: Mapping from rule names to CODE_SMELL_TYPES.
        confidence: Default confidence (tools don't output confidence).

    Returns:
        Normalized finding dict.
    """
    # Map rule name to smell type, trying partial matches
    smell_type = smell_map.get(rule, "")
    if not smell_type:
        for key, val in smell_map.items():
            if key.lower() in rule.lower():
                smell_type = val
                break
    if not smell_type:
        smell_type = f"Other ({rule})"

    severity = SEVERITY_MAP.get(str(severity_raw), "MEDIUM")

    return {
        "tool": tool,
        "file": str(Path(file).name),
        "line": max(int(line), 1),
        "smell_type": smell_type,
        "severity": severity,
        "confidence": confidence,
        "explanation": message.strip(),
        "raw_rule": rule,
    }


# ---------------------------------------------------------------------------
# Tool Runners
# ---------------------------------------------------------------------------


def run_pmd(input_path: Path, timeout: int = DEFAULT_TIMEOUT) -> List[Dict[str, Any]]:
    """Run PMD analysis and return normalized findings.

    Args:
        input_path: Java source file or directory.
        timeout: Max seconds for PMD execution.

    Returns:
        List of normalized finding dicts.
    """
    if not PMD_BIN.exists():
        logger.error("PMD not installed at %s. Run setup_baseline_tools.sh first.", PMD_BIN)
        return []

    output_file = TOOLS_DIR / "test" / "pmd_output.xml"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(PMD_BIN), "check",
        "-d", str(input_path),
        "-R", "rulesets/java/quickstart.xml",
        "-f", "xml",
        "-r", str(output_file),
        "--no-cache",
    ]

    logger.info("Running PMD on %s ...", input_path)
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
        )
        # PMD exit code 4 = violations found (not an error)
        if result.returncode not in (0, 4):
            logger.warning("PMD stderr: %s", result.stderr[:500])
    except subprocess.TimeoutExpired:
        logger.error("PMD timed out after %ds", timeout)
        return []
    except FileNotFoundError:
        logger.error("PMD binary not found")
        return []

    return _parse_pmd_xml(output_file)


def _parse_pmd_xml(xml_path: Path) -> List[Dict[str, Any]]:
    """Parse PMD XML output to normalized findings."""
    if not xml_path.exists():
        return []

    findings = []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        # Handle namespace
        ns = ""
        if root.tag.startswith("{"):
            ns = root.tag.split("}")[0] + "}"

        for file_elem in root.findall(f"{ns}file"):
            filename = file_elem.get("name", "unknown")
            for violation in file_elem.findall(f"{ns}violation"):
                findings.append(_normalize_finding(
                    tool="PMD",
                    file=filename,
                    line=violation.get("beginline", "1"),
                    rule=violation.get("rule", "Unknown"),
                    severity_raw=violation.get("priority", "3"),
                    message=violation.text or "",
                    smell_map=PMD_SMELL_MAP,
                ))
    except ET.ParseError as e:
        logger.error("Failed to parse PMD XML: %s", e)

    logger.info("PMD found %d findings", len(findings))
    return findings


def run_checkstyle(input_path: Path, timeout: int = DEFAULT_TIMEOUT) -> List[Dict[str, Any]]:
    """Run Checkstyle analysis and return normalized findings."""
    if not CHECKSTYLE_JAR.exists():
        logger.error("Checkstyle not installed at %s", CHECKSTYLE_JAR)
        return []

    output_file = TOOLS_DIR / "test" / "checkstyle_output.xml"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "java", "-jar", str(CHECKSTYLE_JAR),
        "-c", "/google_checks.xml",  # Built-in Google style checks
        "-f", "xml",
        "-o", str(output_file),
        str(input_path),
    ]

    logger.info("Running Checkstyle on %s ...", input_path)
    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        logger.error("Checkstyle timed out after %ds", timeout)
        return []
    except FileNotFoundError:
        logger.error("Java not found in PATH")
        return []

    return _parse_checkstyle_xml(output_file)


def _parse_checkstyle_xml(xml_path: Path) -> List[Dict[str, Any]]:
    """Parse Checkstyle XML output to normalized findings."""
    if not xml_path.exists():
        return []

    findings = []
    try:
        tree = ET.parse(xml_path)
        for file_elem in tree.findall(".//file"):
            filename = file_elem.get("name", "unknown")
            for error in file_elem.findall("error"):
                # Extract check name from source (e.g., "...checks.sizes.MethodLengthCheck")
                source = error.get("source", "")
                rule = source.rsplit(".", 1)[-1].replace("Check", "") if source else "Unknown"

                findings.append(_normalize_finding(
                    tool="Checkstyle",
                    file=filename,
                    line=error.get("line", "1"),
                    rule=rule,
                    severity_raw=error.get("severity", "warning"),
                    message=error.get("message", ""),
                    smell_map=CHECKSTYLE_SMELL_MAP,
                ))
    except ET.ParseError as e:
        logger.error("Failed to parse Checkstyle XML: %s", e)

    logger.info("Checkstyle found %d findings", len(findings))
    return findings


def run_spotbugs(input_path: Path, timeout: int = DEFAULT_TIMEOUT) -> List[Dict[str, Any]]:
    """Run SpotBugs analysis (requires compiled .class files).

    Note: SpotBugs operates on bytecode. If input is .java source files,
    they are compiled to a temp directory first.
    """
    if not SPOTBUGS_BIN.exists():
        logger.error("SpotBugs not installed at %s", SPOTBUGS_BIN)
        return []

    # SpotBugs needs .class files — compile first if given .java
    compile_dir = TOOLS_DIR / "test" / "compiled"
    compile_dir.mkdir(parents=True, exist_ok=True)

    java_files = []
    if input_path.is_file() and input_path.suffix == ".java":
        java_files = [input_path]
    elif input_path.is_dir():
        java_files = list(input_path.rglob("*.java"))

    if java_files:
        logger.info("Compiling %d Java files for SpotBugs...", len(java_files))
        cmd_compile = [
            "javac", "-d", str(compile_dir),
            "-source", "17", "-target", "17",
            "-nowarn",
        ] + [str(f) for f in java_files]

        try:
            subprocess.run(cmd_compile, capture_output=True, text=True, timeout=timeout)
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.error("Java compilation failed: %s", e)
            return []

    output_file = TOOLS_DIR / "test" / "spotbugs_output.xml"

    cmd = [
        str(SPOTBUGS_BIN),
        "-xml:withMessages",
        "-output", str(output_file),
        str(compile_dir) if java_files else str(input_path),
    ]

    logger.info("Running SpotBugs on %s ...", input_path)
    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        logger.error("SpotBugs timed out after %ds", timeout)
        return []
    except FileNotFoundError:
        logger.error("SpotBugs binary not found")
        return []

    return _parse_spotbugs_xml(output_file)


def _parse_spotbugs_xml(xml_path: Path) -> List[Dict[str, Any]]:
    """Parse SpotBugs XML output to normalized findings."""
    if not xml_path.exists():
        return []

    findings = []
    try:
        tree = ET.parse(xml_path)
        for bug in tree.findall(".//BugInstance"):
            bug_type = bug.get("type", "Unknown")
            priority = bug.get("priority", "2")
            message = ""
            msg_elem = bug.find("ShortMessage")
            if msg_elem is not None and msg_elem.text:
                message = msg_elem.text

            source = bug.find(".//SourceLine")
            filename = source.get("sourcefile", "unknown") if source is not None else "unknown"
            line = source.get("start", "1") if source is not None else "1"

            findings.append(_normalize_finding(
                tool="SpotBugs",
                file=filename,
                line=line,
                rule=bug_type,
                severity_raw=priority,
                message=message,
                smell_map=SPOTBUGS_SMELL_MAP,
            ))
    except ET.ParseError as e:
        logger.error("Failed to parse SpotBugs XML: %s", e)

    logger.info("SpotBugs found %d findings", len(findings))
    return findings


def run_sonarqube(
    input_path: Path,
    server_url: str = "http://localhost:9000",
    token: Optional[str] = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> List[Dict[str, Any]]:
    """Run SonarQube analysis via sonar-scanner CLI.

    Requires SonarQube server running (use tools/baseline/start_sonarqube.sh).

    Args:
        input_path: Source directory to analyze.
        server_url: SonarQube server URL.
        token: Authentication token (default: initial admin token).
        timeout: Analysis timeout in seconds.

    Returns:
        List of normalized findings.
    """
    if not SONAR_SCANNER_BIN.exists():
        logger.error("SonarScanner not installed at %s", SONAR_SCANNER_BIN)
        return []

    project_key = "codesmell-benchmark"

    cmd = [
        str(SONAR_SCANNER_BIN),
        f"-Dsonar.projectKey={project_key}",
        f"-Dsonar.sources={input_path}",
        f"-Dsonar.host.url={server_url}",
        "-Dsonar.java.binaries=.",
        f"-Dsonar.projectBaseDir={input_path}",
    ]
    if token:
        cmd.append(f"-Dsonar.token={token}")

    logger.info("Running SonarQube scanner on %s ...", input_path)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout * 2)
        if result.returncode != 0:
            logger.warning("SonarScanner exit code: %d", result.returncode)
            logger.warning("SonarScanner stderr: %s", result.stderr[:500])
    except subprocess.TimeoutExpired:
        logger.error("SonarScanner timed out")
        return []
    except FileNotFoundError:
        logger.error("SonarScanner binary not found")
        return []

    return _fetch_sonarqube_issues(server_url, project_key)


def _fetch_sonarqube_issues(
    server_url: str,
    project_key: str,
) -> List[Dict[str, Any]]:
    """Fetch issues from SonarQube REST API after analysis."""
    try:
        import urllib.request

        api_url = f"{server_url}/api/issues/search?componentKeys={project_key}&types=CODE_SMELL&ps=500"
        req = urllib.request.Request(api_url)
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        logger.error("Failed to fetch SonarQube issues: %s", e)
        return []

    findings = []
    for issue in data.get("issues", []):
        rule = issue.get("rule", "Unknown")
        component = issue.get("component", "unknown")
        filename = component.rsplit(":", 1)[-1] if ":" in component else component

        findings.append(_normalize_finding(
            tool="SonarQube",
            file=filename,
            line=issue.get("line", 1),
            rule=rule,
            severity_raw=issue.get("severity", "MAJOR"),
            message=issue.get("message", ""),
            smell_map=SONARQUBE_SMELL_MAP,
        ))

    logger.info("SonarQube found %d findings", len(findings))
    return findings


def run_intellij(input_path: Path, timeout: int = DEFAULT_TIMEOUT) -> List[Dict[str, Any]]:
    """Run IntelliJ IDEA command-line inspections.

    Requires IntelliJ IDEA installed; path recorded by setup script.
    """
    if not INTELLIJ_PATH_FILE.exists():
        logger.warning("IntelliJ IDEA path not registered. Run setup_baseline_tools.sh.")
        return []

    idea_app = INTELLIJ_PATH_FILE.read_text().strip()
    inspect_bin = Path(idea_app) / "Contents" / "bin" / "inspect.sh"

    if not inspect_bin.exists():
        logger.error("IntelliJ inspect.sh not found at %s", inspect_bin)
        return []

    output_dir = TOOLS_DIR / "test" / "intellij_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(inspect_bin),
        str(input_path),              # Project path
        str(input_path),              # Inspection scope
        str(output_dir),              # Output directory
        "-v2",                        # Verbose
    ]

    logger.info("Running IntelliJ inspections on %s ...", input_path)
    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=timeout * 3)
    except subprocess.TimeoutExpired:
        logger.error("IntelliJ timed out after %ds", timeout * 3)
        return []
    except FileNotFoundError:
        logger.error("IntelliJ inspect.sh not found")
        return []

    return _parse_intellij_output(output_dir)


def _parse_intellij_output(output_dir: Path) -> List[Dict[str, Any]]:
    """Parse IntelliJ XML inspection results."""
    findings = []

    for xml_file in output_dir.glob("*.xml"):
        try:
            tree = ET.parse(xml_file)
            for problem in tree.findall(".//problem"):
                file_elem = problem.find("file")
                line_elem = problem.find("line")
                desc_elem = problem.find("description")
                severity_elem = problem.find("problem_class")

                filename = file_elem.text if file_elem is not None and file_elem.text else "unknown"
                line = line_elem.text if line_elem is not None and line_elem.text else "1"
                message = desc_elem.text if desc_elem is not None and desc_elem.text else ""
                severity = severity_elem.get("severity", "WARNING") if severity_elem is not None else "WARNING"
                rule = severity_elem.text if severity_elem is not None and severity_elem.text else "Unknown"

                findings.append(_normalize_finding(
                    tool="IntelliJ",
                    file=filename,
                    line=line,
                    rule=rule.replace(" ", ""),
                    severity_raw=severity,
                    message=message,
                    smell_map=INTELLIJ_SMELL_MAP,
                ))
        except ET.ParseError as e:
            logger.warning("Failed to parse IntelliJ output %s: %s", xml_file.name, e)

    logger.info("IntelliJ found %d findings", len(findings))
    return findings


# ---------------------------------------------------------------------------
# Python Tool Runners
# ---------------------------------------------------------------------------

def run_pylint(input_path: Path, timeout: int = DEFAULT_TIMEOUT) -> List[Dict[str, Any]]:
    """Run pylint analysis on Python files.

    Args:
        input_path: Python file or directory.
        timeout: Max seconds for pylint execution.

    Returns:
        List of normalized finding dicts.
    """
    try:
        import subprocess
        which_result = subprocess.run(["which", "pylint"], capture_output=True, text=True)
        if which_result.returncode != 0:
            logger.error("pylint not installed. Install with: pip install pylint")
            return []
    except Exception:
        logger.error("pylint not installed")
        return []

    output_file = TOOLS_DIR / "test" / "pylint_output.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "pylint", "--output-format=json",
        "--recursive=y" if input_path.is_dir() else "",
        str(input_path),
    ]
    cmd = [c for c in cmd if c]  # Remove empty strings

    logger.info("Running pylint on %s ...", input_path)
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
        )
        # pylint returns non-zero if issues found (not an error)
        findings_json = result.stdout
        if findings_json.strip():
            issues = json.loads(findings_json)
            findings = []
            for issue in issues:
                findings.append(_normalize_finding(
                    tool="pylint",
                    file=issue.get("path", "unknown"),
                    line=issue.get("line", 1),
                    rule=issue.get("symbol", "unknown"),
                    severity_raw=issue.get("type", "refactor"),
                    message=issue.get("message", ""),
                    smell_map=PYLINT_SMELL_MAP,
                ))
            logger.info("pylint found %d findings", len(findings))
            return findings
    except subprocess.TimeoutExpired:
        logger.error("pylint timed out after %ds", timeout)
        return []
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("Failed to parse pylint JSON: %s", e)
        return []

    return []


def run_flake8(input_path: Path, timeout: int = DEFAULT_TIMEOUT) -> List[Dict[str, Any]]:
    """Run flake8 linter on Python files.

    Args:
        input_path: Python file or directory.
        timeout: Max seconds for flake8 execution.

    Returns:
        List of normalized finding dicts.
    """
    try:
        import subprocess
        which_result = subprocess.run(["which", "flake8"], capture_output=True, text=True)
        if which_result.returncode != 0:
            logger.error("flake8 not installed. Install with: pip install flake8")
            return []
    except Exception:
        logger.error("flake8 not installed")
        return []

    cmd = ["flake8", "--format=%(path)s:%(row)d:%(col)d:%(code)s:%(text)s", str(input_path)]

    logger.info("Running flake8 on %s ...", input_path)
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
        )
        output = result.stdout
        if output.strip():
            findings = []
            for line in output.strip().splitlines():
                # Format: path:row:col:code:text
                parts = line.split(":", 4)
                if len(parts) >= 5:
                    findings.append(_normalize_finding(
                        tool="flake8",
                        file=parts[0],
                        line=int(parts[1]) if parts[1].isdigit() else 1,
                        rule=parts[3],
                        severity_raw="warning",  # flake8 doesn't have severity levels
                        message=parts[4],
                        smell_map=FLAKE8_SMELL_MAP,
                    ))
            logger.info("flake8 found %d findings", len(findings))
            return findings
    except subprocess.TimeoutExpired:
        logger.error("flake8 timed out after %ds", timeout)
        return []
    except (ValueError, IndexError) as e:
        logger.warning("Failed to parse flake8 output: %s", e)
        return []

    return []


# ---------------------------------------------------------------------------
# JavaScript Tool Runners
# ---------------------------------------------------------------------------

def run_eslint(input_path: Path, timeout: int = DEFAULT_TIMEOUT) -> List[Dict[str, Any]]:
    """Run ESLint on JavaScript files.

    Args:
        input_path: JavaScript file or directory.
        timeout: Max seconds for ESLint execution.

    Returns:
        List of normalized finding dicts.
    """
    try:
        import subprocess
        which_result = subprocess.run(["which", "eslint"], capture_output=True, text=True)
        if which_result.returncode != 0:
            logger.error("ESLint not installed. Install with: npm install -g eslint")
            return []
    except Exception:
        logger.error("ESLint not installed")
        return []

    cmd = ["eslint", "--format=json", str(input_path)]

    logger.info("Running ESLint on %s ...", input_path)
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
        )
        findings_json = result.stdout
        if findings_json.strip():
            results = json.loads(findings_json)
            findings = []
            for file_result in results:
                for message in file_result.get("messages", []):
                    findings.append(_normalize_finding(
                        tool="ESLint",
                        file=file_result.get("filePath", "unknown"),
                        line=message.get("line", 1),
                        rule=message.get("ruleId", "unknown"),
                        severity_raw=str(message.get("severity", 1)),  # 1=warning, 2=error
                        message=message.get("message", ""),
                        smell_map=ESLINT_SMELL_MAP,
                    ))
            logger.info("ESLint found %d findings", len(findings))
            return findings
    except subprocess.TimeoutExpired:
        logger.error("ESLint timed out after %ds", timeout)
        return []
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("Failed to parse ESLint JSON: %s", e)
        return []

    return []


# ---------------------------------------------------------------------------
# Tool Registry & Orchestration
# ---------------------------------------------------------------------------

# Tool Registry & Orchestration
# ---------------------------------------------------------------------------

TOOL_RUNNERS = {
    # Java tools
    "java": {
        "pmd": run_pmd,
        "checkstyle": run_checkstyle,
        "spotbugs": run_spotbugs,
        "sonarqube": run_sonarqube,
        "intellij": run_intellij,
    },
    # Python tools
    "python": {
        "pylint": run_pylint,
        "flake8": run_flake8,
    },
    # JavaScript tools
    "javascript": {
        "eslint": run_eslint,
    },
}


def run_tool(
    tool_name: str,
    input_path: Path,
    language: Optional[str] = None,
    output_dir: Optional[Path] = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> List[Dict[str, Any]]:
    """Run a single baseline tool and save normalized results.

    Args:
        tool_name: Tool identifier (pmd, checkstyle, pylint, eslint, etc.).
        input_path: Source file or directory.
        language: Programming language (java, python, javascript). Auto-detect if None.
        output_dir: Output directory for results JSON.
        timeout: Analysis timeout in seconds.

    Returns:
        List of normalized findings.
    """
    if language is None:
        language = detect_language(input_path)

    language = language.lower()
    tool_name = tool_name.lower()

    # Get language-specific tools
    lang_tools = TOOL_RUNNERS.get(language, {})
    runner = lang_tools.get(tool_name)

    if runner is None:
        available = list(lang_tools.keys()) if lang_tools else []
        logger.error("Unknown tool '%s' for %s. Available: %s", tool_name, language, available)
        return []

    start_time = datetime.now()
    findings = runner(input_path, timeout=timeout)
    elapsed = (datetime.now() - start_time).total_seconds()

    # Save results
    output_dir = output_dir or PREDICTIONS_DIR / "baseline"
    output_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "tool": tool_name,
        "language": language,
        "input": str(input_path),
        "timestamp": start_time.isoformat(),
        "elapsed_seconds": round(elapsed, 2),
        "total_findings": len(findings),
        "findings": findings,
    }

    output_path = output_dir / f"{language}_{tool_name}.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    logger.info("%s: %d findings in %.1fs → %s", tool_name, len(findings), elapsed, output_path)
    return findings


def run_all_tools(
    input_path: Path,
    language: Optional[str] = None,
    output_dir: Optional[Path] = None,
    tools: Optional[List[str]] = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> Dict[str, List[Dict[str, Any]]]:
    """Run all (or selected) baseline tools and save results.

    Args:
        input_path: Source file or directory.
        language: Programming language (java, python, javascript). Auto-detect if None.
        output_dir: Output directory.
        tools: List of tool names to run. None = all tools for detected language.
        timeout: Per-tool timeout.

    Returns:
        Dict mapping tool_name → list of findings.
    """
    if language is None:
        language = detect_language(input_path)

    language = language.lower()
    lang_tools = TOOL_RUNNERS.get(language, {})

    if not lang_tools:
        logger.error("No tools available for language: %s", language)
        return {}

    # Handle "all" keyword
    if tools is None or "all" in tools:
        tool_names = list(lang_tools.keys())
    else:
        tool_names = [t.lower() for t in tools]

    all_results = {}

    for name in tool_names:
        logger.info("=" * 60)
        logger.info("Running %s (%s)", name.upper(), language)
        logger.info("=" * 60)
        try:
            all_results[name] = run_tool(name, input_path, language, output_dir, timeout)
        except Exception as e:
            logger.error("%s failed: %s", name, e)
            all_results[name] = []

    # Summary
    print("\n" + "=" * 60)
    print("BASELINE SUMMARY")
    print("=" * 60)
    for name in tool_names:
        count = len(all_results.get(name, []))
        status = "✓" if count > 0 else "✗"
        print(f"  {status} {name:15s} → {count} findings")
    print("=" * 60)

    return all_results


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Run baseline static analysis tools for code smell detection benchmarking (Java, Python, JavaScript).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Java (auto-detected)
  python scripts/run_baseline_tools.py --input data/datasets/marv/ --tool all
  python scripts/run_baseline_tools.py --input sample.java --tool pmd checkstyle

  # Python (auto-detected or explicit)
  python scripts/run_baseline_tools.py --input sample.py --language python --tool all
  python scripts/run_baseline_tools.py --input src/ --language python --tool pylint

  # JavaScript (auto-detected or explicit)
  python scripts/run_baseline_tools.py --input app.js --language javascript --tool eslint
  python scripts/run_baseline_tools.py --input src/js/ --language javascript --tool all
        """,
    )
    parser.add_argument(
        "--input", "-i", required=True, type=Path,
        help="Path to source file or directory to analyze.",
    )
    parser.add_argument(
        "--language", "-l", choices=["java", "python", "javascript"],
        default=None,
        help="Programming language (auto-detect if not specified).",
    )
    parser.add_argument(
        "--tool", "-t", nargs="+", default=["all"],
        help="Tools to run (default: all available for language). Examples: pmd, pylint, eslint",
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=None,
        help=f"Output directory (default: {PREDICTIONS_DIR / 'baseline'}).",
    )
    parser.add_argument(
        "--timeout", type=int, default=DEFAULT_TIMEOUT,
        help=f"Per-tool timeout in seconds (default: {DEFAULT_TIMEOUT}).",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging.",
    )

    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Validate input
    if not args.input.exists():
        logger.error("Input path does not exist: %s", args.input)
        sys.exit(1)

    # Auto-detect language if not specified
    language = args.language or detect_language(args.input)
    logger.info("Detected language: %s", language)

    # Run tools
    tools = None if "all" in args.tool else args.tool
    run_all_tools(args.input, language=language, output_dir=args.output, tools=tools, timeout=args.timeout)


if __name__ == "__main__":
    main()
