# Baseline Tools Setup & Running Guide

> Complete guide to installing, configuring, and running all baseline static analysis tools for the Code Smell Detection project.

---

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
  - [One-Command Install](#one-command-install)
  - [Verify Installation](#verify-installation)
  - [What Gets Installed](#what-gets-installed)
- [Running Baseline Analysis](#running-baseline-analysis)
  - [Java Analysis](#java-analysis)
  - [Python Analysis](#python-analysis)
  - [JavaScript Analysis](#javascript-analysis)
  - [Auto-Detect Language](#auto-detect-language)
  - [Advanced Options](#advanced-options)
- [Generating Reports](#generating-reports)
- [Output Format](#output-format)
- [Tools Reference](#tools-reference)
  - [Java Tools](#java-tools)
  - [Python Tools](#python-tools)
  - [JavaScript Tools](#javascript-tools)
- [Directory Structure](#directory-structure)
- [Docker Mode](#docker-mode)
- [Troubleshooting](#troubleshooting)

---

## Overview

The baseline toolkit runs **traditional static analysis tools** against your codebase and produces **normalized JSON output**. These results serve as the ground truth comparison for the LLM-based code smell detection experiments.

**Supported languages:** Java, Python, JavaScript

**Quick summary of commands:**

| Command | Description |
|---------|-------------|
| `make baseline-install` | Install everything (Docker, tools, SonarQube) |
| `make baseline-install-verify` | Check what's already installed |
| `make baseline-java DIR=path` | Run Java tools (PMD, Checkstyle, SpotBugs, SonarQube) |
| `make baseline-python DIR=path` | Run Python tools (pylint, flake8) |
| `make baseline-js DIR=path` | Run JavaScript tools (ESLint) |
| `make baseline-run DIR=path` | Auto-detect language and run all tools |
| `make baseline-report` | Generate summary tables and visualizations |

---

## Prerequisites

Before installing the baseline tools, ensure you have:

| Requirement | Version | Required For | Check |
|-------------|---------|-------------|-------|
| **Docker Desktop** | Any recent | Java tools (Docker mode), SonarQube | `docker --version` |
| **Java (JDK)** | 17+ | PMD, Checkstyle, SpotBugs | `java -version` |
| **Python** | 3.10+ | pylint, flake8, runner scripts | `python3 --version` |
| **Node.js / npm** | 16+ | ESLint (JavaScript) | `node --version` |
| **Homebrew** (macOS) | — | Auto-install missing tools | `brew --version` |

> **Note:** You don't need all prerequisites. If you only analyze Java code, you don't need Node.js. The installer will skip unavailable components gracefully.

---

## Installation

### One-Command Install

Run the full installation with a single command:

```bash
make baseline-install
```

This runs a **13-step installation** that handles everything:

| Step | What It Does |
|------|-------------|
| 1 | Check Docker Desktop is running |
| 2 | Check Docker Compose is available |
| 3 | Check/install Java 17+ (via Homebrew on macOS) |
| 4 | Check Node.js / npm |
| 5 | Check/install Python tools (`pylint`, `flake8`) |
| 6 | Build `Dockerfile.baseline` Docker image |
| 7 | Start Docker Compose services (SonarQube + baseline-tools) |
| 8 | Wait for SonarQube to become healthy (up to 120s) |
| 9 | Configure SonarQube (auth token + default project) |
| 10 | Install local tools (PMD 7.0.0, Checkstyle 10.14.0, SpotBugs 4.8.3, SonarScanner 5.0.1) |
| 11 | Install ESLint globally |
| 12 | Verify tools work with a sample Java file |
| 13 | Create `results/` directory structure |

At the end you'll see a summary:

```
  Passed:  11 / 13
  Failed:  0 / 13
  Skipped: 2 / 13
```

- **Passed** = tool is installed and working
- **Skipped** = optional component not available (e.g., Node.js not installed)
- **Failed** = something went wrong (see error message)

### Verify Installation

To check what's already installed **without making any changes**:

```bash
make baseline-install-verify
```

This runs in read-only mode — it checks tool availability but doesn't install or start anything.

### What Gets Installed

After a full install, these tools will be available:

| Tool | Location | Version |
|------|----------|---------|
| PMD | `tools/baseline/pmd-bin-7.0.0/` | 7.0.0 |
| Checkstyle | `tools/baseline/checkstyle-10.14.0-all.jar` | 10.14.0 |
| SpotBugs | `tools/baseline/spotbugs-4.8.3/` | 4.8.3 |
| SonarScanner | `tools/baseline/sonar-scanner-5.0.1/` | 5.0.1 |
| SonarQube | Docker container on `localhost:9000` | 10.4 Community |
| pylint | System/venv Python package | Latest |
| flake8 | System/venv Python package | Latest |
| ESLint | Global npm package | Latest |

---

## Running Baseline Analysis

### Java Analysis

Run all Java static analysis tools (PMD, Checkstyle, SpotBugs, SonarQube):

```bash
make baseline-java DIR=data/datasets/SmellyCodeDataset
```

To run specific tools only:

```bash
make baseline-java DIR=data/datasets/SmellyCodeDataset TOOLS=pmd,checkstyle
```

### Python Analysis

Run Python linting tools (pylint, flake8):

```bash
make baseline-python DIR=data/datasets/SmellyCodeDataset
```

Run only pylint:

```bash
make baseline-python DIR=data/datasets/SmellyCodeDataset TOOLS=pylint
```

### JavaScript Analysis

Run JavaScript linting (ESLint):

```bash
make baseline-js DIR=data/datasets/SmellyCodeDataset
```

### Auto-Detect Language

If your directory has mixed files or you don't want to specify the language:

```bash
make baseline-run DIR=data/datasets/SmellyCodeDataset
```

The runner counts file extensions (`.java`, `.py`, `.js`/`.ts`) and picks the dominant language.

### Advanced Options

All baseline commands accept these options:

| Option | Default | Description |
|--------|---------|-------------|
| `DIR=path` | `data/datasets/SmellyCodeDataset` | Source code directory to analyze |
| `TIMEOUT=seconds` | `60` | Per-tool timeout in seconds |
| `TOOLS=tool1,tool2` | all tools for language | Comma-separated list of specific tools |
| `VERBOSE=1` | off | Enable detailed output |

**Examples:**

```bash
# Java with 120s timeout and verbose output
make baseline-java DIR=data/datasets/marv TIMEOUT=120 VERBOSE=1

# Only PMD and SpotBugs
make baseline-java DIR=data/datasets/marv TOOLS=pmd,spotbugs

# Python with verbose
make baseline-python DIR=src/ VERBOSE=1
```

---

## Generating Reports

After running baseline analysis, generate a summary report:

```bash
make baseline-report
```

This creates:
- **Visualizations:** `results/figures/baseline_*.png`
- **CSV exports:** `results/exports/baseline_*.csv`
- **LaTeX table:** `results/tables/baseline_summary.tex`
- **Text report:** `results/reports/baseline_report.txt`
- **Markdown summary:** `results/reports/BASELINE_REPORT.md`

---

## Output Format

All tools produce **normalized JSON** output saved in `results/predictions/baseline/`. Each finding follows this schema:

```json
{
  "tool": "pmd",
  "file": "src/MyClass.java",
  "line": 42,
  "smell_type": "Long Method",
  "severity": "major",
  "confidence": 0.9,
  "explanation": "Method exceeds 50 lines",
  "language": "java"
}
```

### Unified Smell Types

All tool-specific detections are mapped to **17 unified code smell types**:

| # | Smell Type | Detected By |
|---|-----------|------------|
| 1 | Long Method | PMD, pylint, SonarQube |
| 2 | God Class | PMD, SonarQube |
| 3 | Large Class | PMD, Checkstyle, SonarQube |
| 4 | Long Parameter List | PMD, Checkstyle, pylint |
| 5 | Feature Envy | SonarQube, IntelliJ |
| 6 | Inappropriate Intimacy | SonarQube |
| 7 | Message Chains | PMD, SonarQube |
| 8 | Divergent Change | SonarQube |
| 9 | Shotgun Surgery | SonarQube |
| 10 | Duplicate Code | PMD, SonarQube |
| 11 | Lazy Class | SonarQube |
| 12 | Data Class | PMD, SonarQube |
| 13 | Dead Code | PMD, pylint, ESLint |
| 14 | Switch Statements | PMD, Checkstyle |
| 15 | Refused Bequest | SonarQube |
| 16 | Speculative Generality | SonarQube |
| 17 | Parallel Inheritance Hierarchies | SonarQube |

---

## Tools Reference

### Java Tools

#### PMD 7.0.0
- **Type:** Rule-based static analysis
- **Detects:** Long methods, large classes, duplicate code, dead code, god classes
- **Config:** Default PMD ruleset (Java best practices + design + error-prone)
- **Location:** `tools/baseline/pmd-bin-7.0.0/`

#### Checkstyle 10.14.0
- **Type:** Style and naming convention checker
- **Detects:** Long parameter lists, large classes, naming violations, code structure issues
- **Config:** Sun/Google checks
- **Location:** `tools/baseline/checkstyle-10.14.0-all.jar`

#### SpotBugs 4.8.3
- **Type:** Bytecode-level bug finder
- **Detects:** Performance issues, bad practices, correctness issues, security vulnerabilities
- **Requires:** Compiled `.class` files (the runner attempts compilation automatically)
- **Location:** `tools/baseline/spotbugs-4.8.3/`

#### SonarQube 10.4 Community Edition
- **Type:** Comprehensive multi-rule platform
- **Detects:** Code smells, bugs, vulnerabilities, security hotspots, duplications
- **Access:** `http://localhost:9000` (default credentials: `admin` / `admin`)
- **Runs in:** Docker container via `docker-compose.yml`

### Python Tools

#### pylint
- **Type:** Comprehensive Python linter
- **Detects:** Long methods, too many arguments, duplicate code, unused variables, refactoring suggestions
- **Install:** `pip install pylint`

#### flake8
- **Type:** Style guide enforcement / PEP 8 checker
- **Detects:** Style violations, complexity issues, unused imports
- **Install:** `pip install flake8`

### JavaScript Tools

#### ESLint
- **Type:** Pluggable JavaScript/TypeScript linter
- **Detects:** Dead code, complexity issues, style violations, best practice violations
- **Config:** Uses project's `eslint.config.js` if present
- **Install:** `npm install -g eslint`

---

## Directory Structure

### Scripts (`scripts/baseline/`)

| File | Purpose |
|------|---------|
| `install.sh` | Complete 13-step installation and verification |
| `run.sh` | Unified runner with language auto-detection |
| `run_tools.py` | Python-based multi-language tool runner (core engine) |
| `run_tools_docker.sh` | Docker-based runner (alternative for Java tools) |
| `setup.sh` | Docker + SonarQube initialization |
| `install_tools.sh` | Download PMD, Checkstyle, SpotBugs, SonarScanner |
| `analyze.sh` | Run all tools pipeline (batch mode) |
| `normalize_output.py` | Normalize tool output to unified JSON format |
| `generate_report.py` | Generate tables, heatmaps, CSV, LaTeX reports |
| `sonarqube_setup.sh` | SonarQube token and project configuration |
| `sonarqube_analyze.py` | SonarQube analysis runner |
| `sonarqube_compile_and_analyze.sh` | Compile Java + SonarQube scan |
| `sonarqube_quick_scan.sh` | Quick SonarQube scan |
| `sonarqube_run.sh` | Start/stop SonarQube |

### Results (`results/`)

| Directory | Content |
|-----------|---------|
| `predictions/baseline/` | Baseline tool predictions (JSON) |
| `predictions/llm_vanilla/` | LLM predictions without RAG |
| `predictions/llm_rag/` | LLM predictions with RAG |
| `confusion_matrices/` | Confusion matrix data |
| `performance/` | Timing/resource measurements |
| `resources/` | Resource usage data |
| `figures/` | Generated visualizations (PNG) |
| `tables/` | LaTeX tables |
| `reports/` | Text and markdown reports |
| `metrics/` | Computed metrics (F1, precision, recall) |
| `logs/` | Run logs (JSON) |
| `exports/` | CSV/JSON data exports |

---

## Docker Mode

For Java analysis, you can use Docker instead of local tool installations:

```bash
# Run with Docker mode
bash scripts/baseline/run.sh --dir data/datasets/SmellyCodeDataset --language java --docker
```

The Docker image (`Dockerfile.baseline`) includes:
- Eclipse Temurin JDK 17
- PMD 7.0.0
- Checkstyle 10.14.0
- SpotBugs 4.8.3

This is useful when you don't want to install Java tools locally.

### Docker Compose Services

```bash
# Start all services
docker compose up -d

# Check status
docker compose ps

# View SonarQube logs
docker compose logs sonarqube

# Stop services
docker compose down
```

| Service | Port | Description |
|---------|------|-------------|
| `app` | — | Main application container |
| `baseline-tools` | — | Java analysis tools container |
| `sonarqube` | 9000 | SonarQube web UI and API |
| `sonarqube-db` | 5432 | PostgreSQL database for SonarQube |

---

## Troubleshooting

### Docker not running

```
[✗] Docker daemon not running
```

**Fix:** Open Docker Desktop and wait for it to start. Then re-run `make baseline-install`.

### Java not found or version too low

```
[✗] Java 17+ required
```

**Fix (macOS):**
```bash
brew install --cask temurin@17
```

**Fix (Ubuntu):**
```bash
sudo apt install openjdk-17-jdk
```

### SonarQube won't start

```
[✗] SonarQube did not become ready within 120s
```

**Fix:** SonarQube needs 2-4 GB RAM and takes time on first boot. Check Docker logs:
```bash
docker compose logs sonarqube
```

Common causes:
- Not enough memory allocated to Docker (increase in Docker Desktop → Settings → Resources)
- Port 9000 already in use (`lsof -i :9000`)

### PMD / Checkstyle / SpotBugs not found

```
[!] Local tools installation failed
```

**Fix:** Manually run the tool installer:
```bash
bash scripts/baseline/install_tools.sh
```

Check the `tools/baseline/` directory for downloaded files.

### pylint / flake8 not found

**Fix:**
```bash
pip install pylint flake8
```

Or if using a virtual environment:
```bash
source .venv/bin/activate
pip install pylint flake8
```

### ESLint not found

**Fix:**
```bash
npm install -g eslint
```

### No files found / wrong language detected

The auto-detect counts file extensions. If your directory is empty or has unexpected files:

```bash
# Specify language explicitly
make baseline-java DIR=path/to/java/code
```

### Permission denied on scripts

```bash
chmod +x scripts/baseline/*.sh
```

---

## Workflow Summary

The typical end-to-end workflow:

```
1. make baseline-install          # Install everything (once)
2. make baseline-install-verify   # Confirm tools are ready
3. make baseline-java DIR=...     # Run Java analysis
4. make baseline-python DIR=...   # Run Python analysis
5. make baseline-js DIR=...       # Run JavaScript analysis
6. make baseline-report           # Generate reports
```

Results are saved in `results/predictions/baseline/` and reports in `results/reports/`.
