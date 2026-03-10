#!/usr/bin/env bash
# ===========================================================================
# Baseline Analysis Runner — Java, Python, JavaScript
#
# Runs all applicable baseline static analysis tools on the given directory.
# Auto-detects language or uses explicit --language flag.
# Outputs normalized JSON to results/predictions/baseline/
#
# Usage:
#   bash scripts/baseline/run.sh --language java    --dir data/datasets/SmellyCodeDataset/Java
#   bash scripts/baseline/run.sh --language python   --dir data/datasets/SmellyCodeDataset/Python
#   bash scripts/baseline/run.sh --language javascript --dir data/datasets/SmellyCodeDataset/JavaScript
#   bash scripts/baseline/run.sh --dir data/datasets/SmellyCodeDataset   # auto-detect per sub-folder
#
# Or via Make:
#   make baseline-java DIR=data/datasets/SmellyCodeDataset/Java
#   make baseline-python DIR=data/datasets/SmellyCodeDataset/Python
#   make baseline-js DIR=data/datasets/SmellyCodeDataset/JavaScript
#   make baseline-run DIR=data/datasets/SmellyCodeDataset
# ===========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[✓]${NC} $1"; }
warn()    { echo -e "${YELLOW}[!]${NC} $1"; }
fail()    { echo -e "${RED}[✗]${NC} $1"; exit 1; }

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
LANGUAGE=""
INPUT_DIR=""
TOOLS=""
TIMEOUT=60
VERBOSE=""
DOCKER_MODE=false

usage() {
    echo ""
    echo "Usage: $0 --dir <directory> [--language java|python|javascript] [--tools pmd,checkstyle,...] [--timeout 60] [--docker] [--verbose]"
    echo ""
    echo "Options:"
    echo "  --dir, -d       Input source directory (required)"
    echo "  --language, -l  Language: java, python, javascript (auto-detect if omitted)"
    echo "  --tools, -t     Comma-separated tool list (default: all for language)"
    echo "  --timeout       Per-tool timeout in seconds (default: 60)"
    echo "  --docker        Use Docker mode for Java tools"
    echo "  --verbose, -v   Enable verbose output"
    echo ""
    echo "Language-specific tools:"
    echo "  Java:       pmd, checkstyle, spotbugs, sonarqube, intellij"
    echo "  Python:     pylint, flake8"
    echo "  JavaScript: eslint"
    echo ""
    echo "Examples:"
    echo "  $0 --dir data/datasets/SmellyCodeDataset/Java --language java"
    echo "  $0 --dir data/datasets/SmellyCodeDataset --language python --tools pylint"
    echo "  $0 --dir data/datasets/SmellyCodeDataset/JavaScript --docker"
    echo ""
    exit 1
}

while [ $# -gt 0 ]; do
    case "$1" in
        --dir|-d)       INPUT_DIR="$2"; shift 2 ;;
        --language|-l)  LANGUAGE="$2"; shift 2 ;;
        --tools|-t)     TOOLS="$2"; shift 2 ;;
        --timeout)      TIMEOUT="$2"; shift 2 ;;
        --docker)       DOCKER_MODE=true; shift ;;
        --verbose|-v)   VERBOSE="--verbose"; shift ;;
        --help|-h)      usage ;;
        *)              fail "Unknown option: $1. Use --help for usage." ;;
    esac
done

if [ -z "$INPUT_DIR" ]; then
    fail "Missing required --dir argument. Use --help for usage."
fi

# Resolve input directory
if [[ "$INPUT_DIR" != /* ]]; then
    INPUT_DIR="$PROJECT_ROOT/$INPUT_DIR"
fi

if [ ! -d "$INPUT_DIR" ]; then
    fail "Directory not found: $INPUT_DIR"
fi

# ---------------------------------------------------------------------------
# Auto-detect language
# ---------------------------------------------------------------------------
detect_language() {
    local dir="$1"
    local java_count python_count js_count

    java_count=$(find "$dir" -name "*.java" -type f 2>/dev/null | wc -l | tr -d ' ')
    python_count=$(find "$dir" -name "*.py" -type f 2>/dev/null | wc -l | tr -d ' ')
    js_count=$(find "$dir" \( -name "*.js" -o -name "*.jsx" -o -name "*.ts" -o -name "*.tsx" \) -type f 2>/dev/null | wc -l | tr -d ' ')

    if [ "$java_count" -gt "$python_count" ] && [ "$java_count" -gt "$js_count" ] && [ "$java_count" -gt 0 ]; then
        echo "java"
    elif [ "$python_count" -gt "$js_count" ] && [ "$python_count" -gt 0 ]; then
        echo "python"
    elif [ "$js_count" -gt 0 ]; then
        echo "javascript"
    else
        echo ""
    fi
}

if [ -z "$LANGUAGE" ]; then
    LANGUAGE=$(detect_language "$INPUT_DIR")
    if [ -z "$LANGUAGE" ]; then
        fail "Could not auto-detect language in $INPUT_DIR. Use --language flag."
    fi
    info "Auto-detected language: $LANGUAGE"
fi

# ---------------------------------------------------------------------------
# Run analysis
# ---------------------------------------------------------------------------
OUTPUT_DIR="$PROJECT_ROOT/results/predictions/baseline"
mkdir -p "$OUTPUT_DIR"

LANGUAGE_UPPER=$(echo "$LANGUAGE" | tr '[:lower:]' '[:upper:]')

echo ""
echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  ${BOLD}Baseline Analysis — ${LANGUAGE_UPPER}${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
info "Input:    $INPUT_DIR"
info "Language: $LANGUAGE"
info "Output:   $OUTPUT_DIR"
info "Timeout:  ${TIMEOUT}s per tool"
echo ""

# Build tool arguments
TOOL_ARGS=""
if [ -n "$TOOLS" ]; then
    # Convert comma-separated to space-separated
    TOOL_ARGS="--tool $(echo "$TOOLS" | tr ',' ' ')"
else
    TOOL_ARGS="--tool all"
fi

if $DOCKER_MODE; then
    # Docker mode: use the Docker-based runner
    info "Using Docker mode for analysis..."
    REL_INPUT=$(python3 -c "import os; print(os.path.relpath('$INPUT_DIR', '$PROJECT_ROOT'))" 2>/dev/null || echo "$INPUT_DIR")
    bash "$SCRIPT_DIR/run_tools_docker.sh" "$LANGUAGE" "$REL_INPUT"
else
    # Python mode: use the Python-based runner (more flexible)
    cd "$PROJECT_ROOT"

    # Activate venv if it exists
    if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
        source "$PROJECT_ROOT/.venv/bin/activate"
    elif [ -f "$PROJECT_ROOT/venv/bin/activate" ]; then
        source "$PROJECT_ROOT/venv/bin/activate"
    fi

    python3 "$SCRIPT_DIR/run_tools.py" \
        --input "$INPUT_DIR" \
        --language "$LANGUAGE" \
        $TOOL_ARGS \
        --output "$OUTPUT_DIR" \
        --timeout "$TIMEOUT" \
        $VERBOSE
fi

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  ✅ Baseline Analysis Complete                                 ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
info "Results saved to: $OUTPUT_DIR/"
echo ""

# List generated files
if ls "$OUTPUT_DIR"/${LANGUAGE}_*.json &>/dev/null 2>&1; then
    info "Generated files:"
    for f in "$OUTPUT_DIR"/${LANGUAGE}_*.json; do
        local_name=$(basename "$f")
        local_size=$(wc -c < "$f" | tr -d ' ')
        echo "  - $local_name (${local_size} bytes)"
    done
fi

echo ""
info "Next: make baseline-report   (generate summary tables & visualizations)"
echo ""
