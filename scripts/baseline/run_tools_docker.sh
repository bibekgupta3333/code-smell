#!/bin/bash
# Simplified Docker-based Baseline Tools Runner
# Executes static analysis tools on code directly via Docker

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORK_DIR="/workspace"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[✓]${NC} $1"; }
warn()    { echo -e "${YELLOW}[!]${NC} $1"; }
fail()    { echo -e "${RED}[✗]${NC} $1"; }

# Parse arguments
LANGUAGE="${1:-}"
INPUT_DIR="${2:-}"

if [ -z "$LANGUAGE" ] || [ -z "$INPUT_DIR" ]; then
    echo ""
    echo "Usage: bash scripts/run_baseline_tools_docker.sh <language> <input-directory>"
    echo ""
    echo "Examples:"
    echo "  bash scripts/baseline/run_tools_docker.sh java data/datasets/SmellyCodeDataset/Java"
    echo "  bash scripts/baseline/run_tools_docker.sh python data/datasets/SmellyCodeDataset/Python"
    echo "  bash scripts/baseline/run_tools_docker.sh javascript data/datasets/SmellyCodeDataset/JavaScript"
    echo ""
    exit 1
fi

# Validate language
case "$LANGUAGE" in
    java|python|javascript) ;;
    *)
        fail "Invalid language: $LANGUAGE"
        exit 1
        ;;
esac

# Validate input
INPUT_DIR="${PROJECT_ROOT}/${INPUT_DIR}"
if [ ! -d "$INPUT_DIR" ]; then
    fail "Input directory not found: $INPUT_DIR"
    exit 1
fi

# Create output directory
OUTPUT_DIR="${PROJECT_ROOT}/results/predictions/baseline"
mkdir -p "$OUTPUT_DIR"

info "============================================================"
info "Running baseline tools (Docker mode)"
info "============================================================"
info "Language: $LANGUAGE"
info "Input: $INPUT_DIR"
info "Output: $OUTPUT_DIR"

# Docker image name
IMAGE="code-smell-baseline:latest"

# Check if image exists
if ! docker image inspect "$IMAGE" &>/dev/null; then
    info "Building $IMAGE..."
    docker build -f "$PROJECT_ROOT/Dockerfile.baseline" -t "$IMAGE" "$PROJECT_ROOT" > /dev/null ||
        { fail "Failed to build Docker image"; exit 1; }
    success "Image built"
fi

case "$LANGUAGE" in
    java)
        info "PMD analysis..."
        docker run --rm \
            -v "$INPUT_DIR:/input:ro" \
            -v "$OUTPUT_DIR:/output:rw" \
            "$IMAGE" \
            /opt/pmd/bin/pmd check -d /input -R rulesets/java/quickstart.xml -f json -r /output/java_pmd.json 2>/dev/null && \
            success "PMD: $(grep -o '"rule"' $OUTPUT_DIR/java_pmd.json 2>/dev/null | wc -l) violations detected" || \
            warn "PMD analysis failed"

        info "Checkstyle analysis..."
        docker run --rm \
            -v "$INPUT_DIR:/input:ro" \
            -v "$OUTPUT_DIR:/output:rw" \
            "$IMAGE" \
            bash -c "java -jar /opt/checkstyle.jar -c /google_checks.xml -f xml /input > /output/java_checkstyle.xml 2>&1" && \
            success "Checkstyle completed" || \
            warn "Checkstyle analysis failed"
        ;;

    python)
        info "Pylint analysis..."
        docker run --rm \
            -v "$INPUT_DIR:/input:ro" \
            -v "$OUTPUT_DIR:/output:rw" \
            "$IMAGE" \
            pylint --output-format=json /input > "$OUTPUT_DIR/pylint.json" 2>/dev/null && \
            success "Pylint: $(jq 'length' $OUTPUT_DIR/pylint.json 2>/dev/null || echo "0") findings" || \
            warn "Pylint analysis failed"
        ;;

    javascript)
        info "ESLint analysis..."
        docker run --rm \
            -v "$INPUT_DIR:/input:ro" \
            -v "$OUTPUT_DIR:/output:rw" \
            -v "$PROJECT_ROOT/.eslintrc.js:/workspace/.eslintrc.js:ro" \
            "$IMAGE" \
            eslint /input --format=json --output-file=/output/eslint.json 2>/dev/null || \
            warn "ESLint analysis completed (may have issues)"
        success "ESLint completed"
        ;;
esac

info "============================================================"
success "Baseline analysis completed!"
info "Results saved to: $OUTPUT_DIR/"
echo ""
info "To generate a report, run:"
echo "  python scripts/baseline/generate_report.py"
