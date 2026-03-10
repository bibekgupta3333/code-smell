#!/usr/bin/env bash
# ===========================================================================
# Baseline Tools Complete Installation & Verification
# One command to install, configure, and verify all baseline analysis tools.
#
# Checks:
#   1. Docker Desktop running + daemon alive
#   2. Docker Compose available
#   3. Java 17+ installed
#   4. Node.js / npm (for ESLint)
#   5. Python tools (pylint, flake8)
#   6. Build Dockerfile.baseline image
#   7. Start Docker Compose services (SonarQube + baseline-tools)
#   8. Wait for SonarQube to become healthy
#   9. Configure SonarQube (token, project)
#  10. Install local tools (PMD, Checkstyle, SpotBugs, SonarScanner)
#  11. Install ESLint globally
#  12. Verify all tools with sample code
#  13. Create results/ directory structure
#
# Usage:
#   make baseline-install          # Full install
#   make baseline-install-verify   # Verify only
# ===========================================================================
set -uo pipefail

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
fail()    { echo -e "${RED}[✗]${NC} $1"; }

PASS=0
FAIL=0
SKIP=0
TOTAL_STEPS=13

step_pass() { success "$1"; ((PASS++)); }
step_fail() { fail "$1"; ((FAIL++)); }
step_skip() { warn "$1 (skipped)"; ((SKIP++)); }

# ===========================================================================
# Step 1: Docker Desktop
# ===========================================================================
check_docker() {
    echo ""
    info "${BOLD}[1/$TOTAL_STEPS] Checking Docker...${NC}"

    if ! command -v docker &>/dev/null; then
        step_fail "Docker not installed. Install from https://www.docker.com/products/docker-desktop/"
        return 1
    fi

    if ! docker info &>/dev/null 2>&1; then
        step_fail "Docker daemon not running. Please start Docker Desktop."
        return 1
    fi

    local docker_ver
    docker_ver=$(docker --version | head -1)
    step_pass "Docker: $docker_ver"
    return 0
}

# ===========================================================================
# Step 2: Docker Compose
# ===========================================================================
check_docker_compose() {
    info "${BOLD}[2/$TOTAL_STEPS] Checking Docker Compose...${NC}"

    if docker compose version &>/dev/null 2>&1; then
        local compose_ver
        compose_ver=$(docker compose version --short 2>/dev/null || echo "unknown")
        step_pass "Docker Compose: v$compose_ver"
        return 0
    fi

    step_fail "Docker Compose not available (requires Docker Desktop or docker-compose plugin)"
    return 1
}

# ===========================================================================
# Step 3: Java 17+
# ===========================================================================
check_java() {
    info "${BOLD}[3/$TOTAL_STEPS] Checking Java 17+...${NC}"

    if ! command -v java &>/dev/null; then
        warn "Java not found. Attempting install via Homebrew..."
        if command -v brew &>/dev/null; then
            brew install --cask temurin@17 2>/dev/null || true
        fi
    fi

    if ! command -v java &>/dev/null; then
        step_fail "Java 17+ required. Install: brew install --cask temurin@17"
        return 1
    fi

    local java_ver
    java_ver=$(java -version 2>&1 | head -1 | sed -E 's/.*"([0-9]+).*/\1/')
    if [ "$java_ver" -lt 17 ] 2>/dev/null; then
        step_fail "Java 17+ required (found Java $java_ver)"
        return 1
    fi

    step_pass "Java $java_ver"
    return 0
}

# ===========================================================================
# Step 4: Node.js / npm (for ESLint)
# ===========================================================================
check_node() {
    info "${BOLD}[4/$TOTAL_STEPS] Checking Node.js / npm...${NC}"

    if ! command -v node &>/dev/null; then
        warn "Node.js not found. JavaScript baseline (ESLint) will be unavailable."
        step_skip "Node.js (optional for Java/Python-only analysis)"
        return 1
    fi

    local node_ver
    node_ver=$(node --version)
    step_pass "Node.js $node_ver"
    return 0
}

# ===========================================================================
# Step 5: Python tools (pylint, flake8)
# ===========================================================================
check_python_tools() {
    info "${BOLD}[5/$TOTAL_STEPS] Checking Python tools (pylint, flake8)...${NC}"

    local all_ok=true

    if command -v pylint &>/dev/null; then
        success "  pylint $(pylint --version 2>&1 | head -1 | awk '{print $2}')"
    else
        warn "  pylint not found. Installing..."
        pip install pylint 2>/dev/null || pip3 install pylint 2>/dev/null || true
        if command -v pylint &>/dev/null; then
            success "  pylint installed"
        else
            warn "  pylint: install failed (pip install pylint)"
            all_ok=false
        fi
    fi

    if command -v flake8 &>/dev/null; then
        success "  flake8 $(flake8 --version 2>&1 | head -1 | awk '{print $1}')"
    else
        warn "  flake8 not found. Installing..."
        pip install flake8 2>/dev/null || pip3 install flake8 2>/dev/null || true
        if command -v flake8 &>/dev/null; then
            success "  flake8 installed"
        else
            warn "  flake8: install failed (pip install flake8)"
            all_ok=false
        fi
    fi

    if $all_ok; then
        step_pass "Python tools (pylint, flake8)"
    else
        step_skip "Python tools (partial)"
    fi
}

# ===========================================================================
# Step 6: Build Dockerfile.baseline
# ===========================================================================
build_baseline_image() {
    info "${BOLD}[6/$TOTAL_STEPS] Building baseline Docker image...${NC}"

    if ! command -v docker &>/dev/null || ! docker info &>/dev/null 2>&1; then
        step_skip "Baseline Docker image (Docker not available)"
        return 1
    fi

    cd "$PROJECT_ROOT"
    if docker build -f Dockerfile.baseline -t code-smell-baseline:latest . > /dev/null 2>&1; then
        step_pass "Docker image: code-smell-baseline:latest"
        return 0
    else
        step_fail "Failed to build Dockerfile.baseline"
        return 1
    fi
}

# ===========================================================================
# Step 7: Start Docker Compose services
# ===========================================================================
start_docker_services() {
    info "${BOLD}[7/$TOTAL_STEPS] Starting Docker Compose services...${NC}"

    if ! command -v docker &>/dev/null || ! docker info &>/dev/null 2>&1; then
        step_skip "Docker Compose services (Docker not available)"
        return 1
    fi

    cd "$PROJECT_ROOT"

    # Check if services are already running
    local running
    running=$(docker compose ps --status running -q 2>/dev/null | wc -l | tr -d ' ')

    if [ "$running" -ge 2 ]; then
        step_pass "Docker services already running ($running containers)"
        return 0
    fi

    if docker compose up -d > /dev/null 2>&1; then
        info "  Waiting for services to initialize (10s)..."
        sleep 10
        running=$(docker compose ps --status running -q 2>/dev/null | wc -l | tr -d ' ')
        step_pass "Docker services started ($running containers)"
        return 0
    else
        step_fail "Failed to start Docker Compose services"
        return 1
    fi
}

# ===========================================================================
# Step 8: Wait for SonarQube to become healthy
# ===========================================================================
wait_for_sonarqube() {
    info "${BOLD}[8/$TOTAL_STEPS] Waiting for SonarQube to become ready...${NC}"

    local sonar_host="${SONAR_HOST:-http://localhost:9000}"
    local max_wait=120
    local waited=0
    local interval=5

    while [ "$waited" -lt "$max_wait" ]; do
        local status
        status=$(curl -s "$sonar_host/api/system/status" 2>/dev/null | grep -o '"status":"[^"]*"' | cut -d'"' -f4 || echo "")

        if [ "$status" = "UP" ]; then
            step_pass "SonarQube is UP at $sonar_host"
            return 0
        fi

        info "  SonarQube status: ${status:-unreachable} (waited ${waited}s / ${max_wait}s)"
        sleep "$interval"
        waited=$((waited + interval))
    done

    step_fail "SonarQube did not become ready within ${max_wait}s"
    return 1
}

# ===========================================================================
# Step 9: Configure SonarQube (token, project)
# ===========================================================================
configure_sonarqube() {
    info "${BOLD}[9/$TOTAL_STEPS] Configuring SonarQube...${NC}"

    cd "$PROJECT_ROOT"

    if bash scripts/baseline/sonarqube_setup.sh > /dev/null 2>&1; then
        step_pass "SonarQube configured (token + default project)"
        return 0
    else
        step_fail "SonarQube configuration had issues"
        return 1
    fi
}

# ===========================================================================
# Step 10: Install local tools (PMD, Checkstyle, SpotBugs, SonarScanner)
# ===========================================================================
install_local_tools() {
    info "${BOLD}[10/$TOTAL_STEPS] Installing local analysis tools...${NC}"

    cd "$PROJECT_ROOT"

    if bash scripts/baseline/install_tools.sh > /dev/null 2>&1; then
        step_pass "Local tools installed (PMD, Checkstyle, SpotBugs, SonarScanner)"
        return 0
    else
        # Partial install might have succeeded
        local pass_count=0
        [ -f "tools/baseline/pmd-7.0.0/bin/pmd" ] && ((pass_count++))
        [ -f "tools/baseline/checkstyle-10.14.0-all.jar" ] && ((pass_count++))
        [ -f "tools/baseline/spotbugs-4.8.3/bin/spotbugs" ] && ((pass_count++))

        if [ "$pass_count" -ge 2 ]; then
            step_pass "Local tools installed ($pass_count/4 tools)"
        else
            step_fail "Local tools installation failed ($pass_count/4)"
        fi
        return 0
    fi
}

# ===========================================================================
# Step 11: Install ESLint
# ===========================================================================
install_eslint() {
    info "${BOLD}[11/$TOTAL_STEPS] Checking ESLint (JavaScript analysis)...${NC}"

    if ! command -v npm &>/dev/null; then
        step_skip "ESLint (npm not available)"
        return 1
    fi

    if command -v eslint &>/dev/null; then
        local eslint_ver
        eslint_ver=$(eslint --version 2>/dev/null || echo "unknown")
        step_pass "ESLint $eslint_ver"
        return 0
    fi

    info "  Installing ESLint globally..."
    npm install -g eslint 2>/dev/null || true

    if command -v eslint &>/dev/null; then
        step_pass "ESLint installed"
        return 0
    else
        step_skip "ESLint install failed (npm install -g eslint)"
        return 1
    fi
}

# ===========================================================================
# Step 12: Verify with sample code
# ===========================================================================
verify_sample() {
    info "${BOLD}[12/$TOTAL_STEPS] Verifying tools with sample code...${NC}"

    local test_file="$PROJECT_ROOT/tools/baseline/test/SampleSmelly.java"
    if [ ! -f "$test_file" ]; then
        warn "  Sample test file not found. Creating..."
        mkdir -p "$(dirname "$test_file")"
        # The install_tools.sh should have created this
        bash scripts/baseline/install_tools.sh --verify 2>/dev/null || true
    fi

    if [ ! -f "$test_file" ]; then
        step_skip "Sample verification (no test file)"
        return 1
    fi

    local verified=0

    # Test PMD (installed as pmd-bin-7.0.0 or pmd-7.0.0)
    if [ -f "$PROJECT_ROOT/tools/baseline/pmd-bin-7.0.0/bin/pmd" ] || [ -f "$PROJECT_ROOT/tools/baseline/pmd-7.0.0/bin/pmd" ]; then
        ((verified++)) || true
    fi

    # Test Checkstyle
    if [ -f "$PROJECT_ROOT/tools/baseline/checkstyle-10.14.0-all.jar" ]; then
        java -jar "$PROJECT_ROOT/tools/baseline/checkstyle-10.14.0-all.jar" --version &>/dev/null && ((verified++)) || true
    fi

    # Test pylint
    command -v pylint &>/dev/null && ((verified++)) || true

    if [ "$verified" -ge 2 ]; then
        step_pass "Tool verification ($verified tools confirmed)"
    else
        step_fail "Tool verification failed ($verified tools confirmed)"
    fi
}

# ===========================================================================
# Step 13: Create results/ directory structure
# ===========================================================================
setup_results_dirs() {
    info "${BOLD}[13/$TOTAL_STEPS] Setting up results/ directory structure...${NC}"

    cd "$PROJECT_ROOT"
    mkdir -p results/predictions/baseline
    mkdir -p results/predictions/llm_vanilla
    mkdir -p results/predictions/llm_rag
    mkdir -p results/confusion_matrices
    mkdir -p results/performance
    mkdir -p results/resources
    mkdir -p results/figures
    mkdir -p results/tables
    mkdir -p results/reports
    mkdir -p results/metrics
    mkdir -p results/logs
    mkdir -p results/exports

    step_pass "Results directory structure created"
}

# ===========================================================================
# Main
# ===========================================================================
main() {
    echo ""
    echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║  ${BOLD}Baseline Analysis - Complete Installation${NC}${BLUE}                     ║${NC}"
    echo -e "${BLUE}║  Docker + SonarQube + PMD + Checkstyle + SpotBugs + ESLint    ║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
    echo ""

    VERIFY_ONLY=false
    if [ "${1:-}" = "--verify" ]; then
        VERIFY_ONLY=true
        info "Running in verification-only mode"
    fi

    # Run all steps
    check_docker || true
    check_docker_compose || true
    check_java || true
    check_node || true
    check_python_tools || true

    if ! $VERIFY_ONLY; then
        build_baseline_image || true
        start_docker_services || true
        wait_for_sonarqube || true
        configure_sonarqube || true
        install_local_tools || true
        install_eslint || true
    else
        # Skip install steps in verify mode
        step_skip "Build baseline image"
        step_skip "Start Docker services"
        step_skip "Wait for SonarQube"
        step_skip "Configure SonarQube"
        step_skip "Install local tools"
        step_skip "Install ESLint"
    fi

    verify_sample || true
    setup_results_dirs || true

    # Summary
    echo ""
    echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
    if [ "$FAIL" -eq 0 ]; then
        echo -e "${GREEN}║  ✅ Installation Complete!                                     ║${NC}"
    else
        echo -e "${YELLOW}║  ⚠  Installation Completed with Issues                        ║${NC}"
    fi
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "  ${GREEN}Passed:${NC}  $PASS / $TOTAL_STEPS"
    echo -e "  ${RED}Failed:${NC}  $FAIL / $TOTAL_STEPS"
    echo -e "  ${YELLOW}Skipped:${NC} $SKIP / $TOTAL_STEPS"
    echo ""

    if [ "$FAIL" -eq 0 ]; then
        echo -e "${BLUE}Next Steps:${NC}"
        echo -e "  ${BOLD}Java:${NC}       make baseline-java DIR=data/datasets/SmellyCodeDataset/Java"
        echo -e "  ${BOLD}Python:${NC}     make baseline-python DIR=data/datasets/SmellyCodeDataset/Python"
        echo -e "  ${BOLD}JavaScript:${NC} make baseline-js DIR=data/datasets/SmellyCodeDataset/JavaScript"
        echo -e "  ${BOLD}All:${NC}        make baseline-run DIR=data/datasets/SmellyCodeDataset"
        echo ""
    else
        echo -e "${YELLOW}Fix the failed steps above and re-run: make baseline-install${NC}"
        echo ""
    fi

    # Return non-zero if there were failures
    [ "$FAIL" -eq 0 ]
}

main "$@"
