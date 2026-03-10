#!/usr/bin/env bash
# ===========================================================================
# Setup Baseline Static Analysis Tools
# LLM-Based Code Smell Detection Research
#
# Tools: PMD, Checkstyle, SpotBugs, SonarQube (Docker), IntelliJ IDEA
# Target: macOS Apple Silicon (M4 Pro)
# Prerequisite: Homebrew, Docker Desktop (for SonarQube)
#
# Locked versions for reproducibility (Benchmarking Section 2):
#   PMD           7.0.0
#   Checkstyle    10.14.0
#   SpotBugs      4.8.3
#   SonarQube     10.4 Community Edition (Docker)
#   SonarScanner  5.0.1
#   Java          17+ (Temurin)
#
# Usage:
#   bash scripts/setup_baseline_tools.sh          # Install all
#   bash scripts/setup_baseline_tools.sh --verify  # Verify only
# ===========================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TOOLS_DIR="$PROJECT_ROOT/tools/baseline"
VERSION_LOCK="$TOOLS_DIR/VERSIONS.lock"

# Locked tool versions
JAVA_MIN_VERSION="17"
PMD_VERSION="7.0.0"
CHECKSTYLE_VERSION="10.14.0"
SPOTBUGS_VERSION="4.8.3"
SONAR_SCANNER_VERSION="5.0.1.3006"
SONARQUBE_DOCKER_TAG="10.4-community"

# Download URLs
PMD_URL="https://github.com/pmd/pmd/releases/download/pmd_releases%2F${PMD_VERSION}/pmd-dist-${PMD_VERSION}-bin.zip"
CHECKSTYLE_URL="https://github.com/checkstyle/checkstyle/releases/download/checkstyle-${CHECKSTYLE_VERSION}/checkstyle-${CHECKSTYLE_VERSION}-all.jar"
SPOTBUGS_URL="https://github.com/spotbugs/spotbugs/releases/download/${SPOTBUGS_VERSION}/spotbugs-${SPOTBUGS_VERSION}.tgz"
SONAR_SCANNER_URL="https://binaries.sonarsource.com/Distribution/sonar-scanner-cli/sonar-scanner-cli-${SONAR_SCANNER_VERSION}.zip"

# Analysis timeout per file (seconds)
ANALYSIS_TIMEOUT=60

# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[✓]${NC} $1"; }
warn()    { echo -e "${YELLOW}[!]${NC} $1"; }
fail()    { echo -e "${RED}[✗]${NC} $1"; }

# ---------------------------------------------------------------------------
# Prerequisites
# ---------------------------------------------------------------------------
check_java() {
    info "Checking Java installation..."
    if ! command -v java &>/dev/null; then
        warn "Java not found. Installing Temurin JDK ${JAVA_MIN_VERSION} via Homebrew..."
        brew install --cask temurin@${JAVA_MIN_VERSION}
    fi

    local java_ver
    java_ver=$(java -version 2>&1 | head -1 | sed -E 's/.*"([0-9]+).*/\1/')
    if [ "$java_ver" -lt "$JAVA_MIN_VERSION" ]; then
        fail "Java $JAVA_MIN_VERSION+ required (found $java_ver)"
        exit 1
    fi
    success "Java $java_ver OK"
}

check_docker() {
    info "Checking Docker (required for SonarQube)..."
    if ! command -v docker &>/dev/null; then
        warn "Docker not found — SonarQube setup will be skipped."
        warn "Install Docker Desktop from https://www.docker.com/products/docker-desktop/"
        return 1
    fi
    if ! docker info &>/dev/null; then
        warn "Docker daemon not running — SonarQube setup will be skipped."
        return 1
    fi
    success "Docker OK"
    return 0
}

# ---------------------------------------------------------------------------
# Tool Installers
# ---------------------------------------------------------------------------
install_pmd() {
    local dest="$TOOLS_DIR/pmd-${PMD_VERSION}"
    if [ -f "$dest/bin/pmd" ]; then
        success "PMD ${PMD_VERSION} already installed"
        return
    fi

    info "Downloading PMD ${PMD_VERSION}..."
    local tmp_zip="$TOOLS_DIR/pmd.zip"
    curl -fsSL "$PMD_URL" -o "$tmp_zip"
    unzip -qo "$tmp_zip" -d "$TOOLS_DIR"
    rm -f "$tmp_zip"
    chmod +x "$dest/bin/pmd"
    success "PMD ${PMD_VERSION} installed → $dest"
}

install_checkstyle() {
    local dest="$TOOLS_DIR/checkstyle-${CHECKSTYLE_VERSION}-all.jar"
    if [ -f "$dest" ]; then
        success "Checkstyle ${CHECKSTYLE_VERSION} already installed"
        return
    fi

    info "Downloading Checkstyle ${CHECKSTYLE_VERSION}..."
    curl -fsSL "$CHECKSTYLE_URL" -o "$dest"
    success "Checkstyle ${CHECKSTYLE_VERSION} installed → $dest"
}

install_spotbugs() {
    local dest="$TOOLS_DIR/spotbugs-${SPOTBUGS_VERSION}"
    if [ -f "$dest/bin/spotbugs" ]; then
        success "SpotBugs ${SPOTBUGS_VERSION} already installed"
        return
    fi

    info "Downloading SpotBugs ${SPOTBUGS_VERSION}..."
    local tmp_tgz="$TOOLS_DIR/spotbugs.tgz"
    curl -fsSL "$SPOTBUGS_URL" -o "$tmp_tgz"
    tar -xzf "$tmp_tgz" -C "$TOOLS_DIR"
    rm -f "$tmp_tgz"
    chmod +x "$dest/bin/spotbugs"
    success "SpotBugs ${SPOTBUGS_VERSION} installed → $dest"
}

install_sonar_scanner() {
    local dest="$TOOLS_DIR/sonar-scanner-${SONAR_SCANNER_VERSION}"
    if [ -d "$dest" ]; then
        success "SonarScanner ${SONAR_SCANNER_VERSION} already installed"
        return
    fi

    info "Downloading SonarScanner CLI ${SONAR_SCANNER_VERSION}..."
    local tmp_zip="$TOOLS_DIR/sonar-scanner.zip"
    curl -fsSL "$SONAR_SCANNER_URL" -o "$tmp_zip"
    unzip -qo "$tmp_zip" -d "$TOOLS_DIR"
    rm -f "$tmp_zip"
    # The extracted dir may have a slightly different name
    local extracted
    extracted=$(find "$TOOLS_DIR" -maxdepth 1 -name "sonar-scanner-*" -type d | grep -v ".zip" | head -1)
    if [ -n "$extracted" ] && [ "$extracted" != "$dest" ]; then
        mv "$extracted" "$dest" 2>/dev/null || true
    fi
    chmod +x "$dest/bin/sonar-scanner" 2>/dev/null || true
    success "SonarScanner ${SONAR_SCANNER_VERSION} installed → $dest"
}

setup_sonarqube_docker() {
    if ! check_docker; then
        warn "Skipping SonarQube Docker setup"
        return
    fi

    info "Pulling SonarQube ${SONARQUBE_DOCKER_TAG} image..."
    docker pull "sonarqube:${SONARQUBE_DOCKER_TAG}" --quiet

    # Create helper scripts
    cat > "$TOOLS_DIR/start_sonarqube.sh" << 'SONAR_START'
#!/usr/bin/env bash
# Start SonarQube container for benchmarking
CONTAINER_NAME="codesmell-sonarqube"
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    docker start "$CONTAINER_NAME"
else
    docker run -d --name "$CONTAINER_NAME" \
        -p 9000:9000 \
        -e SONAR_ES_BOOTSTRAP_CHECKS_DISABLE=true \
        sonarqube:10.4-community
fi
echo "SonarQube starting at http://localhost:9000 (default login: admin/admin)"
echo "Wait ~60s for startup. Check: curl -s http://localhost:9000/api/system/status"
SONAR_START
    chmod +x "$TOOLS_DIR/start_sonarqube.sh"

    cat > "$TOOLS_DIR/stop_sonarqube.sh" << 'SONAR_STOP'
#!/usr/bin/env bash
docker stop codesmell-sonarqube 2>/dev/null || true
echo "SonarQube stopped"
SONAR_STOP
    chmod +x "$TOOLS_DIR/stop_sonarqube.sh"

    success "SonarQube Docker image ready (use tools/baseline/start_sonarqube.sh)"
}

check_intellij() {
    info "Checking IntelliJ IDEA installation..."
    local idea_path=""

    # Common macOS locations
    for path in \
        "/Applications/IntelliJ IDEA.app" \
        "/Applications/IntelliJ IDEA CE.app" \
        "/Applications/IntelliJ IDEA Community Edition.app" \
        "$HOME/Applications/IntelliJ IDEA.app" \
        "$HOME/Applications/IntelliJ IDEA CE.app"; do
        if [ -d "$path" ]; then
            idea_path="$path"
            break
        fi
    done

    if [ -n "$idea_path" ]; then
        local inspect_bin="$idea_path/Contents/bin/inspect.sh"
        if [ -f "$inspect_bin" ]; then
            success "IntelliJ IDEA found → $idea_path"
            echo "$idea_path" > "$TOOLS_DIR/intellij_path.txt"
            return
        fi
    fi

    warn "IntelliJ IDEA not found at standard locations."
    warn "Install from: https://www.jetbrains.com/idea/download/ (Community Edition is free)"
    warn "After installing, re-run this script to register the path."
}

# ---------------------------------------------------------------------------
# Sample Test File
# ---------------------------------------------------------------------------
create_test_sample() {
    local test_dir="$TOOLS_DIR/test"
    mkdir -p "$test_dir"

    cat > "$test_dir/SampleSmelly.java" << 'JAVA_SAMPLE'
/**
 * Sample Java class with intentional code smells for tool verification.
 * Code smells present: Long Method, God Class, Long Parameter List, Data Class.
 */
public class SampleSmelly {

    // Data Class smell: only getters/setters, no behavior
    private String name;
    private int age;
    private String email;
    private String phone;
    private String address;
    private double salary;
    private int departmentId;
    private String role;

    public String getName() { return name; }
    public void setName(String name) { this.name = name; }
    public int getAge() { return age; }
    public void setAge(int age) { this.age = age; }
    public String getEmail() { return email; }
    public void setEmail(String email) { this.email = email; }
    public String getPhone() { return phone; }
    public void setPhone(String phone) { this.phone = phone; }
    public String getAddress() { return address; }
    public void setAddress(String address) { this.address = address; }
    public double getSalary() { return salary; }
    public void setSalary(double salary) { this.salary = salary; }
    public int getDepartmentId() { return departmentId; }
    public void setDepartmentId(int id) { this.departmentId = id; }
    public String getRole() { return role; }
    public void setRole(String role) { this.role = role; }

    // Long Parameter List smell (> 5 parameters)
    public void updateEmployee(String name, int age, String email,
                                String phone, String address, double salary,
                                int departmentId, String role) {
        this.name = name;
        this.age = age;
        this.email = email;
        this.phone = phone;
        this.address = address;
        this.salary = salary;
        this.departmentId = departmentId;
        this.role = role;
    }

    // Long Method smell (> 30 lines of logic)
    public String generateReport() {
        StringBuilder sb = new StringBuilder();
        sb.append("=== Employee Report ===\n");
        sb.append("Name: ").append(name).append("\n");
        sb.append("Age: ").append(age).append("\n");
        sb.append("Email: ").append(email).append("\n");
        sb.append("Phone: ").append(phone).append("\n");
        sb.append("Address: ").append(address).append("\n");
        sb.append("Salary: ").append(salary).append("\n");
        sb.append("Department: ").append(departmentId).append("\n");
        sb.append("Role: ").append(role).append("\n");
        sb.append("\n");
        // Validation logic inline (should be separate method)
        if (name == null || name.isEmpty()) {
            sb.append("WARNING: Name is missing\n");
        }
        if (age < 0 || age > 150) {
            sb.append("WARNING: Invalid age\n");
        }
        if (email == null || !email.contains("@")) {
            sb.append("WARNING: Invalid email\n");
        }
        if (salary < 0) {
            sb.append("WARNING: Negative salary\n");
        }
        // Formatting logic inline (should be separate method)
        String status;
        if (salary > 100000) {
            status = "Senior";
        } else if (salary > 50000) {
            status = "Mid-level";
        } else {
            status = "Junior";
        }
        sb.append("Status: ").append(status).append("\n");
        // Summary calculation inline
        double bonus = 0;
        if ("Senior".equals(status)) {
            bonus = salary * 0.15;
        } else if ("Mid-level".equals(status)) {
            bonus = salary * 0.10;
        } else {
            bonus = salary * 0.05;
        }
        sb.append("Bonus: ").append(bonus).append("\n");
        sb.append("Total: ").append(salary + bonus).append("\n");
        sb.append("========================\n");
        return sb.toString();
    }
}
JAVA_SAMPLE

    success "Test sample created → $test_dir/SampleSmelly.java"
}

# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
verify_installations() {
    info "Verifying tool installations..."
    local pass_count=0
    local total=5

    # PMD
    local pmd_bin="$TOOLS_DIR/pmd-${PMD_VERSION}/bin/pmd"
    if [ -f "$pmd_bin" ]; then
        if "$pmd_bin" --version 2>&1 | grep -q "$PMD_VERSION"; then
            success "PMD ${PMD_VERSION} — verified"
            ((pass_count++))
        else
            fail "PMD binary exists but version mismatch"
        fi
    else
        fail "PMD not installed"
    fi

    # Checkstyle
    local cs_jar="$TOOLS_DIR/checkstyle-${CHECKSTYLE_VERSION}-all.jar"
    if [ -f "$cs_jar" ]; then
        if java -jar "$cs_jar" --version 2>&1 | grep -q "$CHECKSTYLE_VERSION"; then
            success "Checkstyle ${CHECKSTYLE_VERSION} — verified"
            ((pass_count++))
        else
            fail "Checkstyle JAR exists but version mismatch"
        fi
    else
        fail "Checkstyle not installed"
    fi

    # SpotBugs
    local sb_bin="$TOOLS_DIR/spotbugs-${SPOTBUGS_VERSION}/bin/spotbugs"
    if [ -f "$sb_bin" ]; then
        success "SpotBugs ${SPOTBUGS_VERSION} — binary present"
        ((pass_count++))
    else
        fail "SpotBugs not installed"
    fi

    # SonarScanner
    local ss_bin="$TOOLS_DIR/sonar-scanner-${SONAR_SCANNER_VERSION}/bin/sonar-scanner"
    if [ -f "$ss_bin" ]; then
        success "SonarScanner ${SONAR_SCANNER_VERSION} — binary present"
        ((pass_count++))
    else
        fail "SonarScanner not installed"
    fi

    # IntelliJ IDEA
    if [ -f "$TOOLS_DIR/intellij_path.txt" ]; then
        success "IntelliJ IDEA — path registered"
        ((pass_count++))
    else
        warn "IntelliJ IDEA — not found (optional)"
        ((pass_count++))  # Don't block on IntelliJ
    fi

    echo ""
    if [ "$pass_count" -eq "$total" ]; then
        success "All $total tools verified ($pass_count/$total)"
    else
        warn "$pass_count/$total tools verified"
    fi
}

# ---------------------------------------------------------------------------
# Version Lock File
# ---------------------------------------------------------------------------
write_version_lock() {
    cat > "$VERSION_LOCK" << EOF
# Baseline Tool Versions — Locked for Reproducibility
# Generated: $(date -u +"%Y-%m-%dT%H:%M:%SZ")
# Hardware: $(uname -m) / $(sw_vers -productName 2>/dev/null || echo "macOS") $(sw_vers -productVersion 2>/dev/null || echo "unknown")
# Java: $(java -version 2>&1 | head -1)

PMD=${PMD_VERSION}
CHECKSTYLE=${CHECKSTYLE_VERSION}
SPOTBUGS=${SPOTBUGS_VERSION}
SONAR_SCANNER=${SONAR_SCANNER_VERSION}
SONARQUBE_DOCKER=${SONARQUBE_DOCKER_TAG}
ANALYSIS_TIMEOUT=${ANALYSIS_TIMEOUT}
EOF
    success "Version lock written → $VERSION_LOCK"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
main() {
    echo ""
    echo "====================================================="
    echo "  Baseline Tool Setup — Code Smell Detection Research"
    echo "====================================================="
    echo ""

    # Ensure tools directory exists
    mkdir -p "$TOOLS_DIR"

    if [ "${1:-}" = "--verify" ]; then
        verify_installations
        exit 0
    fi

    # Prerequisites
    check_java

    # Install tools
    install_pmd
    install_checkstyle
    install_spotbugs
    install_sonar_scanner
    setup_sonarqube_docker

    # Check IntelliJ
    check_intellij

    # Create test sample
    create_test_sample

    # Write version lock
    write_version_lock

    # Verify
    echo ""
    verify_installations

    echo ""
    info "Add 'tools/' to .gitignore (large binaries, do not commit)."
    info "Run baseline tools: python scripts/baseline/run_tools.py --help"
    echo ""
}

main "$@"
