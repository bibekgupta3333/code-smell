#!/bin/bash
# Quick SonarQube analysis for Java code
# Uses preconfigured SonarQube (docker-compose)
# No manual setup required - just run and get findings

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Configuration
SONAR_HOST="${SONAR_HOST:-http://localhost:9000}"
SONAR_LOGIN="${SONAR_LOGIN:-admin}"
SONAR_PASSWORD="${SONAR_PASSWORD:-admin}"
SONAR_SCANNER_VERSION="5.0.1.3822"

# Arguments
SOURCE_DIR="${1:-.}"
PROJECT_KEY="${2:-code-smell-$(date +%s)}"
OUTPUT_FILE="${3:-results/predictions/baseline/sonarqube_findings.json}"

print_header() {
    echo -e "${BLUE}=== $1 ===${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Check prerequisites
print_header "Checking Prerequisites"

if [ ! -d "$SOURCE_DIR" ]; then
    print_error "Source directory not found: $SOURCE_DIR"
    exit 1
fi
print_success "Source directory exists: $SOURCE_DIR"

if ! command -v curl &> /dev/null; then
    print_error "curl not found"
    exit 1
fi
print_success "curl available"

# Check SonarQube connectivity
print_header "Verifying SonarQube Connection"

if curl -s "$SONAR_HOST/api/system/status" | grep -q '"status":"UP"'; then
    print_success "SonarQube is running at $SONAR_HOST"
else
    print_error "Cannot connect to SonarQube at $SONAR_HOST"
    echo "  Start with: docker compose up -d"
    exit 1
fi

# Check/create project
print_header "Setting Up Project: $PROJECT_KEY"

PROJECT_EXISTS=$(curl -s "$SONAR_HOST/api/projects/search" \
    -u "$SONAR_LOGIN:$SONAR_PASSWORD" \
    -d "projects=$PROJECT_KEY" | grep -c "$PROJECT_KEY" || true)

if [ "$PROJECT_EXISTS" -gt 0 ]; then
    print_success "Project already exists"
else
    print_header "Creating project..."
    curl -s "$SONAR_HOST/api/projects/create" \
        -u "$SONAR_LOGIN:$SONAR_PASSWORD" \
        -d "project=$PROJECT_KEY" \
        -d "name=$PROJECT_KEY" > /dev/null 2>&1 || true
    print_success "Project ready"
fi

# Run SonarScanner
print_header "Running SonarScanner"

# Method 1: Using Docker image
if command -v docker &> /dev/null; then
    print_header "Using Docker for SonarScanner..."

    ABS_SOURCE=$(cd "$SOURCE_DIR" && pwd)

    docker run --rm \
        -v "$ABS_SOURCE:/src:ro" \
        --network=code-smell-network \
        sonarsource/sonar-scanner-cli:$SONAR_SCANNER_VERSION \
        -Dsonar.projectKey="$PROJECT_KEY" \
        -Dsonar.sources=/src \
        -Dsonar.host.url="$SONAR_HOST" \
        -Dsonar.login="$SONAR_LOGIN" \
        -Dsonar.password="$SONAR_PASSWORD" 2>&1 | grep -E "(INFO|ERROR|WARN)" | tail -20

    print_success "SonarScanner completed"
else
    print_error "Docker not available for SonarScanner"
    exit 1
fi

# Fetch and format results
print_header "Retrieving Findings"

mkdir -p "$(dirname "$OUTPUT_FILE")"

# Wait briefly for indexing
sleep 2

FINDINGS=$(curl -s "$SONAR_HOST/api/issues/search" \
    -u "$SONAR_LOGIN:$SONAR_PASSWORD" \
    -d "projectKeys=$PROJECT_KEY")

# Save raw and formatted results
echo "$FINDINGS" > "${OUTPUT_FILE%.json}.raw.json"

# Extract issue count
ISSUE_COUNT=$(echo "$FINDINGS" | grep -o '"total":[0-9]*' | grep -o '[0-9]*' || echo "0")

# Format findings with jq if available, otherwise save raw
if command -v jq &> /dev/null; then
    echo "$FINDINGS" | jq '{
        tool: "SonarQube",
        project_key: "'$PROJECT_KEY'",
        total_issues: .total,
        findings: [.issues[] | {
            tool: "SonarQube",
            file: (.component | split(":") | .[-1]),
            line: .line,
            rule: .rule,
            message: .message,
            severity: (if .severity == "BLOCKER" or .severity == "CRITICAL" then "HIGH" elif .severity == "MAJOR" then "MEDIUM" else "LOW" end),
            type: .type,
            confidence: 0.88
        }]
    }' > "$OUTPUT_FILE"
else
    # Fallback: just format as pretty JSON
    echo "$FINDINGS" | python3 -m json.tool > "$OUTPUT_FILE" 2>/dev/null || echo "$FINDINGS" > "$OUTPUT_FILE"
fi

print_success "Analyzed $ISSUE_COUNT issues found"
print_success "Results saved to: $OUTPUT_FILE"

# Print summary
print_header "Summary"
echo "  Project Key: $PROJECT_KEY"
echo "  Source Dir:  $SOURCE_DIR"
echo "  Issues:      $ISSUE_COUNT"
echo ""
echo "Dashboard: $SONAR_HOST/dashboard?id=$PROJECT_KEY"
