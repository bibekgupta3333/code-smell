#!/bin/bash
# SonarQube Java Analysis - Compile & Analyze
# Automatically compiles Java source files before SonarQube analysis

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check arguments
if [ "$#" -lt 2 ]; then
    echo -e "${BLUE}Usage:${NC}"
    echo "  $0 <source-dir> <project-key> [output-file]"
    echo ""
    echo -e "${BLUE}Examples:${NC}"
    echo "  $0 data/datasets/SmellyCodeDataset/Java code-smell-java"
    echo "  $0 data/datasets/SmellyCodeDataset/Java code-smell-java results/sonarqube.json"
    echo ""
    exit 1
fi

SOURCE_DIR="$1"
PROJECT_KEY="$2"
OUTPUT_FILE="${3:-results/predictions/baseline/sonarqube_${PROJECT_KEY}.json}"

SONAR_HOST="${SONAR_HOST:-http://localhost:9000}"
SONAR_DOCKER_HOST="${SONAR_DOCKER_HOST:-http://sonarqube:9000}"
SONAR_TOKEN_FILE="${SONAR_TOKEN_FILE:-.sonar-token}"

# Verify source directory
if [ ! -d "$SOURCE_DIR" ]; then
    echo -e "${RED}✗ Source directory not found: $SOURCE_DIR${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Source directory: $SOURCE_DIR${NC}"

# Check if token file exists
if [ ! -f "$SONAR_TOKEN_FILE" ]; then
    echo -e "${RED}✗ Token file not found: $SONAR_TOKEN_FILE${NC}"
    echo -e "${YELLOW}Please run setup first:${NC}"
    echo "  bash scripts/baseline/sonarqube_setup.sh"
    exit 1
fi

SONAR_TOKEN=$(cat "$SONAR_TOKEN_FILE")
echo -e "${GREEN}✓ Using token: ${SONAR_TOKEN:0:20}...${NC}"

# Verify SonarQube is running
if ! curl -s "$SONAR_HOST/api/system/status" > /dev/null 2>&1; then
    echo -e "${RED}✗ Cannot reach SonarQube${NC}"
    echo "  Start with: docker compose up -d"
    exit 1
fi
echo -e "${GREEN}✓ SonarQube is running${NC}"

# Create temp directory for compilation
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

echo -e "${BLUE}Preparing compilation environment...${NC}"

# Copy source to temp directory
cp -r "$SOURCE_DIR"/* "$TEMP_DIR/" 2>/dev/null || true

# Create build directory
BUILD_DIR="$TEMP_DIR/build"
mkdir -p "$BUILD_DIR"

# Compile Java files
echo -e "${BLUE}Compiling Java files...${NC}"

JAVA_FILES=$(find "$TEMP_DIR" -name "*.java" -type f)
FILE_COUNT=$(echo "$JAVA_FILES" | wc -l)

if [ "$FILE_COUNT" -eq 0 ]; then
    echo -e "${YELLOW}⚠ No Java files found in $SOURCE_DIR${NC}"
    exit 1
fi

echo "  Found $FILE_COUNT Java files"

# Compile with javac (if available)
if command -v javac &> /dev/null; then
    echo "  Using system javac..."
    javac -d "$BUILD_DIR" $JAVA_FILES 2>/dev/null || true

    # Count compiled classes
    CLASS_COUNT=$(find "$BUILD_DIR" -name "*.class" -type f | wc -l)
    if [ "$CLASS_COUNT" -gt 0 ]; then
        echo -e "${GREEN}✓ Compiled $CLASS_COUNT classes${NC}"
    else
        echo -e "${YELLOW}⚠ No classes compiled (might have syntax errors)${NC}"
    fi
else
    echo -e "${YELLOW}⚠ javac not available, using Docker compilation${NC}"

    # Use Docker to compile
    docker run --rm \
        -v "$TEMP_DIR:/src:rw" \
        eclipse-temurin:17-jdk \
        bash -c "javac -d /src/build /src/**/*.java 2>/dev/null || true" || true

    CLASS_COUNT=$(find "$BUILD_DIR" -name "*.class" -type f | wc -l)
    if [ "$CLASS_COUNT" -gt 0 ]; then
        echo -e "${GREEN}✓ Compiled $CLASS_COUNT classes${NC}"
    fi
fi

# Prepare output directory
mkdir -p "$(dirname "$OUTPUT_FILE")"

# Run sonar-scanner
echo -e "${BLUE}Running SonarQube analysis...${NC}"

ABS_TEMP=$(cd "$TEMP_DIR" && pwd)

docker run --rm \
    -v "$ABS_TEMP:/src:ro" \
    --network=code-smell-network \
    sonarsource/sonar-scanner-cli:latest \
    -Dsonar.projectKey="$PROJECT_KEY" \
    -Dsonar.projectBaseDir=/src \
    -Dsonar.sources=/src \
    -Dsonar.java.binaries=/src/build \
    -Dsonar.host.url="$SONAR_DOCKER_HOST" \
    -Dsonar.token="$SONAR_TOKEN" 2>&1 | grep -E "(files|source|ANALYSIS|ERROR)" | tail -20

echo -e "${GREEN}✓ Analysis complete${NC}"

# Fetch results
echo -e "${BLUE}Retrieving findings...${NC}"

sleep 2

FINDINGS=$(curl -s -H "Authorization: Bearer $SONAR_TOKEN" \
    "$SONAR_HOST/api/issues/search?projectKeys=$PROJECT_KEY&ps=500")

# Save results
echo "$FINDINGS" | python3 -m json.tool > "$OUTPUT_FILE" 2>/dev/null || echo "$FINDINGS" > "$OUTPUT_FILE"

# Count issues
ISSUE_COUNT=$(echo "$FINDINGS" | grep -o '"total":[0-9]*' | grep -o '[0-9]*' || echo "0")

echo -e "${GREEN}✓ Results saved: $OUTPUT_FILE${NC}"
echo -e "${GREEN}✓ Issues found: $ISSUE_COUNT${NC}"
echo ""
echo "Dashboard: $SONAR_HOST/dashboard?id=$PROJECT_KEY"
echo ""
echo "View issues:"
echo "  cat $OUTPUT_FILE | python3 -m json.tool | less"
