#!/bin/bash
# Baseline Tools Complete Setup
# One-time initialization for all code analysis tools

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  Baseline Code Analysis Tools - Complete Setup                 ║${NC}"
echo -e "${BLUE}║  PMD • Checkstyle • SpotBugs • SonarQube                       ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"

# Step 1: Verify Docker
echo -e "\n${BLUE}[1/4] Verifying Docker...${NC}"
if ! command -v docker &> /dev/null; then
    echo -e "${RED}✗ Docker not found${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Docker installed${NC}"

# Step 2: Build baseline tools image
echo -e "\n${BLUE}[2/4] Building baseline tools Docker image...${NC}"
if docker build -f Dockerfile.baseline -t code-smell-baseline:latest . > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Baseline image built (PMD, Checkstyle, SpotBugs)${NC}"
else
    echo -e "${RED}✗ Failed to build baseline image${NC}"
    exit 1
fi

# Step 3: Start docker-compose services
echo -e "\n${BLUE}[3/4] Starting Docker services...${NC}"
if docker compose up -d > /dev/null 2>&1; then
    sleep 10  # Wait for services to initialize
    echo -e "${GREEN}✓ Docker services started${NC}"
    echo "  • app"
    echo "  • baseline-tools"
    echo "  • sonarqube"
    echo "  • sonarqube-db"
else
    echo -e "${RED}✗ Failed to start docker-compose${NC}"
    exit 1
fi

# Step 4: Setup SonarQube
echo -e "\n${BLUE}[4/4] Configuring SonarQube...${NC}"
if bash scripts/baseline/sonarqube_setup.sh > /dev/null 2>&1; then
    echo -e "${GREEN}✓ SonarQube initialized${NC}"
    echo -e "  • Token: $(head -c 20 .sonar-token)..."
    echo -e "  • Dashboard: http://localhost:9000"
else
    echo -e "${YELLOW}⚠ SonarQube setup had issues (continuing anyway)${NC}"
fi

# Verify all services
echo -e "\n${BLUE}Service Status:${NC}"
docker compose ps

echo -e "\n${GREEN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  ✅ Setup Complete!                                            ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════════╝${NC}"

echo -e "\n${BLUE}Next Command:${NC}"
echo -e "  ${YELLOW}bash scripts/baseline/analyze.sh <source-dir> <project-key>${NC}"
echo ""
echo -e "${BLUE}Example:${NC}"
echo -e "  ${YELLOW}bash scripts/baseline/analyze.sh \\${NC}"
echo -e "    ${YELLOW}data/datasets/SmellyCodeDataset/Java \\${NC}"
echo -e "    ${YELLOW}code-smell-java${NC}"
echo ""
echo -e "${BLUE}Tools Available:${NC}"
echo -e "  • PMD (Static analysis)"
echo -e "  • Checkstyle (Code style)"
echo -e "  • SpotBugs (Bug detection)"
echo -e "  • SonarQube (Quality platform)"
echo ""
