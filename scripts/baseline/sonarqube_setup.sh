#!/bin/bash
# SonarQube Initial Setup & Authentication
# One-time setup to enable preconfigured SonarQube for Java bug detection

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}=== SonarQube Initial Setup ===${NC}"

SONAR_HOST="${SONAR_HOST:-http://localhost:9000}"
SONAR_USER="${SONAR_USER:-admin}"
SONAR_OLD_PASSWORD="${SONAR_OLD_PASSWORD:-admin}"
SONAR_NEW_PASSWORD="${SONAR_NEW_PASSWORD:-admin123}"
TOKEN_NAME="${TOKEN_NAME:-code-smell-analyzer}"

# Step 1: Verify SonarQube is running
echo -e "${BLUE}1. Checking SonarQube connection...${NC}"
if ! curl -s "$SONAR_HOST/api/system/status" > /dev/null 2>&1; then
    echo -e "${RED}✗ Cannot reach SonarQube at $SONAR_HOST${NC}"
    echo "  Please start with: docker compose up -d"
    exit 1
fi
echo -e "${GREEN}✓ SonarQube is running${NC}"

# Step 2: Try to change admin password
echo -e "${BLUE}2. Setting up admin password...${NC}"

# First, test if we can authenticate with the old password
if curl -s -u "$SONAR_USER:$SONAR_OLD_PASSWORD" \
    "$SONAR_HOST/api/authentication/validate" | grep -q "true"; then
    echo -e "${GREEN}✓ Admin credentials working (old password)${NC}"

    # Try to change password
    echo -e "${BLUE}   Changing password...${NC}"
    RESPONSE=$(curl -s -u "$SONAR_USER:$SONAR_OLD_PASSWORD" \
        -X POST "$SONAR_HOST/api/users/change_password" \
        -d "login=$SONAR_USER" \
        -d "previousPassword=$SONAR_OLD_PASSWORD" \
        -d "password=$SONAR_NEW_PASSWORD")

    if echo "$RESPONSE" | grep -q "error"; then
        echo -e "${YELLOW}⚠ Could not change password (might already be changed)${NC}"
    else
        echo -e "${GREEN}✓ Password changed to: $SONAR_NEW_PASSWORD${NC}"
    fi
else
    # Try with new password
    if curl -s -u "$SONAR_USER:$SONAR_NEW_PASSWORD" \
        "$SONAR_HOST/api/authentication/validate" | grep -q "true"; then
        echo -e "${GREEN}✓ Using new password: $SONAR_NEW_PASSWORD${NC}"
    else
        echo -e "${YELLOW}⚠ Neither old nor new password works${NC}"
        echo "   Please login to http://localhost:9000 and set password manually"
        echo "   Then run: export SONAR_NEW_PASSWORD=your_password"
        SONAR_NEW_PASSWORD=""
    fi
fi

# Step 3: Generate API token
if [ -n "$SONAR_NEW_PASSWORD" ]; then
    echo -e "${BLUE}3. Generating API token...${NC}"

    # Delete old token if exists
    curl -s -u "$SONAR_USER:$SONAR_NEW_PASSWORD" \
        -X POST "$SONAR_HOST/api/user_tokens/revoke" \
        -d "name=$TOKEN_NAME" > /dev/null 2>&1 || true

    TOKEN_RESPONSE=$(curl -s -u "$SONAR_USER:$SONAR_NEW_PASSWORD" \
        -X POST "$SONAR_HOST/api/user_tokens/generate" \
        -d "name=$TOKEN_NAME")

    if echo "$TOKEN_RESPONSE" | grep -q "token"; then
        TOKEN=$(echo "$TOKEN_RESPONSE" | grep -o '"token":"[^"]*"' | cut -d'"' -f4)
        echo -e "${GREEN}✓ Token generated: ${TOKEN:0:20}...${NC}"

        # Save to file
        echo "$TOKEN" > .sonar-token
        echo -e "${GREEN}✓ Saved to: .sonar-token${NC}"
    else
        echo -e "${YELLOW}⚠ Could not generate token${NC}"
        echo "$TOKEN_RESPONSE"
    fi
else
    echo -e "${YELLOW}⚠ Skipping token generation (no valid password)${NC}"
fi

# Step 4: Create default project
echo -e "${BLUE}4. Creating default project...${NC}"

if [ -n "$SONAR_NEW_PASSWORD" ]; then
    PROJECT_KEY="code-smell-java"

    # Check if project exists
    if curl -s -u "$SONAR_USER:$SONAR_NEW_PASSWORD" \
        "$SONAR_HOST/api/projects/search" \
        -d "projects=$PROJECT_KEY" | grep -q "$PROJECT_KEY"; then
        echo -e "${GREEN}✓ Project already exists: $PROJECT_KEY${NC}"
    else
        # Create project
        curl -s -u "$SONAR_USER:$SONAR_NEW_PASSWORD" \
            -X POST "$SONAR_HOST/api/projects/create" \
            -d "project=$PROJECT_KEY" \
            -d "name=Code Smell Detection - Java" > /dev/null 2>&1
        echo -e "${GREEN}✓ Created project: $PROJECT_KEY${NC}"
    fi
else
    echo -e "${YELLOW}⚠ Skipping project creation${NC}"
fi

# Summary
echo -e "${BLUE}=== Setup Complete ===${NC}"
echo ""
echo "Next steps:"
echo "1. Access SonarQube Dashboard:"
echo "   open $SONAR_HOST"
echo ""
echo "2. Run analysis:"
echo "   bash scripts/sonarqube_quick_scan.sh <dir> code-smell-java <output>"
echo ""
echo "3. Or use token directly:"
if [ -f ".sonar-token" ]; then
    TOKEN=$(cat .sonar-token)
    echo "   export SONAR_TOKEN=$TOKEN"
else
    echo "   export SONAR_TOKEN=<your-token>"
fi
echo ""
echo "4. Use in docker run:"
echo "   docker run --rm \\"
echo "     -v \$(pwd):/src:ro \\"
echo "     --network=code-smell-network \\"
echo "     sonarsource/sonar-scanner-cli:latest \\"
echo "     -Dsonar.projectKey=code-smell-java \\"
echo "     -Dsonar.sources=/src \\"
echo "     -Dsonar.host.url=$SONAR_HOST \\"
if [ -f ".sonar-token" ]; then
    echo "     -Dsonar.token=\$(cat .sonar-token)"
else
    echo "     -Dsonar.login=$SONAR_USER \\"
    echo "     -Dsonar.password=<PASSWORD>"
fi
echo ""
