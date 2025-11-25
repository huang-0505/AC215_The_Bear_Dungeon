#!/bin/bash

# Test script for Combat Agent Enemy AI
# This script tests if the enemy AI (GenAI) is working correctly and not freezing

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
API_BASE="http://localhost:8080/api"
TIMEOUT=10
USE_DIRECT=false

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Combat Agent Enemy AI Test${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if Docker containers are running
echo -e "${YELLOW}Checking if Docker containers are running...${NC}"
if ! docker ps | grep -q "dnd-combat-agent\|dnd-api-gateway\|dnd-nginx"; then
    echo -e "${RED}ERROR: Docker containers are not running${NC}"
    echo ""
    echo "Please start the services first:"
    echo "  docker compose up -d"
    echo ""
    echo "Then wait a few seconds for services to start, and run this test again."
    exit 1
fi
echo -e "${GREEN}✓ Docker containers are running${NC}"

# Check if API is accessible (skip detailed check, just verify container is running)
echo -e "${YELLOW}Checking if API is accessible...${NC}"
# Just check if we can reach the combat agent directly (we'll use it anyway for /start)
if docker exec dnd-combat-agent curl -s --max-time 2 -X POST http://localhost:9000/combat/start \
  -H "Content-Type: application/json" -d '{}' 2>/dev/null | grep -q "session_id"; then
    echo -e "${GREEN}✓ Combat agent is accessible${NC}"
    USE_DIRECT=true
else
    # Try nginx as fallback
    if curl -s --max-time 2 -X POST "$API_BASE/combat/start" \
      -H "Content-Type: application/json" -d '{}' 2>/dev/null | grep -q "session_id"; then
        echo -e "${GREEN}✓ API is accessible via nginx${NC}"
        USE_DIRECT=false
    else
        echo -e "${RED}ERROR: Cannot reach combat agent${NC}"
        echo ""
        echo "Troubleshooting:"
        echo "  1. Check if containers are running: docker ps"
        echo "  2. Check combat agent logs: docker logs dnd-combat-agent"
        echo "  3. Make sure nginx is running: docker ps | grep nginx"
        exit 1
    fi
fi
echo ""

# Step 1: Start combat (always use combat agent directly for /start)
echo -e "${YELLOW}Step 1: Starting combat session...${NC}"
START_RESPONSE=$(docker exec dnd-combat-agent curl -s -X POST "http://localhost:9000/combat/start" \
  -H "Content-Type: application/json" \
  -d '{}')

SESSION_ID=$(echo "$START_RESPONSE" | jq -r '.session_id // empty')

if [ -z "$SESSION_ID" ] || [ "$SESSION_ID" = "null" ]; then
    echo -e "${RED}ERROR: Failed to start combat session${NC}"
    echo "Response: $START_RESPONSE"
    exit 1
fi

echo -e "${GREEN}✓ Combat started${NC}"
echo -e "  Session ID: ${BLUE}$SESSION_ID${NC}"
echo ""

# Step 2: Get initial state
echo -e "${YELLOW}Step 2: Getting initial combat state...${NC}"
if [ "$USE_DIRECT" = true ]; then
    INITIAL_STATE=$(docker exec dnd-combat-agent curl -s "http://localhost:9000/combat/state/$SESSION_ID")
else
    # Use orchestrator proxy endpoint
    INITIAL_STATE=$(curl -s "http://localhost:8080/api/combat/state/$SESSION_ID")
fi
CURRENT_ACTOR=$(echo "$INITIAL_STATE" | jq -r '.current_actor // "Unknown"')
ROUND=$(echo "$INITIAL_STATE" | jq -r '.round // 0')

echo -e "${GREEN}✓ Initial state retrieved${NC}"
echo -e "  Round: ${BLUE}$ROUND${NC}"
echo -e "  Current Actor: ${BLUE}$CURRENT_ACTOR${NC}"
echo ""

# Step 3: Test enemy AI (this is the critical test)
echo -e "${YELLOW}Step 3: Testing Enemy AI (this may take a few seconds)...${NC}"
echo -e "  ${BLUE}Triggering enemy turn - GenAI should respond within 3 seconds...${NC}"
echo ""

# Start timer
START_TIME=$(date +%s)

# Make the action request (this will trigger enemy AI if it's enemy's turn)
if [ "$USE_DIRECT" = true ]; then
    ACTION_RESPONSE=$(docker exec dnd-combat-agent curl -s --max-time $TIMEOUT -X POST "http://localhost:9000/combat/action/$SESSION_ID" \
      -H "Content-Type: application/json" \
      -d '{"action": "I wait and observe"}')
else
    # Use orchestrator proxy endpoint
    ACTION_RESPONSE=$(curl -s --max-time $TIMEOUT -X POST "http://localhost:8080/api/combat/action/$SESSION_ID" \
      -H "Content-Type: application/json" \
      -d '{"action": "I wait and observe"}')
fi

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ ERROR: Request timed out or failed after ${ELAPSED}s${NC}"
    echo "This indicates the enemy AI is freezing!"
    echo ""
    echo "Check the logs with: docker logs dnd-combat-agent --tail 50"
    exit 1
fi

NARRATIVE=$(echo "$ACTION_RESPONSE" | jq -r '.narrative // "No narrative"')
if [ "$USE_DIRECT" = true ]; then
    NEW_STATE=$(docker exec dnd-combat-agent curl -s "http://localhost:9000/combat/state/$SESSION_ID")
else
    # Use orchestrator proxy endpoint
    NEW_STATE=$(curl -s "http://localhost:8080/api/combat/state/$SESSION_ID")
fi
NEW_ACTOR=$(echo "$NEW_STATE" | jq -r '.current_actor // "Unknown"')
NEW_ROUND=$(echo "$NEW_STATE" | jq -r '.round // 0')

echo -e "${GREEN}✓ Action processed in ${ELAPSED}s${NC}"
echo -e "  Narrative: ${BLUE}$NARRATIVE${NC}"
echo -e "  New Round: ${BLUE}$NEW_ROUND${NC}"
echo -e "  New Actor: ${BLUE}$NEW_ACTOR${NC}"
echo ""

# Step 4: Verify state progression
echo -e "${YELLOW}Step 4: Verifying combat progression...${NC}"
if [ "$ROUND" != "$NEW_ROUND" ] || [ "$CURRENT_ACTOR" != "$NEW_ACTOR" ]; then
    echo -e "${GREEN}✓ Combat state progressed correctly${NC}"
else
    echo -e "${YELLOW}⚠ Combat state didn't change (may be normal if battle ended)${NC}"
fi
echo ""

# Step 5: Test multiple enemy turns
echo -e "${YELLOW}Step 5: Testing multiple enemy turns (stress test)...${NC}"
SUCCESS_COUNT=0
FAIL_COUNT=0

for i in {1..3}; do
    echo -e "  Test ${i}/3: Triggering action..."
    TEST_START=$(date +%s)
    
    if [ "$USE_DIRECT" = true ]; then
        TEST_RESPONSE=$(docker exec dnd-combat-agent curl -s --max-time $TIMEOUT -X POST "http://localhost:9000/combat/action/$SESSION_ID" \
          -H "Content-Type: application/json" \
          -d '{"action": "attack"}')
    else
        # Use orchestrator proxy endpoint
        TEST_RESPONSE=$(curl -s --max-time $TIMEOUT -X POST "http://localhost:8080/api/combat/action/$SESSION_ID" \
          -H "Content-Type: application/json" \
          -d '{"action": "attack"}')
    fi
    
    TEST_END=$(date +%s)
    TEST_ELAPSED=$((TEST_END - TEST_START))
    
    if [ $? -eq 0 ] && [ "$TEST_ELAPSED" -lt 5 ]; then
        echo -e "    ${GREEN}✓ Completed in ${TEST_ELAPSED}s${NC}"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        echo -e "    ${RED}✗ Failed or took too long (${TEST_ELAPSED}s)${NC}"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
    
    sleep 1
done

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Test Results${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "  Successful requests: ${GREEN}$SUCCESS_COUNT${NC}"
echo -e "  Failed requests: ${RED}$FAIL_COUNT${NC}"
echo ""

if [ $FAIL_COUNT -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed! Enemy AI is working correctly.${NC}"
    echo ""
    echo "To monitor logs in real-time, run:"
    echo "  docker logs dnd-combat-agent -f"
    exit 0
else
    echo -e "${RED}✗ Some tests failed. Enemy AI may be freezing.${NC}"
    echo ""
    echo "Check the logs for details:"
    echo "  docker logs dnd-combat-agent --tail 100"
    echo ""
    echo "Look for messages like:"
    echo "  - [DnDBot] Starting GenAI call..."
    echo "  - [DnDBot] GenAI call timed out..."
    echo "  - [DnDBot] Returning fallback action"
    exit 1
fi

