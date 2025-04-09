#!/bin/bash
set -e

# Configuration
API_BASE_URL="http://localhost:8000/api/v1"
EMAIL="admin@example.com"
PASSWORD="change-this-password"

# Test scenarios
TESTS=(
  "fetch:https://example.com:use fetch to get the URL https://example.com"
  "time:current:what is the current time in Tokyo?"
  "time:convert:convert 14:30 from New York time to Tokyo time"
  "sequentialthinking:math:solve this math problem step by step: what is the sum of all numbers from 1 to 100?"
)

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}Testing Doogie Chat Bot API Tool Functions${NC}"
echo "=================================="

# Function to check if jq is installed
check_dependencies() {
  if ! command -v jq &> /dev/null; then
    echo -e "${RED}Error: jq is not installed. Please install it to run this script.${NC}"
    echo "On Ubuntu/Debian: sudo apt-get install jq"
    echo "On macOS: brew install jq"
    exit 1
  fi
}

# Step 1: Login to get token
login() {
  echo -e "${BLUE}Step 1: Logging in to get auth token...${NC}"
  
  LOGIN_RESPONSE=$(curl -s -X POST "${API_BASE_URL}/auth/login" \
    -H "Content-Type: application/json" \
    -d "{\"username\":\"${EMAIL}\",\"password\":\"${PASSWORD}\"}")
  echo "Debug - Login response: ${LOGIN_RESPONSE}"
  
  TOKEN=$(echo $LOGIN_RESPONSE | jq -r .access_token)
  
  if [[ -z "$TOKEN" || "$TOKEN" == "null" ]]; then
    echo -e "${RED}Login failed. Response:${NC}"
    echo $LOGIN_RESPONSE | jq .
    exit 1
  fi
  
  echo -e "${GREEN}Successfully logged in${NC}"
}

# Step 2: Create a new chat for a test
create_chat() {
  local test_name=$1
  echo -e "${BLUE}Creating new chat for test: ${test_name}...${NC}"
  
  CHAT_RESPONSE=$(curl -s -X POST "${API_BASE_URL}/chats" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer ${TOKEN}" \
    -d "{\"title\":\"Tool Test: ${test_name}\"}")
  
  CHAT_ID=$(echo $CHAT_RESPONSE | jq -r .id)
  
  if [[ -z "$CHAT_ID" || "$CHAT_ID" == "null" ]]; then
    echo -e "${RED}Failed to create chat. Response:${NC}"
    echo $CHAT_RESPONSE | jq .
    exit 1
  fi
  
  echo -e "${GREEN}Successfully created chat with ID: ${CHAT_ID}${NC}"
}

# Step 3: Test a specific tool scenario
test_tool() {
  local tool=$1
  local subtype=$2
  local message=$3
  
  echo -e "${YELLOW}=====================${NC}"
  echo -e "${BLUE}Testing ${tool} tool (${subtype})...${NC}"
  echo -e "${YELLOW}=====================${NC}"
  
  # Create a chat for this test
  create_chat "${tool}-${subtype}"
  
  echo "Sending message: \"${message}\""
  
  # Use the GET stream endpoint which works well with curl
  STREAM_URL="${API_BASE_URL}/chats/${CHAT_ID}/stream?content=$(echo "$message" | jq -sRr @uri)"
  
  echo "Requesting stream from: ${STREAM_URL}"
  
  echo -e "${BLUE}Streaming response (looking for tool calls)...${NC}"
  curl -s -N -m 30 \
    -H "Authorization: Bearer ${TOKEN}" \
    "${STREAM_URL}" | {
      count=0
      found_tool_call=false
      while read -r line && [ $count -lt 50 ]; do
        if [[ $line == data:* ]]; then
          json_data=${line#data: }
          if [[ $json_data == *"tool_calls"* || $json_data == *"tool_calls_delta"* ]]; then
            echo -e "${GREEN}Found tool call in response!${NC}"
            echo $json_data | jq .
            found_tool_call=true
            break
          fi
          
          if [[ $count -eq 0 || $(($count % 10)) -eq 0 ]]; then
            echo "Processing stream data chunk #$count"
          fi
          count=$((count + 1))
        fi
      done
      
      if [ "$found_tool_call" = false ]; then
        echo -e "${RED}No tool call detected in the first 50 events.${NC}"
      fi
    }
  
  # Check if the message was created by getting messages for the chat
  echo -e "${BLUE}Verifying message was created...${NC}"
  sleep 5 # Give the server time to process
  
  MESSAGES_RESPONSE=$(curl -s -X GET "${API_BASE_URL}/chats/${CHAT_ID}/messages" \
    -H "Authorization: Bearer ${TOKEN}")
  
  LAST_ASSISTANT_MESSAGE=$(echo $MESSAGES_RESPONSE | jq '.[] | select(.role=="assistant") | .content' | tail -1)
  
  if [[ -n "$LAST_ASSISTANT_MESSAGE" && "$LAST_ASSISTANT_MESSAGE" != "null" ]]; then
    echo -e "${GREEN}Assistant responded with message:${NC}"
    echo $LAST_ASSISTANT_MESSAGE | jq -r .
  else
    echo -e "${RED}No assistant message found. Full response:${NC}"
    echo $MESSAGES_RESPONSE | jq .
  fi
  
  echo -e "${YELLOW}Test complete: ${tool} ${subtype}${NC}"
  echo ""
}

# Run the tests
check_dependencies
login

# Run each test case
for test_info in "${TESTS[@]}"; do
  IFS=':' read -ra TEST <<< "$test_info"
  test_tool "${TEST[0]}" "${TEST[1]}" "${TEST[2]}"
done

echo -e "${GREEN}All tests completed!${NC}"
echo "=================================="
