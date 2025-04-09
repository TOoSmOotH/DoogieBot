#!/bin/bash
set -e

# Configuration
API_BASE_URL="http://localhost:8000/api/v1"
EMAIL="admin@example.com"
PASSWORD="change-this-password"
TEST_URL="https://example.com"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}Debugging Fetch Tool Issues${NC}"
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

# Step 2: Check LLM configuration
check_llm_config() {
  echo -e "${BLUE}Step 2: Checking active LLM configuration...${NC}"
  
  LLM_CONFIG=$(curl -s -X GET "${API_BASE_URL}/admin/llm-configs/active" \
    -H "Authorization: Bearer ${TOKEN}")
  
  echo -e "${YELLOW}Active LLM Config:${NC}"
  echo $LLM_CONFIG | jq .
  
  # Extract provider and model
  PROVIDER=$(echo $LLM_CONFIG | jq -r .chat_provider)
  MODEL=$(echo $LLM_CONFIG | jq -r .model)
  
  echo -e "${GREEN}Provider: ${PROVIDER}, Model: ${MODEL}${NC}"
  
  if [[ "$PROVIDER" == "ollama" ]]; then
    echo -e "${YELLOW}Note: You are using Ollama. Make sure your model supports tool calling.${NC}"
    echo -e "${YELLOW}Models like qwen2.5-coder:32b might not fully support tool calling.${NC}"
  fi
}

# Step 3: Check available MCP tools for user
check_mcp_tools() {
  echo -e "${BLUE}Step 3: Checking available MCP tools...${NC}"
  
  MCP_CONFIGS=$(curl -s -X GET "${API_BASE_URL}/mcp/configs" \
    -H "Authorization: Bearer ${TOKEN}")
  
  echo -e "${YELLOW}Available MCP Configurations:${NC}"
  echo $MCP_CONFIGS | jq .
  
  # Check if there's a fetch tool configuration
  FETCH_CONFIG=$(echo $MCP_CONFIGS | jq '[.[] | select(.name | test("fetch"; "i"))]')
  FETCH_CONFIG_COUNT=$(echo $FETCH_CONFIG | jq 'length')
  
  if [[ "$FETCH_CONFIG_COUNT" -gt 0 ]]; then
    echo -e "${GREEN}Found ${FETCH_CONFIG_COUNT} fetch-related configuration(s)${NC}"
    echo $FETCH_CONFIG | jq .
  else
    echo -e "${RED}No fetch-related MCP configurations found. You need to create one.${NC}"
    echo -e "${YELLOW}Here's how to create a fetch tool:${NC}"
    echo "1. Go to Admin > MCP in the web UI"
    echo "2. Create a new configuration with name 'fetch'"
    echo "3. Set it as enabled"
  fi
}

# Step 4: Create a test chat with different prompts
test_various_prompts() {
  echo -e "${BLUE}Step 4: Testing various prompts to trigger fetch tool...${NC}"
  
  # Create a chat
  CHAT_RESPONSE=$(curl -s -X POST "${API_BASE_URL}/chats" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer ${TOKEN}" \
    -d "{\"title\":\"Fetch Tool Debug Test\"}")
  
  CHAT_ID=$(echo $CHAT_RESPONSE | jq -r .id)
  
  if [[ -z "$CHAT_ID" || "$CHAT_ID" == "null" ]]; then
    echo -e "${RED}Failed to create chat. Response:${NC}"
    echo $CHAT_RESPONSE | jq .
    exit 1
  fi
  
  echo -e "${GREEN}Successfully created chat with ID: ${CHAT_ID}${NC}"
  
  # Test prompts
  PROMPTS=(
    "Use the fetch tool to get the URL ${TEST_URL}" 
    "I need you to use your fetch tool capability to retrieve the content from ${TEST_URL}"
    "Get the content from ${TEST_URL} using your fetch tool"
    "Make a web request to ${TEST_URL} using fetch"
    "Please fetch the content from ${TEST_URL} and summarize it"
  )
  
  for prompt in "${PROMPTS[@]}"; do
    echo -e "${YELLOW}Testing prompt: \"${prompt}\"${NC}"
    
    # Using non-streaming endpoint for simplicity
    RESPONSE=$(curl -s -X POST "${API_BASE_URL}/chats/${CHAT_ID}/llm" \
      -H "Content-Type: application/json" \
      -H "Authorization: Bearer ${TOKEN}" \
      -d "{\"content\":\"${prompt}\"}")
    # Capture logs immediately after the request, especially if it might fail
    echo -e "${YELLOW}Capturing Docker logs after non-streaming /llm call...${NC}"
    docker logs doogietest-app-1 --tail 100 || echo -e "${RED}Failed to capture docker logs.${NC}"
    
    
    # Check for tool_calls in the response
    TOOL_CALLS=$(echo $RESPONSE | jq -r '.tool_calls')
    
    if [[ "$TOOL_CALLS" != "null" ]]; then
      echo -e "${GREEN}SUCCESS! Found tool_calls in response:${NC}"
      echo $TOOL_CALLS | jq .
      echo -e "${GREEN}This prompt successfully triggered the fetch tool!${NC}"
      break
    else
      echo -e "${RED}No tool_calls found in response. This prompt didn't trigger the fetch tool.${NC}"
      echo -e "${YELLOW}Response content:${NC}"
      echo $RESPONSE | jq -r '.content' | head -n 10
      echo "..."
    fi
    
    echo -e "${BLUE}------------------------------------------${NC}"
    sleep 2 # Give some time between requests
  done
}

# Run the debug steps
check_dependencies
login
check_llm_config
check_mcp_tools
test_various_prompts

echo -e "${GREEN}Debug script completed!${NC}"
echo "=================================="
echo "Check the results above to understand why the fetch tool isn't being triggered."
