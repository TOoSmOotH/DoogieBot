#!/bin/bash
set -e

# Configuration
API_BASE_URL="http://localhost:8000/api/v1"
EMAIL="admin@example.com"
PASSWORD="change-this-password"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}Verifying Fetch Tool Configuration${NC}"
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
  
  # Try form-based authentication
  LOGIN_RESPONSE=$(curl -s -X POST "${API_BASE_URL}/auth/login" \
    -H "Content-Type: application/x-www-form-urlencoded" \
    -d "username=${EMAIL}&password=${PASSWORD}")
  
  TOKEN=$(echo $LOGIN_RESPONSE | jq -r .access_token)
  
  if [[ -z "$TOKEN" || "$TOKEN" == "null" ]]; then
    echo -e "${RED}Login failed. Could not get auth token.${NC}"
    exit 1
  fi
  
  echo -e "${GREEN}Successfully logged in${NC}"
}

# Step 2: Check Active LLM Configuration
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
    echo -e "${YELLOW}NOTE: You are using Ollama.${NC}"
    echo -e "${YELLOW}Some models like qwen2.5-coder:32b might not fully support tool calling${NC}"
    echo -e "${YELLOW}Try a different model like llama3, or switch to Anthropic/OpenAI provider.${NC}"
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
  FETCH_CONFIG=$(echo $MCP_CONFIGS | jq '[.[] | select(.name | test("fetch"; "i"))] | first')
  FETCH_CONFIG_COUNT=$(echo $FETCH_CONFIG | jq 'length')
  
  if [[ "$FETCH_CONFIG" != "null" ]]; then
    FETCH_CONFIG_ID=$(echo $FETCH_CONFIG | jq -r '.id')
    echo -e "${GREEN}Found fetch tool configuration with ID: ${FETCH_CONFIG_ID}${NC}"
    
    # Check fetch tool status
    FETCH_TOOL_STATUS=$(curl -s -X GET "${API_BASE_URL}/mcp/configs/${FETCH_CONFIG_ID}/status" \
      -H "Authorization: Bearer ${TOKEN}")
    
    echo -e "${YELLOW}Fetch Tool Status:${NC}"
    echo $FETCH_TOOL_STATUS | jq .
    
    FETCH_STATUS=$(echo $FETCH_TOOL_STATUS | jq -r '.status')
    FETCH_ENABLED=$(echo $FETCH_TOOL_STATUS | jq -r '.enabled')
    
    if [[ "$FETCH_ENABLED" != "true" ]]; then
      echo -e "${RED}⚠️ WARNING: Fetch tool is not enabled. This means it won't be available to the LLM.${NC}"
      echo -e "${YELLOW}You should update the configuration to set 'enabled' to true.${NC}"
    else
      echo -e "${GREEN}✓ Fetch tool is enabled.${NC}"
    fi
    
    if [[ "$FETCH_STATUS" != "running" ]]; then
      echo -e "${YELLOW}⚠️ Fetch tool container is not running (status: ${FETCH_STATUS}).${NC}"
      echo -e "${YELLOW}You may want to start it using the API or UI.${NC}"
      
      # Ask if user wants to start it
      read -p "Do you want to start the fetch tool container now? (y/n) " -n 1 -r
      echo
      if [[ $REPLY =~ ^[Yy]$ ]]; then
        START_RESPONSE=$(curl -s -X POST "${API_BASE_URL}/mcp/configs/${FETCH_CONFIG_ID}/start" \
          -H "Authorization: Bearer ${TOKEN}")
        
        echo -e "${YELLOW}Start container response:${NC}"
        echo $START_RESPONSE | jq .
        
        # Check status again
        sleep 2
        FETCH_TOOL_STATUS=$(curl -s -X GET "${API_BASE_URL}/mcp/configs/${FETCH_CONFIG_ID}/status" \
          -H "Authorization: Bearer ${TOKEN}")
        
        echo -e "${YELLOW}Updated Fetch Tool Status:${NC}"
        echo $FETCH_TOOL_STATUS | jq .
        
        FETCH_STATUS=$(echo $FETCH_TOOL_STATUS | jq -r '.status')
        if [[ "$FETCH_STATUS" == "running" ]]; then
          echo -e "${GREEN}✓ Fetch tool container is now running.${NC}"
        else
          echo -e "${RED}⚠️ Failed to start fetch tool container.${NC}"
        fi
      fi
    else
      echo -e "${GREEN}✓ Fetch tool container is running.${NC}"
    fi
  else
    echo -e "${RED}No fetch tool configuration found. You need to create one using configure_fetch_tool.sh${NC}"
  fi
}

# Step 4: Check User ID and Permissions
check_user_permissions() {
  echo -e "${BLUE}Step 4: Checking user permissions...${NC}"
  
  USER_INFO=$(curl -s -X GET "${API_BASE_URL}/users/me" \
    -H "Authorization: Bearer ${TOKEN}")
  
  echo -e "${YELLOW}User Info:${NC}"
  echo $USER_INFO | jq .
  
  USER_ID=$(echo $USER_INFO | jq -r '.id')
  USER_ROLE=$(echo $USER_INFO | jq -r '.role')
  
  echo -e "${GREEN}User ID: ${USER_ID}, Role: ${USER_ROLE}${NC}"
  
  if [[ "$USER_ROLE" != "admin" ]]; then
    echo -e "${RED}⚠️ WARNING: User is not an admin. This might affect tool permissions.${NC}"
  else
    echo -e "${GREEN}✓ User has admin permissions.${NC}"
  fi
}

# Run the verification
check_dependencies
login
check_llm_config
check_mcp_tools
check_user_permissions

echo -e "${GREEN}Fetch tool verification completed!${NC}"
echo "=================================="
