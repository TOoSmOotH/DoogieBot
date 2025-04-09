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

echo -e "${BLUE}Setting up Fetch Tool MCP Configuration${NC}"
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
  
  # Try form-based authentication first as it worked
  LOGIN_RESPONSE=$(curl -s -X POST "${API_BASE_URL}/auth/login" \
    -H "Content-Type: application/x-www-form-urlencoded" \
    -d "username=${EMAIL}&password=${PASSWORD}")
  
  echo "Debug - Login response: ${LOGIN_RESPONSE}"
  TOKEN=$(echo $LOGIN_RESPONSE | jq -r .access_token)
  
  if [[ -z "$TOKEN" || "$TOKEN" == "null" ]]; then
    echo -e "${RED}Login failed. Could not get auth token.${NC}"
    exit 1
  fi
  
  echo -e "${GREEN}Successfully logged in${NC}"
}

# Step 2: Check if fetch tool exists
check_fetch_tool() {
  echo -e "${BLUE}Step 2: Checking if fetch tool configuration exists...${NC}"
  
  MCP_CONFIGS=$(curl -s -X GET "${API_BASE_URL}/mcp/configs" \
    -H "Authorization: Bearer ${TOKEN}")
  
  # Check for fetch tool
  FETCH_CONFIG=$(echo $MCP_CONFIGS | jq '[.[] | select(.name == "fetch")] | first')
  
  if [[ "$FETCH_CONFIG" != "null" ]]; then
    FETCH_CONFIG_ID=$(echo $FETCH_CONFIG | jq -r '.id')
    echo -e "${YELLOW}Found existing fetch tool configuration with ID: ${FETCH_CONFIG_ID}${NC}"
    echo -e "Current configuration:"
    echo $FETCH_CONFIG | jq .
    
    # Ask if user wants to update it
    read -p "Do you want to update this configuration? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
      update_fetch_tool "$FETCH_CONFIG_ID"
    else
      echo -e "${GREEN}Keeping existing configuration${NC}"
    fi
  else
    echo -e "${YELLOW}No fetch tool configuration found. Creating a new one...${NC}"
    create_fetch_tool
  fi
}

# Step 3: Create fetch tool configuration
create_fetch_tool() {
  echo -e "${BLUE}Step 3: Creating fetch tool configuration...${NC}"
  
  # Create a temporary file for the JSON payload
  PAYLOAD_FILE=$(mktemp)
  cat > $PAYLOAD_FILE << EOF
{
  "name": "fetch",
  "command": "docker",
  "args": ["run", "-i", "--rm", "--dns", "8.8.8.8", "--dns", "8.8.4.4", "mcp/fetch"],
  "enabled": true
}
EOF
  
  CREATE_CONFIG=$(curl -s -X POST "${API_BASE_URL}/mcp/configs" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer ${TOKEN}" \
    --data @$PAYLOAD_FILE)
  
  # Clean up
  rm $PAYLOAD_FILE
  
  CONFIG_ID=$(echo $CREATE_CONFIG | jq -r '.id')
  
  if [[ -z "$CONFIG_ID" || "$CONFIG_ID" == "null" ]]; then
    echo -e "${RED}Failed to create fetch tool configuration. Response:${NC}"
    echo $CREATE_CONFIG | jq .
    exit 1
  fi
  
  echo -e "${GREEN}Successfully created fetch tool configuration with ID: ${CONFIG_ID}${NC}"
  echo -e "Configuration details:"
  echo $CREATE_CONFIG | jq .
}

# Step 4: Update fetch tool configuration 
update_fetch_tool() {
  CONFIG_ID=$1
  echo -e "${BLUE}Step 4: Updating fetch tool configuration with ID: ${CONFIG_ID}...${NC}"
  
  # Create a temporary file for the JSON payload - this matches the format the backend expects
  PAYLOAD_FILE=$(mktemp)
  cat > $PAYLOAD_FILE << EOF
{
  "args": ["run", "-i", "--rm", "--dns", "8.8.8.8", "--dns", "8.8.4.4", "mcp/fetch"],
  "enabled": true
}
EOF
  
  echo "Debug - Update payload: $(cat $PAYLOAD_FILE)"
  
  UPDATE_CONFIG=$(curl -v -X PUT "${API_BASE_URL}/mcp/configs/${CONFIG_ID}" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer ${TOKEN}" \
    --data @$PAYLOAD_FILE)
  
  # Clean up
  rm $PAYLOAD_FILE
  
  echo "Debug - Update response: ${UPDATE_CONFIG}"
  
  if [[ $(echo $UPDATE_CONFIG | jq -r '.id') != "$CONFIG_ID" ]]; then
    echo -e "${RED}Failed to update fetch tool configuration. Response:${NC}"
    echo $UPDATE_CONFIG | jq .
  fi
  
  echo -e "${GREEN}Successfully updated fetch tool configuration${NC}"
  echo -e "Updated configuration details:"
  echo $UPDATE_CONFIG | jq .
  
  # Verify the configuration was saved by getting it again
  VERIFY_CONFIG=$(curl -s -X GET "${API_BASE_URL}/mcp/configs/${CONFIG_ID}" \
    -H "Authorization: Bearer ${TOKEN}")
  
  echo -e "${YELLOW}Verifying configuration was properly updated:${NC}"
  echo $VERIFY_CONFIG | jq .
}

# Run the setup
check_dependencies
login
check_fetch_tool

echo -e "${GREEN}Fetch tool configuration setup complete!${NC}"
echo "=================================="
echo "Now you can use the fetch tool in your chats."
echo "Try asking the LLM to fetch a URL like:"
echo "- 'Use your fetch tool to get the content from https://example.com'"
echo "- 'I need you to access the URL https://example.com using the fetch tool'"
echo "- 'Retrieve the content from https://example.com with the fetch tool and summarize it'"
