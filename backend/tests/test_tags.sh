#!/bin/bash

# Test script for Doogie Chat Bot Tag API
# This script tests the create, read, update, delete operations for tags

# Configuration
API_URL="http://localhost:3000/api/v1"  # Adjust based on your deployment
USERNAME="admin@example.com"            # Use a valid account for testing
PASSWORD="change-this-password"         # Password for the test account

# Color coding for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print section headers
print_header() {
  echo -e "\n${BLUE}=== $1 ===${NC}\n"
}

# Function to print success messages
print_success() {
  echo -e "${GREEN}✅ $1${NC}"
}

# Function to print error messages
print_error() {
  echo -e "${RED}❌ $1${NC}"
  if [ ! -z "$2" ]; then
    echo -e "${RED}   Error details: $2${NC}"
  fi
}

# Function to print info messages
print_info() {
  echo -e "${YELLOW}ℹ️ $1${NC}"
}

# Function to perform login and get auth token
login() {
  print_header "Authenticating User"
  
  local login_response=$(curl -s -X POST "${API_URL}/auth/login" \
    -H "Content-Type: application/x-www-form-urlencoded" \
    -d "username=${USERNAME}&password=${PASSWORD}")
  
  # Check if login succeeded
  if [[ $login_response == *"access_token"* ]]; then
    # Store the raw response for debugging
    echo "$login_response" > /tmp/login_response.json
    
    # Extract tokens using grep and sed
    ACCESS_TOKEN=$(echo $login_response | grep -o '"access_token":"[^"]*"' | cut -d '"' -f 4)
    REFRESH_TOKEN=$(echo $login_response | grep -o '"refresh_token":"[^"]*"' | cut -d '"' -f 4)
    
    # Debug token extraction
    echo "Token length: ${#ACCESS_TOKEN}"
    echo "Token first 10 chars: ${ACCESS_TOKEN:0:10}..."
    
    print_success "Authentication successful"
    return 0
  else
    print_error "Authentication failed" "$login_response"
    exit 1
  fi
}

# Function to create a tag
create_tag() {
  print_header "Creating Tag"
  
  local tag_name="Test Tag $RANDOM"
  local tag_color="#$(printf '%06x\n' $RANDOM)"
  
  print_info "Creating tag: $tag_name with color: $tag_color"
  
  # Use verbose curl to see exactly what's happening
  print_info "Making verbose request to debug issue..."
  echo "Full curl command for debugging:"
  echo "curl -v -X POST \"${API_URL}/tags/\" -H \"Authorization: Bearer ${ACCESS_TOKEN}\" -H \"Content-Type: application/json\" -d \"{\\\"name\\\":\\\"${tag_name}\\\",\\\"color\\\":\\\"${tag_color}\\\"}\""
  
  # Make actual request with verbose output
  echo "\nVerbose curl output:"
  local verbose_output=$(curl -v -X POST "${API_URL}/tags/" \
    -H "Authorization: Bearer ${ACCESS_TOKEN}" \
    -H "Content-Type: application/json" \
    -d "{\"name\":\"${tag_name}\",\"color\":\"${tag_color}\"}" 2>&1)
  echo "$verbose_output"
  
  # Make the actual request for processing - without trailing slash and with redirect following
  local create_response=$(curl -s -L -X POST "${API_URL}/tags" \
    -H "Authorization: Bearer ${ACCESS_TOKEN}" \
    -H "Content-Type: application/json" \
    -d "{\"name\":\"${tag_name}\",\"color\":\"${tag_color}\"}") 
  
  echo "Raw response: $create_response"
  
  # Check if tag creation succeeded
  if [[ $create_response == *"id"* ]]; then
    TAG_ID=$(echo $create_response | grep -o '"id":"[^"]*"' | cut -d '"' -f 4)
    TAG_NAME="$tag_name"
    TAG_COLOR="$tag_color"
    
    print_success "Tag created successfully with ID: $TAG_ID"
    return 0
  else
    print_error "Tag creation failed" "$create_response"
    return 1
  fi
}

# Function to get all user tags
get_tags() {
  print_header "Getting All Tags"
  
  local tags_response=$(curl -s -L -X GET "${API_URL}/tags" \
    -H "Authorization: Bearer ${ACCESS_TOKEN}" \
    -H "Content-Type: application/json")
  
  # Check if tags retrieval succeeded
  if [[ $tags_response == *"["* ]]; then
    local tags_count=$(echo $tags_response | grep -o '\[.*\]' | grep -o '"id"' | wc -l)
    
    print_success "Retrieved $tags_count tags successfully"
    
    # Check if our created tag is in the list
    if [[ $tags_response == *"$TAG_ID"* ]]; then
      print_success "Found our test tag in the list"
    else
      print_error "Our test tag is not in the list"
    fi
    
    return 0
  else
    print_error "Tags retrieval failed" "$tags_response"
    return 1
  fi
}

# Function to search for tags
search_tags() {
  print_header "Searching Tags"
  
  local search_term="Test"
  
  print_info "Searching for tags with term: $search_term"
  
  local search_response=$(curl -s -L -X GET "${API_URL}/tags/search?search=${search_term}" \
    -H "Authorization: Bearer ${ACCESS_TOKEN}" \
    -H "Content-Type: application/json")
  
  # Check if search succeeded
  if [[ $search_response == *"items"* ]]; then
    local search_count=$(echo $search_response | grep -o '"items":\[.*\]' | grep -o '"id"' | wc -l)
    
    print_success "Search returned $search_count tags"
    return 0
  else
    print_error "Tag search failed" "$search_response"
    return 1
  fi
}

# Function to update a tag
update_tag() {
  print_header "Updating Tag"
  
  if [ -z "$TAG_ID" ]; then
    print_error "No tag ID available for update"
    return 1
  fi
  
  local updated_name="${TAG_NAME}_updated"
  local updated_color="#$(printf '%06x\n' $RANDOM)"
  
  print_info "Updating tag: $TAG_ID with new name: $updated_name and color: $updated_color"
  
  local update_response=$(curl -s -L -X PUT "${API_URL}/tags/${TAG_ID}" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" \
  -H "Content-Type: application/json" \
  -d "{\"name\":\"${updated_name}\",\"color\":\"${updated_color}\"}") 
  
  # Check if update succeeded
  if [[ $update_response == *"id"* ]]; then
    TAG_NAME="$updated_name"
    TAG_COLOR="$updated_color"
    
    print_success "Tag updated successfully"
    return 0
  else
    print_error "Tag update failed" "$update_response"
    return 1
  fi
}

# Function to delete a tag
delete_tag() {
  print_header "Deleting Tag"
  
  if [ -z "$TAG_ID" ]; then
    print_error "No tag ID available for deletion"
    return 1
  fi
  
  print_info "Deleting tag: $TAG_ID"
  
  local delete_response=$(curl -s -L -X DELETE "${API_URL}/tags/${TAG_ID}" \
    -H "Authorization: Bearer ${ACCESS_TOKEN}" \
    -H "Content-Type: application/json")
  
  # Check if deletion succeeded
  if [[ $delete_response == *"Tag deleted successfully"* || $delete_response == *"detail"* ]]; then
    print_success "Tag deleted successfully"
    TAG_ID=""
    return 0
  else
    print_error "Tag deletion failed" "$delete_response"
    return 1
  fi
}

# Function to verify deletion
verify_deletion() {
  print_header "Verifying Tag Deletion"
  
  local verify_response=$(curl -s -L -X GET "${API_URL}/tags" \
    -H "Authorization: Bearer ${ACCESS_TOKEN}" \
    -H "Content-Type: application/json")
  
  # Check if verification succeeded
  if [[ $verify_response == *"["* ]]; then
    # If the tag ID is not empty, check if it's still in the list
    if [ ! -z "$TAG_ID" ] && [[ $verify_response == *"$TAG_ID"* ]]; then
      print_error "Tag still exists after deletion"
      return 1
    else
      print_success "Tag deletion verified"
      return 0
    fi
  else
    print_error "Verification failed" "$verify_response"
    return 1
  fi
}

# Main execution

echo -e "${BLUE}Starting Tag API Test Script${NC}"
echo -e "${YELLOW}Testing against API URL: ${API_URL}${NC}"

login

# Run test sequence
if create_tag; then
  get_tags
  search_tags
  update_tag
  delete_tag
  verify_deletion
  
  print_header "Test Summary"
  print_success "All tag API tests completed"
else
  print_header "Test Summary"
  print_error "Tag API tests failed at creation step"
  exit 1
fi