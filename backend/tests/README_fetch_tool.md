# Fetch Tool API Test Script

This script tests the functionality of the Fetch tool in the Doogie Chat Bot API. It verifies that the LLM can properly recognize a request to fetch a URL, make the tool call, and respond with the results.

## Purpose

The main purpose of this script is to test that:

1. The authentication and chat creation endpoints work correctly
2. The LLM correctly identifies when to use the Fetch tool
3. The tool call is made with proper parameters
4. The response from the tool is correctly used in the final answer

## Prerequisites

- The Doogie Chat Bot server must be running on `http://localhost:8000`
- `jq` must be installed (for JSON parsing)
- The default admin account must be available

## Usage

1. Make the script executable:
   ```
   chmod +x test_fetch_tool.sh
   ```

2. Run the script:
   ```
   ./test_fetch_tool.sh
   ```

## How It Works

The script follows these steps:

1. Logs in with admin credentials to get an auth token
2. Creates a new chat with the title "Fetch Tool Test"
3. Sends a message that should trigger the fetch tool: "use fetch to get the URL https://example.com"
4. Processes the streaming response looking for tool call events
5. Verifies that a complete response was saved to the chat

## Customization

You can modify these variables at the top of the script:

- `API_BASE_URL`: The base URL of the API
- `EMAIL`: Admin email address
- `PASSWORD`: Admin password
- `TEST_URL`: The URL to fetch in the test

## Troubleshooting

If the test fails:

1. Check that your server is running and accessible
2. Verify that the admin credentials are correct
3. Ensure that the LLM is properly configured to use tools
4. Check that the Fetch tool is available in your tools configuration
5. Look for error messages in the server logs

## Notes

This script is designed for testing during development and should not be used in production environments with sensitive credentials.
