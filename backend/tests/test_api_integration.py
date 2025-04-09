# backend/tests/test_api_integration.py

import requests
import time
import json
import uuid

# --- Configuration ---
BASE_URL = "http://localhost:8000/api/v1"
ADMIN_EMAIL = "admin@example.com"
ADMIN_PASSWORD = "change-this-password" # Default password

# --- Global State ---
access_token = None
refresh_token = None

# --- Helper Functions ---

def print_status(step: str, success: bool, details: str = ""):
    """Prints the status of a test step."""
    status = "✅ SUCCESS" if success else "❌ FAILED"
    print(f"{status} - {step}{': ' + details if details else ''}")

def login():
    """Logs in the admin user and stores tokens."""
    global access_token, refresh_token
    step = "Admin Login"
    try:
        response = requests.post(
            f"{BASE_URL}/auth/login",
            data={"username": ADMIN_EMAIL, "password": ADMIN_PASSWORD}
        )
        response.raise_for_status() # Raise exception for bad status codes
        data = response.json()
        access_token = data.get("access_token")
        refresh_token = data.get("refresh_token")
        if access_token and refresh_token:
            print_status(step, True, f"Token Type: {data.get('token_type')}")
            return True
        else:
            print_status(step, False, "Tokens not found in response")
            return False
    except requests.exceptions.RequestException as e:
        print_status(step, False, f"Error: {e}")
        return False

def get_auth_headers():
    """Returns headers with the current access token."""
    if not access_token:
        raise Exception("Not logged in. Run login() first.")
    return {"Authorization": f"Bearer {access_token}"}

# --- Test Functions ---

def test_health_check():
    step = "Health Check"
    try:
        response = requests.get(f"{BASE_URL}/health")
        response.raise_for_status()
        data = response.json()
        if data.get("status") == "healthy":
            print_status(step, True)
        else:
            print_status(step, False, f"Unexpected status: {data.get('status')}")
    except requests.exceptions.RequestException as e:
        print_status(step, False, f"Error: {e}")

def test_get_current_user():
    step = "Get Current User (/users/me)"
    try:
        headers = get_auth_headers()
        response = requests.get(f"{BASE_URL}/users/me", headers=headers)
        response.raise_for_status()
        data = response.json()
        if data.get("email") == ADMIN_EMAIL:
            print_status(step, True, f"User: {data.get('email')}, Role: {data.get('role')}")
        else:
            print_status(step, False, f"Unexpected user data: {data}")
    except Exception as e:
        print_status(step, False, f"Error: {e}")

def test_chat_crud():
    chat_id = None
    step = "Create Chat"
    try:
        headers = get_auth_headers()
        payload = {"title": "API Test Chat"}
        response = requests.post(f"{BASE_URL}/chats", headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        chat_id = data.get("id")
        if chat_id and data.get("title") == "API Test Chat":
            print_status(step, True, f"Chat ID: {chat_id}")
        else:
            print_status(step, False, f"Unexpected response: {data}")
            return # Stop if creation failed

        step = "Get Chats"
        response = requests.get(f"{BASE_URL}/chats", headers=headers)
        response.raise_for_status()
        chats = response.json()
        if any(c["id"] == chat_id for c in chats):
            print_status(step, True, f"Found {len(chats)} chats including the new one.")
        else:
            print_status(step, False, "Newly created chat not found in list.")

        step = "Get Specific Chat"
        response = requests.get(f"{BASE_URL}/chats/{chat_id}", headers=headers)
        response.raise_for_status()
        chat_data = response.json()
        if chat_data.get("id") == chat_id:
            print_status(step, True)
        else:
            print_status(step, False, f"Unexpected chat data: {chat_data}")

    except Exception as e:
        print_status(step, False, f"Error: {e}")
    finally:
        # Teardown: Delete the chat
        if chat_id:
            step = "Delete Chat"
            try:
                headers = get_auth_headers()
                response = requests.delete(f"{BASE_URL}/chats/{chat_id}", headers=headers)
                # Delete returns 200 OK with boolean true/false in FastAPI, check content
                if response.status_code == 200 and response.json() is True:
                     print_status(step, True, f"Chat ID: {chat_id}")
                elif response.status_code == 204: # Or maybe 204 No Content
                     print_status(step, True, f"Chat ID: {chat_id} (Status 204)")
                else:
                     print_status(step, False, f"Status: {response.status_code}, Body: {response.text}")
            except Exception as e:
                print_status(step, False, f"Error during cleanup: {e}")

def test_llm_interaction():
    chat_id = None
    step = "LLM Test - Create Chat"
    try:
        # Setup: Create a chat
        headers = get_auth_headers()
        payload = {"title": "LLM Test Chat"}
        response = requests.post(f"{BASE_URL}/chats", headers=headers, json=payload)
        response.raise_for_status()
        chat_id = response.json().get("id")
        if not chat_id: raise Exception("Failed to create chat for LLM test")
        print_status(step, True, f"Chat ID: {chat_id}")

        # Test non-streaming LLM call
        step = "LLM Test - Send Message (Non-Streaming)"
        llm_payload = {"role": "user", "content": "Hello LLM, how are you?"}
        # Use a longer timeout for LLM calls
        response = requests.post(f"{BASE_URL}/chats/{chat_id}/llm", headers=headers, json=llm_payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        if data.get("role") == "assistant" and data.get("content"):
            print_status(step, True, f"Received assistant response: {data['content'][:50]}...")
        else:
            print_status(step, False, f"Unexpected LLM response: {data}")

        # Test streaming LLM call (POST)
        step = "LLM Test - Send Message (Streaming POST)"
        llm_payload_stream = {"role": "user", "content": "Tell me a short story."}
        # Use a longer timeout for streaming initiation
        with requests.post(
            f"{BASE_URL}/chats/{chat_id}/stream",
            headers=headers,
            json=llm_payload_stream,
            stream=True,
            timeout=120
        ) as response:
            response.raise_for_status()
            stream_content = ""
            received_done = False
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith('data:'):
                        try:
                            data_str = decoded_line[len('data:'):].strip()
                            chunk = json.loads(data_str)
                            stream_content = chunk.get("content", stream_content) # Keep last content
                            if chunk.get("done"):
                                received_done = True
                        except json.JSONDecodeError:
                            print(f"Warning: Could not decode stream line: {decoded_line}")
            if received_done and stream_content:
                 print_status(step, True, f"Received streamed response ending with: ...{stream_content[-50:]}")
            elif stream_content:
                 print_status(step, False, f"Stream finished but 'done' flag not received. Last content: ...{stream_content[-50:]}")
            else:
                 print_status(step, False, "No content received from stream.")


    except Exception as e:
        print_status(step, False, f"Error: {e}")
    finally:
        # Teardown: Delete the chat
        if chat_id:
            step = "LLM Test - Delete Chat"
            try:
                headers = get_auth_headers()
                response = requests.delete(f"{BASE_URL}/chats/{chat_id}", headers=headers)
                if response.status_code == 200 and response.json() is True:
                     print_status(step, True, f"Chat ID: {chat_id}")
                elif response.status_code == 204:
                     print_status(step, True, f"Chat ID: {chat_id} (Status 204)")
                else:
                     print_status(step, False, f"Status: {response.status_code}, Body: {response.text}")
            except Exception as e:
                print_status(step, False, f"Error during cleanup: {e}")

def test_mcp_management():
    config_id = None
    step = "MCP Test - Create Config"
    try:
        headers = get_auth_headers()
        # Example config for testing (adjust image/args if needed)
        mcp_name = f"test-script-server-{uuid.uuid4()}"
        payload = {
            "name": mcp_name,
            "command": "docker",
            "args": ["run", "-i", "--rm", "hello-world"], # Simple test image
            "enabled": True
        }
        response = requests.post(f"{BASE_URL}/mcp/configs", headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        config_id = data.get("id")
        if config_id and data.get("name") == mcp_name:
            print_status(step, True, f"Config ID: {config_id}")
        else:
            print_status(step, False, f"Unexpected response: {data}")
            return # Stop if creation failed

        step = "MCP Test - List Configs"
        response = requests.get(f"{BASE_URL}/mcp/configs", headers=headers)
        response.raise_for_status()
        configs = response.json()
        if any(c["id"] == config_id for c in configs):
            print_status(step, True, f"Found {len(configs)} configs including the new one.")
        else:
            print_status(step, False, "Newly created MCP config not found in list.")

        step = "MCP Test - Get Specific Config"
        response = requests.get(f"{BASE_URL}/mcp/configs/{config_id}", headers=headers)
        response.raise_for_status()
        config_data = response.json()
        if config_data.get("id") == config_id:
            print_status(step, True)
        else:
            print_status(step, False, f"Unexpected config data: {config_data}")

        step = "MCP Test - Get Initial Status"
        response = requests.get(f"{BASE_URL}/mcp/configs/{config_id}/status", headers=headers)
        response.raise_for_status()
        status_data = response.json()
        if status_data.get("status") == "stopped": # Should be stopped initially
             print_status(step, True, f"Status: {status_data.get('status')}")
        else:
             print_status(step, False, f"Unexpected initial status: {status_data}")

        step = "MCP Test - Start Server"
        response = requests.post(f"{BASE_URL}/mcp/configs/{config_id}/start", headers=headers)
        response.raise_for_status()
        status_data = response.json()
        # Status might take a moment to update, check for 'running' or 'exited' (for hello-world)
        if status_data.get("status") in ["running", "stopped", "exited"]:
             print_status(step, True, f"Start command successful, status: {status_data.get('status')}")
             # Wait a bit for hello-world to potentially exit
             if status_data.get("status") == "running": time.sleep(3)
        else:
             print_status(step, False, f"Unexpected status after start: {status_data}")

        step = "MCP Test - Get Status After Start"
        response = requests.get(f"{BASE_URL}/mcp/configs/{config_id}/status", headers=headers)
        response.raise_for_status()
        status_data = response.json()
        if status_data.get("status") in ["running", "stopped", "exited"]: # hello-world exits quickly
             print_status(step, True, f"Status: {status_data.get('status')}")
        else:
             print_status(step, False, f"Unexpected status: {status_data}")

        step = "MCP Test - Stop Server"
        response = requests.post(f"{BASE_URL}/mcp/configs/{config_id}/stop", headers=headers)
        response.raise_for_status()
        status_data = response.json()
        # Stop should result in 'stopped' or handle already exited container
        if status_data.get("status") == "stopped":
             print_status(step, True, f"Stop command successful, status: {status_data.get('status')}")
        else:
             print_status(step, False, f"Unexpected status after stop: {status_data}")


    except Exception as e:
        print_status(step, False, f"Error: {e}")
    finally:
        # Teardown: Delete the config
        if config_id:
            step = "MCP Test - Delete Config"
            try:
                headers = get_auth_headers()
                response = requests.delete(f"{BASE_URL}/mcp/configs/{config_id}", headers=headers)
                if response.status_code == 204: # Expect 204 No Content on successful delete
                     print_status(step, True, f"Config ID: {config_id}")
                else:
                     print_status(step, False, f"Status: {response.status_code}, Body: {response.text}")
            except Exception as e:
                print_status(step, False, f"Error during cleanup: {e}")


def test_mcp_tool_call():
    """Tests LLM interaction involving an MCP tool call (fetch)."""
    # Assumption: 'fetch' MCP server is configured and enabled for the admin user.
    chat_id = None
    step = "MCP Tool Call - Create Chat"
    try:
        # Setup: Create a chat
        headers = get_auth_headers()
        payload = {"title": "MCP Tool Call Test Chat"}
        response = requests.post(f"{BASE_URL}/chats", headers=headers, json=payload)
        response.raise_for_status()
        chat_id = response.json().get("id")
        if not chat_id: raise Exception("Failed to create chat for MCP tool call test")
        print_status(step, True, f"Chat ID: {chat_id}")

        # Test non-streaming LLM call that should trigger the fetch tool
        step = "MCP Tool Call - Send Message (Trigger Fetch)"
        # Use a simple, reliable URL for testing
        test_url = "https://example.com"
        # Phrase the request clearly for the LLM
        message_content = f"Please fetch the content of the website {test_url} using the available tool."
        llm_payload = {"role": "user", "content": message_content}

        # Use a longer timeout as this involves LLM + tool execution + LLM again
        response = requests.post(f"{BASE_URL}/chats/{chat_id}/llm", headers=headers, json=llm_payload, timeout=180)
        response.raise_for_status()
        data = response.json()

        # Assertions
        if data.get("role") == "assistant" and data.get("content"):
            final_content = data['content'].lower()
            # Check for keywords expected from example.com
            if "example domain" in final_content and "illustrative examples" in final_content:
                 print_status(step, True, f"Received assistant response containing expected fetched content.")
            else:
                 print_status(step, False, f"Response content missing expected keywords from {test_url}. Content: {data['content'][:200]}...")
        else:
            print_status(step, False, f"Unexpected final LLM response after tool call: {data}")

    except Exception as e:
        print_status(step, False, f"Error: {e}")
    finally:
        # Teardown: Delete the chat
        if chat_id:
            step = "MCP Tool Call - Delete Chat"
            try:
                headers = get_auth_headers()
                response = requests.delete(f"{BASE_URL}/chats/{chat_id}", headers=headers)
                if response.status_code == 200 and response.json() is True:
                     print_status(step, True, f"Chat ID: {chat_id}")
                elif response.status_code == 204:
                     print_status(step, True, f"Chat ID: {chat_id} (Status 204)")
                else:
                     print_status(step, False, f"Status: {response.status_code}, Body: {response.text}")
            except Exception as e:
                print_status(step, False, f"Error during cleanup: {e}")


# --- Main Execution ---

if __name__ == "__main__":
    print("--- Starting API Integration Tests ---")

    if login():
        test_health_check()
        test_get_current_user()
        test_chat_crud()
        test_llm_interaction()
        test_mcp_management()
        test_mcp_tool_call() # Add the new test to the execution flow
    else:
        print("Login failed, skipping remaining tests.")

    print("--- API Integration Tests Finished ---")