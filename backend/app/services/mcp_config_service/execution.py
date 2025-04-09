# backend/app/services/mcp_config_service/execution.py
"""
Handles executing tool calls on MCP servers, primarily via docker attach.
"""
import json
import logging
import asyncio
import os
import shutil
from typing import Any, Dict, Optional

# Import the session manager instance
from .manager import mcp_session_manager, MCP_SDK_AVAILABLE

# Conditional MCP SDK type imports for type hinting
if MCP_SDK_AVAILABLE:
    from mcp import types

logger = logging.getLogger(__name__)

from sqlalchemy.orm import Session
from app.db.base import SessionLocal
from .crud import get_config_by_id


# Old docker attach helper functions removed.

# Removed _execute_mcp_tool_via_sdk helper function.
# Logic is now integrated into execute_mcp_tool and the MCPSessionManager.


# --- Main Execution Function (called by background task) ---
# --- Main Execution Function (Refactored for Async and Session Manager) ---
async def execute_mcp_tool(
    config_id: str,
    tool_call_id: str, # Used for logging and potentially MCP request ID
    tool_name: str,
    arguments_str: str # Arguments as JSON string
) -> Dict[str, Any]: # Returns dict with 'result' key containing JSON string of response/error
    """
    Executes a specific tool using a managed MCP session.
    This function is intended to be run asynchronously (e.g., via FastAPI background task or directly).
    """
    logger.info(f"Executing tool '{tool_name}' via MCP config '{config_id}' (Call ID: {tool_call_id}) using Session Manager")

    # Check if SDK is available early
    if not MCP_SDK_AVAILABLE:
        logger.error("MCP SDK not available. Cannot execute tool.")
        return {"result": json.dumps({"error": {"code": -32000, "message": "MCP SDK (modelcontextprotocol) not installed."}})}

    # 1. Get Config (Requires DB Session)
    db_config_dict = None
    try:
        with SessionLocal() as db:
            db_config = get_config_by_id(db, config_id)
            if not db_config:
                logger.error(f"MCP configuration {config_id} not found in DB.")
                # Return error structure expected by caller (often wrapped in 'result')
                return {"result": json.dumps({"error": {"code": -32603, "message": "MCP configuration not found."}})}
            if not db_config.config or not db_config.config.get('enabled', False):
                logger.error(f"MCP server configuration {config_id} is disabled.")
                return {"result": json.dumps({"error": {"code": -32603, "message": "MCP server configuration is disabled."}})}
            db_config_dict = db_config.config # Get the config dict
    except Exception as db_err:
         logger.exception(f"Database error retrieving MCP config {config_id}: {db_err}")
         return {"result": json.dumps({"error": {"code": -32000, "message": f"Database error: {db_err}"}})}

    # 2. Parse Arguments (Keep this synchronous)
    try:
        arguments = json.loads(arguments_str)
        if not isinstance(arguments, dict): raise ValueError("Arguments must be a JSON object (dict)")
        logger.debug(f"Parsed arguments for tool '{tool_name}' (ID: {tool_call_id}): {arguments}")
    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"Invalid arguments for tool call {tool_call_id}: {e}")
        return {"result": json.dumps({"error": {"code": -32602, "message": f"Invalid tool arguments: {e}"}})}

    # 3. Get Session and Execute Tool (Async)
    session = None
    try:
        # Get session from manager (establishes connection if needed)
        session = await mcp_session_manager.get_session(config_id, db_config_dict)

        if not session:
            logger.error(f"Failed to get or establish MCP session for config {config_id}.")
            return {"result": json.dumps({"error": {"code": -32000, "message": "Failed to establish MCP session."}})}

        # Extract the actual MCP tool name (remove potential prefix if used)
        mcp_tool_name = tool_name.split("__")[-1] if "__" in tool_name else tool_name

        # Execute the tool call using the managed session
        logger.info(f"Calling tool '{mcp_tool_name}' via managed session (ID: {tool_call_id})")
        # Add retry logic here (Phase 3) - basic example for now
        max_retries = 1 # Example: 1 retry (total 2 attempts)
        attempt = 0
        last_exception = None
        while attempt <= max_retries:
            try:
                # Call the tool without the 'id' keyword argument
                result = await session.call_tool(mcp_tool_name, arguments=arguments)
                logger.info(f"Tool '{mcp_tool_name}' executed successfully via session (ID: {tool_call_id}).")
                logger.debug(f"Raw result from session.call_tool: {result}")

                # Process result: Extract text from CallToolResult's content list
                final_result_content = None
                if MCP_SDK_AVAILABLE and hasattr(result, 'content') and isinstance(result.content, list):
                    # Extract text from TextContent items within the list
                    text_parts = []
                    for item in result.content:
                        if isinstance(item, types.TextContent) and hasattr(item, 'text'):
                            text_parts.append(item.text)
                    final_result_content = "".join(text_parts)
                    logger.debug(f"Extracted text content from CallToolResult: '{final_result_content[:100]}...'")
                elif isinstance(result, (dict, list, str, int, float, bool)):
                    # Handle cases where the result might be a simple type directly (less common for call_tool)
                    final_result_content = result
                    logger.debug(f"Result is a simple type: {type(result)}")
                else:
                    # Fallback for unexpected result types
                    logger.warning(f"Unexpected result type from session.call_tool: {type(result)}. Converting raw result to string.")
                    final_result_content = str(result) # Use the string representation of the whole result object

                # Return success structure (wrapping the extracted content)
                return {"result": json.dumps({"result": final_result_content})}

            except Exception as e:
                last_exception = e
                attempt += 1
                logger.warning(f"Attempt {attempt}/{max_retries + 1} failed for tool '{mcp_tool_name}' (ID: {tool_call_id}): {e}")
                if attempt <= max_retries:
                    await asyncio.sleep(0.5 * attempt) # Simple backoff
                    logger.info(f"Retrying tool call...")
                else:
                    logger.error(f"Max retries reached for tool '{mcp_tool_name}' (ID: {tool_call_id}).")
                    raise last_exception # Re-raise after retries exhausted

        # This part should not be reached if loop completes normally or raises
        raise RuntimeError("Tool execution loop finished unexpectedly.")

    except Exception as e:
        logger.exception(f"Error during managed MCP tool execution for '{tool_name}' (ID: {tool_call_id}): {e}")
        # Note: If the error was due to a bad connection, the session manager's health check
        # should handle cleanup on the *next* call attempt. We don't explicitly close here.
        return {"result": json.dumps({"error": {"code": -32000, "message": f"Error executing tool '{tool_name}': {e}"}})}

# Old _execute_mcp_tool_logic function removed.