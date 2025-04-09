from typing import Any, Dict # Keep Dict import just in case
import json
import logging
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.base import get_db
from app.models.user import User
from app.utils.deps import get_current_user
from app.schemas.chat import (
    MessageCreate,
    MessageResponse, # Use MessageResponse as response_model again
    ToolRetryRequest,
)
from app.services.chat import ChatService
from app.services.llm_service import LLMService
from app.services.mcp_config_service import MCPConfigService

# Set up logging
logger = logging.getLogger(__name__)

router = APIRouter()

# Revert response_model to MessageResponse
@router.post("/{chat_id}/llm", response_model=MessageResponse)
async def send_to_llm(
    chat_id: str,
    message_in: MessageCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Send a message to the LLM and get a response (non-streaming).
    Returns the final assistant message saved to the database after
    the full turn (including potential tool execution) completes.
    """
    chat = ChatService.get_chat(db, chat_id)
    if not chat:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat not found",
        )

    # Check if user owns the chat
    if chat.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions",
        )

    # Save the user message first
    # The test script expects this behavior for non-streaming calls.
    # Use add_message and extract relevant fields from message_in
    user_message_db = ChatService.add_message(
        db=db,
        chat_id=chat_id,
        role=message_in.role,
        content=message_in.content,
        # Pass optional tool-related fields if they exist in the input
        tool_calls=message_in.tool_calls,
        tool_call_id=message_in.tool_call_id,
        name=message_in.name
    )
    if not user_message_db:
         raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save user message",
        )

    # Initialize LLM service, passing user_id
    llm_service = LLMService(db, user_id=current_user.id) # <-- Pass user_id

    # Send message to LLM and wait for the internal process (including tool calls) to complete.
    # The return value of the service call is ignored here.
    await llm_service.chat(
        chat_id=chat_id,
        user_message=message_in.content,
        use_rag=True, # Or determine based on request/config
        stream=False
    )

    # NOTE: Tool calls are handled directly in the LLM service for non-streaming requests.
    # We now fetch the final result from the database.

    # --- Restore database fetch logic ---
    # Get the last message (assistant's response)
    messages = ChatService.get_messages(db, chat_id)
    if not messages:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get assistant response after LLM call",
        )

    # Return the assistant's message
    for message in reversed(messages):
        # Look for the *last* assistant message, which should be the final response
        # after any tool calls.
        if message.role == "assistant":
            # Ensure context_documents is correctly formatted if needed
            # (This check might be redundant if DB schema/service handles it)
            if message.context_documents is not None and not isinstance(message.context_documents, list):
                message.context_documents = [] # Or handle conversion if format is known
            return message # Return the Message ORM model, FastAPI handles serialization

    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Failed to find final assistant response in database",
    )
    # --- End Restore ---

@router.post("/{chat_id}/retry-tool", response_model=MessageResponse)
async def retry_tool_call(
    chat_id: str,
    tool_data: ToolRetryRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Retry a failed tool call.
    
    This endpoint allows retrying a specific tool call that may have failed previously.
    It executes the tool with the same arguments and returns the new tool result message.
    """
    # Check if the chat exists and belongs to the user
    chat = ChatService.get_chat(db, chat_id)
    if not chat:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat not found",
        )

    # Check if user owns the chat
    if chat.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions",
        )
    
    try:
        # Parse the arguments string to ensure it's valid JSON
        try:
            arguments = json.loads(tool_data.arguments)
            arguments_str = tool_data.arguments
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid JSON in arguments",
            )
        
        # Extract server name from function_name if it uses the prefix format
        function_name = tool_data.function_name
        server_name_prefix = None
        
        if "__" in function_name:
            server_name_prefix = function_name.split("__")[0]
        
        # Get all enabled MCP configs for this user
        enabled_configs = [
            c for c in MCPConfigService.get_configs_by_user(db, current_user.id) 
            if c.config and c.config.get('enabled', False)
        ]
        
        # Find the matching config
        config_id = None
        for config in enabled_configs:
            if server_name_prefix and config.name.replace('-', '_') == server_name_prefix:
                config_id = config.id
                break
        
        if not config_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"MCP server configuration for tool '{function_name}' not found",
            )
        
        # Execute the tool call
        tool_result = await MCPConfigService.execute_mcp_tool(
            db=db,
            config_id=config_id,
            tool_call_id=tool_data.tool_call_id,
            tool_name=function_name,
            arguments_str=arguments_str
        )
        
        if not tool_result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to execute tool call",
            )
        
        tool_result_content = tool_result.get("result", "{}")
        
        # Create a new tool result message
        new_message = ChatService.add_message(
            db=db,
            chat_id=chat_id,
            role="tool",
            content=tool_result_content,
            tool_call_id=tool_data.tool_call_id,
            name=function_name
        )
        
        return new_message
        
    except Exception as e:
        logger.exception(f"Error retrying tool call: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )