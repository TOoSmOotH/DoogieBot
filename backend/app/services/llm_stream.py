import logging
import asyncio
import time
from typing import List, Dict, Any, Optional, AsyncGenerator
from sqlalchemy.orm import Session

from app.llm.base import LLMClient
# Import specific clients for type checking if needed
from app.llm.anthropic_client import AnthropicClient
from app.llm.google_gemini_client import GoogleGeminiClient
from app.services.chat import ChatService
from app.core.config import settings # For default model

logger = logging.getLogger(__name__)

async def stream_llm_response(
    db: Session,
    chat_client: LLMClient,
    chat_id: str,
    formatted_messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: Optional[int],
    context_documents: Optional[List[Dict[str, Any]]],
    system_prompt: Optional[str], # Added system_prompt
    model: Optional[str], # Added model
    provider: Optional[str], # Added provider
    tools: Optional[List[Dict[str, Any]]] = None # <-- Add tools parameter
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Stream response from the LLM and save the final message.

    Args:
        db: Database session.
        chat_client: The LLM client instance to use for generation.
        chat_id: Chat ID.
        formatted_messages: Formatted messages for the LLM.
        temperature: Temperature for generation.
        max_tokens: Maximum number of tokens to generate.
        context_documents: Context documents from RAG.
        system_prompt: The system prompt to use.
        model: The model name being used.
        provider: The provider name being used.
        tools: Optional list of tool definitions to pass to the LLM.

    Yields:
        Chunks of the response.
    """
    logger.debug(f"Starting streaming response for chat {chat_id}")

    # Initialize variables for tracking the response
    full_content = ""
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    tokens_per_second = 0.0
    finish_reason = None
    # Use provided model/provider or fall back to defaults if needed (though should be set by caller)
    current_model = model or settings.DEFAULT_CHAT_MODEL
    current_provider = provider
    chunk_count = 0
    error_occurred = False
    error_message = ""

    try:
        # Get streaming response from LLM client
        logger.debug(f"Requesting streaming response from LLM client for chat {chat_id}")

        # Prepare arguments for the generate call
        generate_args = {
            "messages": formatted_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
            "tools": tools, # <-- Pass tools if provided
        }
        # Conditionally add system_prompt for clients that support it as a separate argument
        if isinstance(chat_client, (AnthropicClient, GoogleGeminiClient)):
            generate_args["system_prompt"] = system_prompt
            logger.debug("Passing system_prompt as a separate argument to generate()")
        else:
             # For other clients (like OpenAI), system prompt is expected in messages list
             logger.debug("System prompt included in messages list for generate()")

        # Await the generate call to get the async generator
        response_stream = await chat_client.generate(**generate_args)

        logger.debug(f"Got response_stream generator from LLM client")
        logger.debug(f"Starting to iterate through response_stream chunks")

        async for chunk in response_stream:
            chunk_count += 1
            chunk_type = chunk.get("type")

            logger.debug(f"Received chunk {chunk_count} of type '{chunk_type}' for chat {chat_id}")

            # Yield the raw chunk immediately
            yield chunk

            # Accumulate content from delta chunks
            if chunk_type == "delta":
                delta_content = chunk.get("content", "")
                if delta_content:
                    full_content += delta_content

            # Check for error chunk first
            if chunk_type == "error":
                error_message = chunk.get("error", "Unknown streaming error")
                logger.error(f"Error received during stream for chat {chat_id}: {error_message}")
                error_occurred = True
                full_content = f"An error occurred: {error_message}" # Use error as content

                # Save the error message (already yielded the error chunk)
                logger.debug(f"Saving error message for chat {chat_id} after yielding error chunk.")
                ChatService.add_message(
                    db,
                    chat_id,
                    "assistant",
                    full_content, # Save error message
                    model=current_model,
                    provider=current_provider,
                    context_documents=[doc["id"] for doc in context_documents] if context_documents else None,
                )
                logger.debug(f"Error message saved for chat {chat_id}")
                break # Exit loop on error

            # Check if this is the final chunk based on the 'done' flag
            if chunk.get("done") is True:
                logger.debug(f"Final chunk received (done=True) for chat {chat_id}. Processing final state.")
                logger.debug(f"Raw final chunk data: {chunk}") # Log the entire final chunk

                # Gather final metadata from this chunk
                usage = chunk.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                completion_tokens = usage.get("completion_tokens", 0)
                total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)
                tokens_per_second = chunk.get("tokens_per_second", 0.0)
                finish_reason = chunk.get("finish_reason")
                tool_calls = chunk.get("tool_calls") # Check for tool calls
                current_model = chunk.get("model", current_model) # Update model if provided in final chunk
                current_provider = chunk.get("provider", current_provider) # Update provider if provided

                logger.debug(f"Final chunk details - Usage: {usage}, Finish Reason: {finish_reason}, Tool Calls: {tool_calls}")

                # Accumulate any final delta content within the 'done' chunk
                final_delta_content = chunk.get("content", "")
                if final_delta_content:
                     full_content += final_delta_content
                     logger.debug(f"Accumulated final delta content: '{final_delta_content[:50]}...'")

                # Save the final accumulated message BEFORE breaking
                logger.debug(f"Preparing to save final message for chat {chat_id} after processing final chunk.")
                logger.debug(f"Final message content to save: '{full_content[:100]}...'")
                logger.debug(f"Final metadata for saving: tokens={total_tokens}, tps={tokens_per_second}, model={current_model}, provider={current_provider}, finish_reason={finish_reason}")
                logger.debug(f"Context doc IDs for saving: {[doc['id'] for doc in context_documents] if context_documents else None}")
                # Note: Tool calls from the final chunk are not explicitly saved in ChatService.add_message here.
                # The 'internal_final_state' chunk yielded *should* contain them for the background task.
                ChatService.add_message(
                    db,
                    chat_id,
                    "assistant",
                    full_content, # Save accumulated content
                    tokens=total_tokens,
                    tokens_per_second=tokens_per_second,
                    model=current_model,
                    provider=current_provider,
                    context_documents=[doc["id"] for doc in context_documents] if context_documents else None,
                )
                logger.debug(f"Final message saved for chat {chat_id}")
                
                # Check if tool calls were detected in any chunk
                # The tool_calls variable might be None even if tool calls were detected in delta chunks
                # So we need to check if any tool calls were detected in the Ollama client
                client_tool_calls = getattr(chat_client, 'detected_tool_calls', None)
                
                # Use either the final chunk tool_calls or the ones detected by the client
                final_tool_calls = tool_calls or client_tool_calls
                
                # Log the tool calls for debugging
                logger.debug(f"Final tool calls for internal_final_state: {final_tool_calls}")
                
                # Get the user_id from the chat_client
                user_id = getattr(chat_client, 'user_id', None)
                
                # Generate internal_final_state chunk with the state needed for the background task
                internal_state = {
                    "full_content": full_content,
                    "tool_call_occurred": bool(final_tool_calls),
                    "final_tool_calls_list": final_tool_calls,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "finish_reason": finish_reason,
                    "first_stream_yielded_final": True,
                    "current_model": current_model,
                    "current_provider": current_provider,
                    "start_time": time.time(),  # Current time as a reference
                    "user_id": user_id,
                    "context_documents": context_documents,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "system_prompt": system_prompt,
                    "original_messages": formatted_messages,
                    "first_stream_completed_normally": True
                }
                
                # Yield the internal_final_state chunk
                yield {
                    "type": "internal_final_state",
                    "state": internal_state
                }
                
                # Chunk was already yielded at the top of the loop
                # yield chunk # Removed redundant yield
                break # Exit loop after saving final message

            # Process start chunk for initial metadata (if necessary)
            elif chunk_type == "start":
                prompt_tokens = chunk.get("usage", {}).get("prompt_tokens", 0)
                current_model = chunk.get("model", current_model)
                current_provider = chunk.get("provider", current_provider)

            # Add a small delay to prevent overwhelming the client, less frequently
            if chunk_count % 20 == 0:
                await asyncio.sleep(0.01)
            else:
                await asyncio.sleep(0) # Yield control briefly

        # After loop completion
        if not error_occurred and finish_reason:
            logger.debug(f"Stream finished successfully for chat {chat_id}. Final chunk processed. Finish reason: {finish_reason}")
        elif error_occurred:
             logger.debug(f"Stream finished due to error for chat {chat_id}. Error message: {error_message}")
        else:
             logger.warning(f"Stream finished for chat {chat_id} without processing a final 'done' chunk or encountering a known error.")

    except asyncio.TimeoutError: # Handle timeout during initial generation call
        logger.error(f"Timeout occurred during initial generation call for chat {chat_id}")
        error_message = "The LLM took too long to respond initially. Please try again."
        error_chunk = {
            "content": error_message,
            "error": True,
            "done": True
        }
        yield error_chunk
        # Save the error message to the chat
        ChatService.add_message(
            db, chat_id, "assistant", error_message, model=current_model, provider=current_provider,
            context_documents=[doc["id"] for doc in context_documents] if context_documents else None
        )
        return # Stop the generator

    # Handle broader errors during the streaming process itself
    except Exception as e:
        logger.exception(f"Caught exception during streaming response for chat {chat_id}: {type(e).__name__} - {str(e)}") # Log exception type and message
        # Determine a user-friendly error message
        error_details = str(e)
        if "context length" in error_details.lower():
             error_message = "The request exceeded the model's context limit. Please try shortening your message or reducing the number of documents used."
        elif "rate limit" in error_details.lower():
             error_message = "The request was rate-limited by the AI provider. Please wait a moment and try again."
        else:
             error_message = f"An unexpected error occurred while generating the response: {error_details}"
        logger.debug(f"Generated user-facing error message: {error_message}")

        error_chunk = {
            "content": error_message,
            "error": True,
            "done": True
        }
        yield error_chunk

        # Save the error message to the chat
        ChatService.add_message(
            db,
            chat_id,
            "assistant",
            error_message,
            model=current_model,
            provider=current_provider,
            context_documents=[doc["id"] for doc in context_documents] if context_documents else None
        )