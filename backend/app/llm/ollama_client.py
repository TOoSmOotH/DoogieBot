from typing import List, Dict, Any, Optional, Union, AsyncGenerator
import aiohttp
import json
import time
import logging
import asyncio
import uuid # For tool call IDs

from app.llm.base import LLMClient
from app.core.config import settings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OllamaClient(LLMClient):
    """
    Client for Ollama API.
    """

    def __init__(
        self,
        model: str = "llama2",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        embedding_model: Optional[str] = None,
        user_id: Optional[str] = None # Add user_id parameter
    ):
        """
        Initialize the Ollama client.
        """
        # Pass user_id to base class constructor
        super().__init__(model=model, api_key=api_key, base_url=base_url, embedding_model=embedding_model, user_id=user_id)
        # Initialize detected_tool_calls to store tool calls detected during streaming
        self.detected_tool_calls = None
        self.base_url = base_url or settings.OLLAMA_BASE_URL
        if not self.base_url:
            logger.warning("Ollama base URL is not set. API calls will likely fail.")

    async def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None
    ) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        """
        Generate a response from Ollama.
        """
        url = f"{self.base_url}/api/chat"
        headers = {"Content-Type": "application/json"}

        # Convert messages to Ollama format
        formatted_messages = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")
            msg_tool_calls = msg.get("tool_calls") # Check for tool calls in assistant message
            tool_call_id = msg.get("tool_call_id") # Check for tool call ID in tool message

            if role == "tool":
                 # Format tool result message for Ollama
                 if tool_call_id and content:
                     formatted_messages.append({
                         "role": "tool",
                         "content": content, # Ollama expects content string for tool result
                         # Include tool_call_id if the model supports it (newer Ollama versions)
                         "tool_call_id": tool_call_id
                     })
                 else:
                      logger.warning(f"Skipping tool message due to missing content or tool_call_id: {msg}")
            elif role == "assistant" and msg_tool_calls:
                 # Format assistant message with tool calls
                 # Include content if it exists alongside tool calls
                 assistant_msg = {"role": "assistant", "tool_calls": msg_tool_calls}
                 if content:
                     assistant_msg["content"] = content
                 formatted_messages.append(assistant_msg)
            elif role and content: # Normal user/system/assistant message without tool calls
                formatted_messages.append({"role": role, "content": content})
            elif role == "assistant" and not content and not msg_tool_calls:
                 # Handle empty assistant message if needed (e.g., could be start of tool use)
                 # For now, we might skip it or add an empty content placeholder
                 logger.debug(f"Skipping empty assistant message without tool calls: {msg}")
            elif not role:
                 logger.warning(f"Skipping message due to missing role: {msg}")


        logger.info(f"Sending {len(formatted_messages)} messages to Ollama. First message role: {formatted_messages[0]['role'] if formatted_messages else 'none'}")

        payload: Dict[str, Any] = {
            "model": self.model, "messages": formatted_messages, "stream": stream,
            "options": {"temperature": temperature}
        }
        if max_tokens: payload["options"]["num_predict"] = max_tokens
        if tools: payload["tools"] = tools
        # Add tool_choice to the top level if tools are present
        if tools:
            # Default to "auto" if not provided or None, otherwise use the provided value
            effective_tool_choice = tool_choice if tool_choice is not None else "auto"
            payload["tool_choice"] = effective_tool_choice
            logger.info(f"Passing tool_choice='{effective_tool_choice}' in Ollama payload.")

        if tools:
            try:
                logger.debug(f"Sending tools payload to Ollama: {json.dumps(tools, indent=2)}") # Log the exact tools payload
            except Exception as e:
                logger.error(f"Error logging tools payload: {e}")
        if settings.LLM_DEBUG_LOGGING:
             try: logger.info(f"Ollama request payload:\n{json.dumps(payload, indent=2)}")
             except Exception as e: logger.error(f"Error logging Ollama payload: {e}")

        start_time = time.time()

        if stream:
            return self._stream_response(url, headers, payload, start_time)
        else:
            # Non-streaming logic
            timeout = aiohttp.ClientTimeout(total=60) # Explicit 60-second timeout
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Ollama API error: {error_text}")
                        raise Exception(f"Ollama API error: {response.status} - {error_text}")

                    result = await response.json()
                    logger.debug(f"Ollama non-stream response: {result}")

                    message_data = result.get("message", {})
                    content = message_data.get("content")
                    tool_calls = message_data.get("tool_calls") # Check for tool calls

                    prompt_tokens = result.get("prompt_eval_count", 0)
                    completion_tokens = result.get("eval_count", 0)
                    total_tokens = prompt_tokens + completion_tokens
                    tokens_per_second = self.calculate_tokens_per_second(start_time, completion_tokens)

                    # Determine finish reason
                    finish_reason = "stop"
                    if tool_calls:
                        finish_reason = "tool_calls"
                        logger.info(f"Ollama returned tool calls: {tool_calls}")
                    # Ollama doesn't explicitly return 'length' reason in non-streaming /api/chat
                    # We might infer it if content is cut off and completion_tokens > 0?

                    response_data = {
                        "content": content, "tool_calls": tool_calls,
                        "model": self.model, "provider": "ollama",
                        "tokens": total_tokens, "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens, "tokens_per_second": tokens_per_second,
                        "finish_reason": finish_reason
                    }
                    return {k: v for k, v in response_data.items() if v is not None}

    async def _stream_response(
        self,
        url: str,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        start_time: float
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """ Stream response from Ollama. """
        timeout = aiohttp.ClientTimeout(total=60) # Explicit 60-second timeout
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Ollama API stream error: {error_text}")
                    yield {"type": "error", "error": f"Ollama API error: {response.status} - {error_text}", "done": True}
                    return

                full_content = ""
                accumulated_tool_calls = []
                prompt_tokens = 0
                completion_tokens = 0
                total_tokens = 0
                finish_reason = "stop"
                chunk_count = 0
                
                # Reset detected_tool_calls at the start of each stream
                self.detected_tool_calls = None

                yield {"type": "start", "role": "assistant", "model": self.model}

                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    if not line: continue

                    try:
                        data = json.loads(line)
                        chunk_count += 1
                        yield_chunk: Dict[str, Any] = {"type": "delta", "done": False}
                        has_update = False

                        message_data = data.get("message", {})
                        delta_content = message_data.get("content", "")
                        # Check for tool calls in the message part of the chunk
                        delta_tool_calls = message_data.get("tool_calls")

                        if delta_content:
                            full_content += delta_content
                            yield_chunk["content"] = delta_content
                            has_update = True

                        if delta_tool_calls:
                            # Handle potentially incremental tool calls from Ollama stream
                            logger.debug(f"Ollama stream received tool calls delta: {delta_tool_calls}")
                            yield_chunk["tool_calls_delta"] = delta_tool_calls # Pass delta through
                            
                            # Add an ID to each tool call if it doesn't have one
                            for i, tool_call in enumerate(delta_tool_calls):
                                if not tool_call.get("id"):
                                    tool_call["id"] = f"call_{int(time.time())}_{i}"
                            
                            # Store the tool calls for access by llm_stream.py
                            self.detected_tool_calls = delta_tool_calls
                            logger.debug(f"Stored detected_tool_calls: {self.detected_tool_calls}")
                            
                            # Accumulate tool calls similar to llm_stream.py
                            for tool_call_delta in delta_tool_calls:
                                index = tool_call_delta.get("index")
                                if index is None: continue
                                if index not in accumulated_tool_calls:
                                    # Initialize structure if index is new
                                    accumulated_tool_calls.append({"id": None, "type": "function", "function": {"name": None, "arguments": ""}})
                                
                                # Ensure list is long enough (shouldn't happen if index is sequential, but safety check)
                                while len(accumulated_tool_calls) <= index:
                                     accumulated_tool_calls.append({"id": None, "type": "function", "function": {"name": None, "arguments": ""}})

                                call_part = accumulated_tool_calls[index]
                                if tool_call_delta.get("id"): call_part["id"] = tool_call_delta["id"]
                                if tool_call_delta.get("type"): call_part["type"] = tool_call_delta["type"]
                                func_delta = tool_call_delta.get("function", {})
                                if func_delta.get("name"): call_part["function"]["name"] = func_delta["name"]
                                if func_delta.get("arguments"): call_part["function"]["arguments"] += func_delta["arguments"]
                                
                            finish_reason = "tool_calls" # Mark that tools were involved
                            has_update = True

                        if has_update:
                             yield yield_chunk

                        # Check if this is the final message for the stream
                        if data.get("done", False):
                            prompt_tokens = data.get("prompt_eval_count", 0)
                            completion_tokens = data.get("eval_count", 0)
                            total_tokens = prompt_tokens + completion_tokens
                            # Check done_reason if available (newer Ollama versions)
                            done_reason = data.get("done_reason")
                            if done_reason == "stop" and not accumulated_tool_calls: finish_reason = "stop"
                            elif done_reason == "length": finish_reason = "length"
                            # If tool calls were detected earlier, finish_reason remains "tool_calls"
                            logger.debug("Ollama stream received done=True")
                            break

                    except json.JSONDecodeError:
                        logger.warning(f"Could not parse Ollama stream line: {line}")
                    except Exception as e:
                         logger.error(f"Error processing Ollama stream chunk: {e}")
                         yield {"type": "error", "error": str(e), "done": True}
                         return

                # Store accumulated tool calls for access by llm_stream.py
                if accumulated_tool_calls:
                    self.detected_tool_calls = accumulated_tool_calls
                    logger.debug(f"Stored detected tool calls: {self.detected_tool_calls}")
                
                # Yield final chunk
                tokens_per_second = self.calculate_tokens_per_second(start_time, completion_tokens)
                final_yield: Dict[str, Any] = {
                    "type": "final", "done": True,
                    "content": full_content if full_content else None,
                    "tool_calls": accumulated_tool_calls if accumulated_tool_calls else None,
                    "model": self.model, "provider": "ollama",
                    "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": total_tokens},
                    "tokens_per_second": tokens_per_second, "finish_reason": finish_reason
                }
                yield {k: v for k, v in final_yield.items() if v is not None}
                logger.debug("Ollama stream yielded final chunk.")


    async def list_models(self) -> List[str]:
        """ List available models from Ollama. """
        try:
            url = f"{self.base_url}/api/tags"
            timeout = aiohttp.ClientTimeout(total=60) # Explicit 60-second timeout
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Ollama API error listing models: {error_text}")
                        return []
                    result = await response.json()
                    models = [model["name"] for model in result.get("models", [])]
                    return models
        except Exception as e:
            logger.error(f"Error listing Ollama models: {str(e)}")
            return []

    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """ Get embeddings for a list of texts using Ollama. """
        model_to_use = self.embedding_model or self.model
        logger.info(f"Using model {model_to_use} for Ollama embeddings")
        url = f"{self.base_url}/api/embeddings"
        headers = {"Content-Type": "application/json"}
        embeddings = []
        logger.info(f"Generating embeddings for {len(texts)} texts using Ollama")

        for i, text in enumerate(texts):
            try:
                if i % 10 == 0: logger.info(f"Processing Ollama embedding {i+1}/{len(texts)}")
                payload = {"model": model_to_use, "prompt": text}
                logger.debug(f"Sending embedding request to Ollama for text {i+1}: {text[:50]}...")
                timeout = aiohttp.ClientTimeout(total=60) # Explicit 60-second timeout
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(url, headers=headers, json=payload) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            logger.error(f"Ollama API error for embedding text {i+1}: {error_text}")
                            embeddings.append([])
                            continue
                        result = await response.json()
                        embedding = result.get("embedding", [])
                        if not embedding:
                            logger.warning(f"Ollama returned empty embedding for text {i+1}")
                            embeddings.append([])
                        else:
                            logger.debug(f"Successfully generated Ollama embedding for text {i+1} with dimension {len(embedding)}")
                            embeddings.append(embedding)
            except Exception as e:
                logger.error(f"Error generating Ollama embedding for text {i+1}: {str(e)}")
                embeddings.append([])

        first_valid_embedding = next((e for e in embeddings if e), None)
        dimension = len(first_valid_embedding) if first_valid_embedding else 768
        final_embeddings = [e if e else [0.0] * dimension for e in embeddings]
        logger.info(f"Completed Ollama embedding generation: {len(final_embeddings)} embeddings generated.")
        return final_embeddings