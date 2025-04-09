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

class OpenAIClient(LLMClient):
    """
    Client for OpenAI API.
    """

    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
        base_url: Optional[str] = "https://api.openai.com/v1",
        embedding_model: Optional[str] = None,
        user_id: Optional[str] = None # Add user_id parameter
    ):
        """
        Initialize the OpenAI client.
        """
        # Pass user_id to base class constructor
        super().__init__(model, api_key, base_url, embedding_model, user_id=user_id)
        self.api_key = api_key or settings.OPENAI_API_KEY
        if not self.api_key:
            raise ValueError("OpenAI API key is required")

        if self.base_url is None:
            self.base_url = "https://api.openai.com/v1"

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
        Generate a response from OpenAI.
        """
        if not self.base_url or not self.base_url.startswith("http"):
            self.base_url = "https://api.openai.com/v1"
        url = f"{self.base_url}/chat/completions"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        logger.info(f"Sending {len(messages)} messages to OpenAI. First message role: {messages[0]['role'] if messages else 'none'}")

        payload: Dict[str, Any] = {
            "model": self.model, "messages": messages, "temperature": temperature, "stream": stream
        }
        if max_tokens: payload["max_tokens"] = max_tokens
        if tools: payload["tools"] = tools
        if tool_choice: payload["tool_choice"] = tool_choice

        if settings.LLM_DEBUG_LOGGING:
            # Log payload details (simplified)
            log_payload = {k: v for k, v in payload.items() if k != 'messages'}
            logger.info(f"Other OpenAI Payload Params: {json.dumps(log_payload, indent=2)}")
            # Log message previews if needed

        start_time = time.time()

        if stream:
            return self._stream_response(url, headers, payload, start_time)
        else:
            # Non-streaming logic
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"OpenAI API error: {error_text}")
                        raise Exception(f"OpenAI API error: {response.status} - {error_text}")

                    result = await response.json()
                    logger.debug(f"OpenAI non-stream response: {result}")

                    usage = result.get("usage", {})
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("completion_tokens", 0)
                    total_tokens = prompt_tokens + completion_tokens
                    tokens_per_second = self.calculate_tokens_per_second(start_time, completion_tokens)

                    message = result["choices"][0]["message"]
                    content = message.get("content")
                    tool_calls = message.get("tool_calls") # Check for tool calls
                    finish_reason = result["choices"][0].get("finish_reason")

                    # If tool_calls are present, the finish reason should reflect that
                    if tool_calls:
                        finish_reason = "tool_calls"
                        logger.info(f"OpenAI returned tool calls: {tool_calls}")

                    response_data = {
                        "content": content,
                        "tool_calls": tool_calls,
                        "model": self.model,
                        "provider": "openai",
                        "tokens": total_tokens,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "tokens_per_second": tokens_per_second,
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
        """ Stream response from OpenAI, handling content and tool call deltas. """
        if not url.startswith("http"): # Ensure URL is valid
            if not self.base_url or not self.base_url.startswith("http"): self.base_url = "https://api.openai.com/v1"
            if url.startswith("/"): url = f"{self.base_url}{url}"
            else: url = f"{self.base_url}/{url}"

        logger.debug(f"Starting OpenAI streaming request to {url}")
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"OpenAI API error during stream connection: {error_text}")
                    yield {"type": "error", "error": f"OpenAI API error: {response.status} - {error_text}", "done": True}
                    return

                logger.debug(f"OpenAI streaming connection established")

                full_content = ""
                # --- Tool Call Accumulation Logic ---
                # Stores partially built tool calls, keyed by index
                current_tool_calls: Dict[int, Dict[str, Any]] = {}
                # ---

                prompt_tokens = 0 # Cannot get reliably from stream
                completion_tokens = 0 # Estimate based on chunks
                finish_reason = None
                model_name = self.model
                chunk_count = 0

                # Yield start chunk (synthesized)
                yield {"type": "start", "role": "assistant", "model": model_name}

                logger.debug(f"Processing OpenAI stream")
                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    if not line or line == "data: [DONE]":
                        if line == "data: [DONE]": logger.debug("Received [DONE]")
                        continue
                    if line.startswith("data: "): line = line[6:]

                    try:
                        if not line.strip(): continue
                        data = json.loads(line)
                        chunk_count += 1
                        choice = data.get("choices", [{}])[0]
                        delta = choice.get("delta", {})
                        # Finish reason can be in the last delta chunk or the choice itself
                        current_finish_reason = choice.get("finish_reason") or delta.get("finish_reason")
                        if current_finish_reason:
                             finish_reason = current_finish_reason # Store the latest non-null reason

                        delta_content = delta.get("content")
                        delta_tool_calls = delta.get("tool_calls")

                        yield_chunk: Dict[str, Any] = {"type": "delta", "done": False}
                        has_update = False

                        if delta_content:
                            full_content += delta_content
                            yield_chunk["content"] = delta_content
                            has_update = True

                        if delta_tool_calls:
                            has_update = True
                            yield_chunk["tool_calls_delta"] = delta_tool_calls # Yield the raw delta
                            # --- Accumulate Tool Call Deltas ---
                            for tool_call_delta in delta_tool_calls:
                                index = tool_call_delta.get("index")
                                if index is None: continue # Should always have index

                                if index not in current_tool_calls:
                                    # Initialize structure for this tool call index
                                    current_tool_calls[index] = {
                                        "id": None,
                                        "type": "function",
                                        "function": {"name": None, "arguments": ""}
                                    }

                                call_part = current_tool_calls[index]
                                if tool_call_delta.get("id"):
                                    call_part["id"] = tool_call_delta["id"]
                                if tool_call_delta.get("type"):
                                    call_part["type"] = tool_call_delta["type"] # Usually 'function'

                                func_delta = tool_call_delta.get("function", {})
                                if func_delta.get("name"):
                                    call_part["function"]["name"] = func_delta["name"]
                                if func_delta.get("arguments"):
                                    call_part["function"]["arguments"] += func_delta["arguments"]
                            # ---

                        if has_update:
                            yield yield_chunk

                        if finish_reason:
                            logger.debug(f"Finish reason '{finish_reason}' received in chunk {chunk_count}")
                            # Don't break immediately, process potential final content/tool calls
                            # The [DONE] message will terminate the loop

                    except json.JSONDecodeError:
                        logger.warning(f"Could not parse OpenAI stream line: {line}")
                    except Exception as e:
                        logger.error(f"Error processing OpenAI stream chunk: {e}")
                        yield {"type": "error", "error": str(e), "done": True}
                        return

                # --- Final Chunk Processing ---
                logger.debug(f"OpenAI stream processing complete after {chunk_count} delta chunks.")

                completion_tokens = chunk_count # Very rough estimate
                tokens_per_second = self.calculate_tokens_per_second(start_time, completion_tokens)

                # Convert accumulated tool calls dict to list
                final_tool_calls = [current_tool_calls[i] for i in sorted(current_tool_calls.keys())] if current_tool_calls else None

                final_yield: Dict[str, Any] = {
                    "type": "final",
                    "done": True,
                    "content": full_content if full_content else None,
                    "tool_calls": final_tool_calls, # Send fully accumulated calls
                    "model": model_name,
                    "provider": "openai",
                    "usage": { # Estimated usage
                        "prompt_tokens": prompt_tokens, # Unknown from stream
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens
                    },
                    "tokens_per_second": tokens_per_second,
                    "finish_reason": finish_reason or "stop" # Use detected reason or default
                }
                yield {k: v for k, v in final_yield.items() if v is not None}
                logger.debug(f"OpenAI streaming yielded final chunk.")


    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """ Get embeddings for a list of texts using OpenAI. """
        if not self.base_url or not self.base_url.startswith("http"): self.base_url = "https://api.openai.com/v1"
        url = f"{self.base_url}/embeddings"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        embedding_model = self.embedding_model or "text-embedding-ada-002"
        logger.info(f"Generating embeddings for {len(texts)} texts using OpenAI model: {embedding_model}")

        try:
            payload = {"model": embedding_model, "input": texts}
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"OpenAI API error generating embeddings: {error_text}")
                        dimension = 1536
                        if "3-small" in embedding_model: dimension = 1536
                        elif "3-large" in embedding_model: dimension = 3072
                        return [[0.0] * dimension for _ in range(len(texts))]

                    result = await response.json()
                    embeddings = [item["embedding"] for item in result["data"]]
                    logger.info(f"Successfully generated {len(embeddings)} embeddings with OpenAI")
                    if embeddings: logger.debug(f"Embedding dimensions: {len(embeddings[0])}")
                    return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings with OpenAI: {str(e)}")
            logger.exception("Detailed embedding generation error:")
            dimension = 1536
            if "3-small" in embedding_model: dimension = 1536
            elif "3-large" in embedding_model: dimension = 3072
            return [[0.0] * dimension for _ in range(len(texts))]