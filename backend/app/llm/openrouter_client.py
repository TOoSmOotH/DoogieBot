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

class OpenRouterClient(LLMClient):
    """
    Client for OpenRouter API.
    """

    def __init__(
        self,
        model: str = "openai/gpt-3.5-turbo",
        api_key: Optional[str] = None,
        base_url: Optional[str] = "https://openrouter.ai/api", # Base URL is managed internally now
        embedding_model: Optional[str] = None,
        user_id: Optional[str] = None # Add user_id parameter
    ):
        """
        Initialize the OpenRouter client.
        """
        # OpenRouter base URL is fixed
        # Pass user_id to base class constructor
        super().__init__(model, api_key, "https://openrouter.ai/api/v1", embedding_model, user_id=user_id)
        self._current_reasoning = ""
        self._has_reasoning = False
        self.api_key = api_key or settings.OPENROUTER_API_KEY
        if not self.api_key:
            raise ValueError("OpenRouter API key is required")

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
        Generate a response from OpenRouter.
        """
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": getattr(settings, "OPENROUTER_REFERRER", "https://github.com/rooveterinary/doogie"),
            "X-Title": getattr(settings, "OPENROUTER_APP_TITLE", "Doogie")
        }
        logger.info(f"Sending {len(messages)} messages to OpenRouter. First message role: {messages[0]['role'] if messages else 'none'}")

        payload: Dict[str, Any] = {
            "model": self.model, "messages": messages, "temperature": temperature, "stream": stream
        }
        if max_tokens: payload["max_tokens"] = max_tokens
        if tools: payload["tools"] = tools
        if tool_choice: payload["tool_choice"] = tool_choice

        if settings.LLM_DEBUG_LOGGING:
            log_payload = {k: v for k, v in payload.items() if k != 'messages'}
            logger.info(f"Other OpenRouter Payload Params: {json.dumps(log_payload, indent=2)}")
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
                        logger.error(f"OpenRouter API error: {error_text}")
                        raise Exception(f"OpenRouter API error: {response.status} - {error_text}")

                    result = await response.json()
                    logger.debug(f"OpenRouter non-stream response: {result}")

                    usage = result.get("usage", {})
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("completion_tokens", 0)
                    total_tokens = prompt_tokens + completion_tokens
                    tokens_per_second = self.calculate_tokens_per_second(start_time, completion_tokens)

                    message = result["choices"][0]["message"]
                    content = message.get("content")
                    tool_calls = message.get("tool_calls") # Check for tool calls
                    finish_reason = result["choices"][0].get("finish_reason")

                    # Update finish_reason if tool_calls are present
                    if tool_calls:
                        finish_reason = "tool_calls"
                        logger.info(f"OpenRouter returned tool calls: {tool_calls}")

                    response_data = {
                        "content": content,
                        "tool_calls": tool_calls,
                        "model": self.model,
                        "provider": "openrouter",
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
        """ Stream response from OpenRouter, handling content and tool call deltas. """
        logger.debug(f"Starting OpenRouter streaming request to {url}")
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"OpenRouter API error during stream connection: {error_text}")
                    yield {"type": "error", "error": f"OpenRouter API error: {response.status} - {error_text}", "done": True}
                    return

                logger.debug(f"OpenRouter streaming connection established")

                full_content = ""
                # --- Tool Call Accumulation Logic ---
                current_tool_calls: Dict[int, Dict[str, Any]] = {}
                # ---

                prompt_tokens = 0
                completion_tokens = 0
                finish_reason = None
                model_name = self.model
                chunk_count = 0
                self._has_reasoning = False # Reset reasoning flag

                yield {"type": "start", "role": "assistant", "model": model_name}

                logger.debug(f"Processing OpenRouter stream")
                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    if not line or line == "data: [DONE]" or line.startswith(": OPENROUTER PROCESSING"):
                        if line == "data: [DONE]": logger.debug("Received [DONE]")
                        continue
                    if line.startswith("data: "): line = line[6:]

                    try:
                        if not line.strip(): continue
                        data = json.loads(line)
                        chunk_count += 1
                        choice = data.get("choices", [{}])[0]
                        delta = choice.get("delta", {})
                        current_finish_reason = choice.get("finish_reason") or delta.get("finish_reason")
                        if current_finish_reason: finish_reason = current_finish_reason

                        delta_content = delta.get("content")
                        delta_reasoning = delta.get("reasoning")
                        delta_tool_calls = delta.get("tool_calls") # Check for tool calls delta

                        yield_chunk: Dict[str, Any] = {"type": "delta", "done": False}
                        has_update = False

                        if delta_content and isinstance(delta_content, str):
                            full_content += delta_content
                            yield_chunk["content"] = delta_content
                            has_update = True

                        if delta_reasoning and isinstance(delta_reasoning, str):
                            self._has_reasoning = True
                            yield_chunk["content"] = (yield_chunk.get("content", "") or "") + f"<think>{delta_reasoning}</think>"
                            has_update = True

                        if delta_tool_calls:
                            has_update = True
                            yield_chunk["tool_calls_delta"] = delta_tool_calls
                            # --- Accumulate Tool Call Deltas ---
                            for tool_call_delta in delta_tool_calls:
                                index = tool_call_delta.get("index")
                                if index is None: continue

                                if index not in current_tool_calls:
                                    current_tool_calls[index] = {"id": None, "type": "function", "function": {"name": None, "arguments": ""}}

                                call_part = current_tool_calls[index]
                                if tool_call_delta.get("id"): call_part["id"] = tool_call_delta["id"]
                                if tool_call_delta.get("type"): call_part["type"] = tool_call_delta["type"]

                                func_delta = tool_call_delta.get("function", {})
                                if func_delta.get("name"): call_part["function"]["name"] = func_delta["name"]
                                if func_delta.get("arguments"): call_part["function"]["arguments"] += func_delta["arguments"]
                            # ---
                            if finish_reason is None or finish_reason == "stop": # Update finish reason if tool call detected
                                finish_reason = "tool_calls"

                        if has_update:
                            yield yield_chunk

                        if finish_reason and finish_reason != "tool_calls": # Don't break early if tool calls are streaming
                            logger.debug(f"Finish reason '{finish_reason}' received in chunk {chunk_count}")
                            # Wait for [DONE]

                    except json.JSONDecodeError:
                        logger.warning(f"Could not parse OpenRouter stream line: {line}")
                    except Exception as e:
                        logger.error(f"Error processing OpenRouter stream chunk: {e}")
                        yield {"type": "error", "error": str(e), "done": True}
                        return

                # --- Final Chunk Processing ---
                logger.debug(f"OpenRouter stream processing complete after {chunk_count} delta chunks.")

                completion_tokens = chunk_count # Rough estimate
                tokens_per_second = self.calculate_tokens_per_second(start_time, completion_tokens)
                total_tokens = prompt_tokens + completion_tokens

                final_tool_calls = [current_tool_calls[i] for i in sorted(current_tool_calls.keys())] if current_tool_calls else None

                final_yield: Dict[str, Any] = {
                    "type": "final", "done": True,
                    "content": full_content if full_content else None,
                    "tool_calls": final_tool_calls,
                    "model": model_name, "provider": "openrouter",
                    "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": total_tokens},
                    "tokens_per_second": tokens_per_second, "finish_reason": finish_reason or "stop"
                }
                yield {k: v for k, v in final_yield.items() if v is not None}
                logger.debug(f"OpenRouter streaming yielded final chunk.")


    async def get_available_models(self) -> tuple[List[str], List[str]]:
        """ Get available chat and embedding models from OpenRouter. """
        logger.info("Getting available models from OpenRouter")
        all_models_info = await self.list_models()
        if not all_models_info: return [], []

        chat_models, embedding_models = [], []
        embedding_keywords = ["embed", "embedding", "ada-002"]

        for model_info in all_models_info:
            model_id = model_info.get("id")
            if not model_id: continue
            is_embedding = any(keyword in model_id.lower() for keyword in embedding_keywords)
            if is_embedding: embedding_models.append(model_id)
            else: chat_models.append(model_id)

        chat_models, embedding_models = sorted(list(set(chat_models))), sorted(list(set(embedding_models)))
        logger.info(f"Categorized models: {len(chat_models)} chat, {len(embedding_models)} embedding.")
        return chat_models, embedding_models

    async def list_models(self) -> List[Dict[str, Any]]:
        """ List available models from OpenRouter. """
        url = f"{self.base_url}/models"
        logger.info(f"Fetching OpenRouter models from {url}")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": getattr(settings, "OPENROUTER_REFERRER", "https://github.com/rooveterinary/doogie"),
            "X-Title": getattr(settings, "OPENROUTER_APP_TITLE", "Doogie"),
            "Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache", "Expires": "0"
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"OpenRouter models API error: {response.status} - {error_text}")
                        return []
                    result = await response.json()
                    models = result.get("data", [])
                    logger.info(f"Received {len(models)} models from OpenRouter")
                    return models
        except Exception as e:
            logger.error(f"Error listing OpenRouter models: {str(e)}")
            return []

    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """ Get embeddings using OpenRouter (proxies to OpenAI compatible endpoint). """
        url = f"{self.base_url}/embeddings"
        headers = {
            "Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": settings.OPENROUTER_REFERRER or "https://github.com/rooveterinary/doogie",
            "X-Title": settings.OPENROUTER_APP_TITLE or "Doogie"
        }
        embedding_model = self.embedding_model or "openai/text-embedding-ada-002"
        logger.info(f"Generating embeddings for {len(texts)} texts using OpenRouter model: {embedding_model}")

        try:
            payload = {"model": embedding_model, "input": texts}
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"OpenRouter API error generating embeddings: {error_text}")
                        dimension = 1536
                        if "3-small" in embedding_model: dimension = 1536
                        elif "3-large" in embedding_model: dimension = 3072
                        return [[0.0] * dimension for _ in range(len(texts))]
                    result = await response.json()
                    embeddings = [item["embedding"] for item in result["data"]]
                    logger.info(f"Successfully generated {len(embeddings)} embeddings with OpenRouter")
                    return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings with OpenRouter: {str(e)}")
            dimension = 1536
            if "3-small" in embedding_model: dimension = 1536
            elif "3-large" in embedding_model: dimension = 3072
            return [[0.0] * dimension for _ in range(len(texts))]