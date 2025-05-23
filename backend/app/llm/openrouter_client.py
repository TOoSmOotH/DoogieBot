from typing import List, Dict, Any, Optional, Union, AsyncGenerator
import aiohttp
import json
import time
import logging
import asyncio

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
        base_url: Optional[str] = "https://openrouter.ai/api",
        embedding_model: Optional[str] = None
    ):
        """
        Initialize the OpenRouter client.
        
        Args:
            model: Model name (e.g. "openai/gpt-3.5-turbo")
            api_key: OpenRouter API key
            base_url: Base URL for API
            embedding_model: Model to use for embeddings (if different from chat model)
        """
        super().__init__(model, api_key, base_url, embedding_model)
        # Initialize streaming state variables
        self._current_reasoning = ""
        self._reasoning_complete = False
        self.api_key = api_key or settings.OPENROUTER_API_KEY
        if not self.api_key:
            raise ValueError("OpenRouter API key is required")
        
        # Always use the correct base URL for OpenRouter
        self.base_url = "https://openrouter.ai/api"
    
    async def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        """
        Generate a response from OpenRouter.
        
        Args:
            messages: List of messages in the conversation
            temperature: Temperature for generation
            max_tokens: Maximum number of tokens to generate
            stream: Whether to stream the response
            
        Returns:
            Response from OpenRouter or an async generator for streaming
        """
        # Always use the correct base URL for OpenRouter
        self.base_url = "https://openrouter.ai/api/v1"
            
        url = f"{self.base_url}/chat/completions"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": getattr(settings, "OPENROUTER_REFERRER", "https://github.com/rooveterinary/doogie"),
            "X-Title": getattr(settings, "OPENROUTER_APP_TITLE", "Doogie")
        }
        
        # Log the messages for debugging
        logger.info(f"Sending {len(messages)} messages to OpenRouter. First message role: {messages[0]['role'] if messages else 'none'}")
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": stream
        }
        
        # Log the full payload for debugging if LLM_DEBUG_LOGGING is enabled
        if settings.LLM_DEBUG_LOGGING:
            logger.info("Full OpenRouter request payload (LLM_DEBUG_LOGGING enabled):")
            try:
                # Log each message separately to avoid log size limitations
                for i, msg in enumerate(messages):
                    role = msg.get('role', 'unknown')
                    content = msg.get('content', '')
                    
                    # For system messages that might contain RAG context, log in detail
                    if role == 'system':
                        logger.info(f"Message {i} - Role: {role}")
                        # Log the system message in chunks to avoid log size limitations
                        content_chunks = [content[i:i+1000] for i in range(0, len(content), 1000)]
                        for j, chunk in enumerate(content_chunks):
                            logger.info(f"System message chunk {j}: {chunk}")
                    else:
                        # For non-system messages, log a preview
                        content_preview = content[:200] + "..." if len(content) > 200 else content
                        logger.info(f"Message {i} - Role: {role}, Content preview: {content_preview}")
                
                # Log other payload parameters
                logger.info(f"Model: {self.model}, Temperature: {temperature}, Stream: {stream}")
            except Exception as e:
                logger.error(f"Error logging payload: {str(e)}")
        
        if max_tokens:
            payload["max_tokens"] = max_tokens
        
        start_time = time.time()
        
        if stream:
            return self._stream_response(url, headers, payload, start_time)
        else:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"OpenRouter API error: {error_text}")
                        raise Exception(f"OpenRouter API error: {response.status} - {error_text}")
                    
                    result = await response.json()
                    
                    # Calculate tokens per second
                    tokens = result.get("usage", {}).get("completion_tokens", 0)
                    tokens_per_second = self.calculate_tokens_per_second(start_time, tokens)
                    
                    return {
                        "content": result["choices"][0]["message"]["content"],
                        "model": self.model,
                        "provider": "openrouter",
                        "tokens": tokens,
                        "tokens_per_second": tokens_per_second
                    }
    
    async def _stream_response(
        self,
        url: str,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        start_time: float
    ) -> AsyncGenerator[Dict[str, Any], None]:
        # Initialize per-request state
        self._current_reasoning = ""
        self._has_reasoning = False
        """
        Stream response from OpenRouter.
        
        Args:
            url: API URL
            headers: Request headers
            payload: Request payload
            start_time: Start time for calculating tokens per second
            
        Yields:
            Chunks of the response
        """
        # Ensure URL is valid
        if not url.startswith("http"):
            # Always use the correct base URL for OpenRouter
            self.base_url = "https://openrouter.ai/api/v1"
            
            # If url is just a path, prepend the base_url
            if url.startswith("/"):
                url = f"{self.base_url}{url}"
            else:
                url = f"{self.base_url}/{url}"
                
            # Ensure we don't have duplicate v1 in the path
            url = url.replace("/v1/v1/", "/v1/")
        
        logger.debug(f"Starting OpenRouter streaming request to {url}")
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"OpenRouter API error: {error_text}")
                    raise Exception(f"OpenRouter API error: {response.status} - {error_text}")
                
                logger.debug(f"OpenRouter streaming connection established with status {response.status}")
                
                # Initialize variables for streaming
                content = ""
                # token_count = 0 # Remove incorrect counter
                chunk_count = 0
                has_reasoning_support = False

                # Variables to store final usage details if provided by the API
                final_prompt_tokens = 0
                final_completion_tokens = 0
                final_total_tokens = 0
                finish_reason = None
                
                # Process the stream
                logger.debug(f"Starting to process OpenRouter stream")
                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    
                    # Skip empty lines, [DONE], or processing messages
                    if not line or line == "data: [DONE]" or line.startswith(": OPENROUTER PROCESSING"):
                        if line == "data: [DONE]":
                            logger.debug("Received [DONE] from OpenRouter stream")
                        elif line.startswith(": OPENROUTER PROCESSING"):
                            logger.debug("OpenRouter processing message received")
                        # Don't necessarily continue here, the [DONE] line might be followed by a final JSON with usage
                        # Let the JSON parser handle potential errors
                        # continue
                    
                    # Remove "data: " prefix if present
                    if line.startswith("data: "):
                        line = line[6:]
                    
                    try:
                        # Skip if line is empty after processing
                        if not line.strip():
                            continue
                            
                        data = json.loads(line)
                        logger.debug(f"Raw OpenRouter response: {data}")
                        
                        # Validate response structure
                        if not isinstance(data, dict):
                            logger.warning(f"Invalid response format: {data}")
                            continue
                            
                        # Extract delta content and finish reason
                        choices = data.get("choices", [])
                        delta_content = "" # Initialize delta content for this chunk

                        if choices and isinstance(choices, list):
                            delta = choices[0].get("delta", {})
                            if isinstance(delta, dict):
                                delta_content = delta.get("content", "")
                                # ... (handle delta_reasoning if needed) ...
                                if delta_content and isinstance(delta_content, str):
                                    chunk_count += 1
                                    # ... (handle boxed content) ...
                                    content += delta_content
                            # Check for finish reason in the choice
                            finish_reason = choices[0].get("finish_reason", finish_reason)
                        else:
                            # Log if choices are missing or invalid, but continue processing for usage info
                            logger.debug(f"No valid choices in chunk: {data}")


                        # Check for usage information in the main data object (often in the final chunk)
                        usage = data.get("usage")
                        if usage and isinstance(usage, dict):
                             logger.debug(f"Found usage info in chunk: {usage}")
                             final_prompt_tokens = usage.get("prompt_tokens", final_prompt_tokens)
                             final_completion_tokens = usage.get("completion_tokens", final_completion_tokens)
                             final_total_tokens = usage.get("total_tokens", final_prompt_tokens + final_completion_tokens)


                        # Calculate tokens per second based on accumulated content length (approximation)
                        # A more accurate way would be to use final_completion_tokens if available at the end
                        # Use chunk_count as a proxy for completion tokens for intermediate TPS calculation
                        approx_tokens_per_second = self.calculate_tokens_per_second(start_time, chunk_count)

                        logger.debug(f"Yielding chunk {chunk_count} at {time.time()}")
                        
                        # Yield immediately without any delay
                        yield {
                            "content": content, # Yield accumulated content so far
                            "model": self.model,
                            "provider": "openrouter",
                            # "tokens": token_count, # Remove incorrect count
                            "tokens_per_second": approx_tokens_per_second, # Yield approximate TPS
                            "done": False,
                            "timestamp": time.time()
                            # Do not yield usage here, wait for the final chunk
                        }
                        
                        # Ensure the chunk is sent immediately
                        await asyncio.sleep(0)
                    except json.JSONDecodeError:
                        logger.warning(f"Could not parse line: {line}")
                
                logger.debug(f"OpenRouter stream complete, yielding final chunk with done=True")
                # Final yield with done=True
                # Use the actual token counts if they were found during the stream
                final_tokens_to_yield = final_total_tokens if final_total_tokens > 0 else final_completion_tokens
                # Calculate final TPS based on actual completion tokens if available
                final_tps = self.calculate_tokens_per_second(start_time, final_completion_tokens) if final_completion_tokens > 0 else 0.0
                
                # Add a note if the model doesn't support reasoning
                if not has_reasoning_support and self._current_reasoning == "":
                    logger.info(f"Model {self.model} does not appear to support reasoning output")
                
                yield {
                    "content": content, # Final accumulated content
                    "model": self.model,
                    "provider": "openrouter",
                    "tokens": final_tokens_to_yield, # Use actual tokens from usage if found
                    "tokens_per_second": final_tps, # Use TPS based on actual completion tokens if found
                    "done": True,
                    "usage": { # Include full usage details if available
                        "prompt_tokens": final_prompt_tokens,
                        "completion_tokens": final_completion_tokens,
                        "total_tokens": final_total_tokens
                    },
                    "finish_reason": finish_reason
                }
                logger.debug(f"OpenRouter streaming complete, yielded final chunk with {final_tokens_to_yield} tokens")

    async def get_available_models(self) -> tuple[List[str], List[str]]:
        """
        Get available chat and embedding models from OpenRouter.

        Returns:
            A tuple containing two lists: (chat_model_ids, embedding_model_ids)
        """
        logger.info("Getting available models from OpenRouter using get_available_models")
        all_models_info = await self.list_models() # Call the existing method

        if not all_models_info:
            logger.warning("No models returned from OpenRouter list_models.")
            return [], []

        chat_models = []
        embedding_models = []

        # Common embedding model identifiers
        embedding_keywords = ["embed", "embedding", "ada-002"]

        for model_info in all_models_info:
            model_id = model_info.get("id")
            if not model_id:
                continue

            # Check if it's likely an embedding model
            is_embedding = any(keyword in model_id.lower() for keyword in embedding_keywords)

            if is_embedding:
                embedding_models.append(model_id)
            else:
                # Assume others are potential chat models
                # We could add more filtering here if needed based on OpenRouter's data
                chat_models.append(model_id)

        # Ensure uniqueness and sort
        chat_models = sorted(list(set(chat_models)))
        embedding_models = sorted(list(set(embedding_models)))

        logger.info(f"Categorized models: {len(chat_models)} chat, {len(embedding_models)} embedding.")
        logger.debug(f"Sample chat models: {chat_models[:10]}...")
        logger.debug(f"Sample embedding models: {embedding_models[:10]}...")

        return chat_models, embedding_models
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """
        List available models from OpenRouter.
        
        Returns:
            List of model info dictionaries
        """
        # Always use the correct base URL for OpenRouter
        self.base_url = "https://openrouter.ai/api/v1"
            
        # The correct endpoint for OpenRouter models
        url = f"{self.base_url}/models"
        logger.info(f"Using OpenRouter models URL: {url}")
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": getattr(settings, "OPENROUTER_REFERRER", "https://github.com/rooveterinary/doogie"),
            "X-Title": getattr(settings, "OPENROUTER_APP_TITLE", "Doogie")
        }
        
        try:
            logger.info(f"Fetching OpenRouter models from {url}")
            async with aiohttp.ClientSession() as session:
                # Add cache-control headers to prevent caching
                headers.update({
                    "Cache-Control": "no-cache, no-store, must-revalidate",
                    "Pragma": "no-cache",
                    "Expires": "0"
                })
                async with session.get(url, headers=headers) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"OpenRouter models API error: {response.status} - {error_text}")
                        return []
                    
                    result = await response.json()
                    logger.info(f"OpenRouter API response: {result}")
                    models = result.get("data", [])
                    logger.info(f"Received {len(models)} models from OpenRouter")
                    if models:
                        logger.debug(f"First model: {models[0].get('id')}")
                        logger.info(f"Sample models: {[m.get('id') for m in models[:5] if m.get('id')]}")
                    return models
        except Exception as e:
            logger.error(f"Error listing OpenRouter models: {str(e)}")
            return []

    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for a list of texts using OpenRouter.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        # OpenRouter doesn't currently support embeddings, so we'll use OpenAI's embeddings
        # through OpenRouter by specifying an OpenAI model
        
        # Always use the correct base URL for OpenRouter
        self.base_url = "https://openrouter.ai/api/v1"
            
        # Construct the embeddings URL
        url = f"{self.base_url}/embeddings"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": settings.OPENROUTER_REFERRER or "https://github.com/rooveterinary/doogie",
            "X-Title": settings.OPENROUTER_APP_TITLE or "Doogie"
        }
        
        # Use the instance's embedding model or fall back to a default
        embedding_model = self.embedding_model
        if not embedding_model:
            embedding_model = "text-embedding-ada-002"  # Default OpenAI embedding model
        
        # Log the embedding request
        logger.info(f"Generating embeddings for {len(texts)} texts using OpenRouter model: {embedding_model}")
        
        try:
            payload = {
                "model": embedding_model,
                "input": texts
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"OpenRouter API error: {error_text}")
                        # Return empty embeddings instead of raising an exception
                        return [[0.0] * 1536 for _ in range(len(texts))]  # OpenAI embeddings are typically 1536 dimensions
                    
                    result = await response.json()
                    
                    # Extract embeddings
                    embeddings = [item["embedding"] for item in result["data"]]
                    
                    logger.info(f"Successfully generated {len(embeddings)} embeddings with OpenRouter")
                    if embeddings:
                        logger.debug(f"Embedding dimensions: {len(embeddings[0])}")
                    
                    return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings with OpenRouter: {str(e)}")
            logger.exception("Detailed embedding generation error:")
            # Return empty embeddings
            return [[0.0] * 1536 for _ in range(len(texts))]  # OpenAI embeddings are typically 1536 dimensions