import asyncio
import json
import logging
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

import aiohttp
from app.core.config import settings
from app.llm.base import LLMClient

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
    ):
        """
        Initialize the Ollama client.

        Args:
            model: Model name
            api_key: Not used for Ollama
            base_url: Base URL for Ollama API
            embedding_model: Model to use for embeddings (if different from chat model)
        """
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url,
            embedding_model=embedding_model,
        )
        self.base_url = base_url or settings.OLLAMA_BASE_URL
        if not self.base_url:
            raise ValueError("Ollama base URL is required")

    async def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
    ) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        """
        Generate a response from Ollama.

        Args:
            messages: List of messages in the conversation
            temperature: Temperature for generation
            max_tokens: Maximum number of tokens to generate
            stream: Whether to stream the response

        Returns:
            Response from Ollama or an async generator for streaming
        """
        url = f"{self.base_url}/api/chat"

        headers = {"Content-Type": "application/json"}

        # Convert messages to Ollama format
        formatted_messages = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            # No role conversion needed for Ollama - it supports standard roles
            # Just log the role for debugging
            logger.debug(f"Formatting message with role: {role}")

            formatted_messages.append({"role": role, "content": content})

        # Log the formatted messages for debugging
        logger.info(
            f"Sending {len(formatted_messages)} messages to Ollama. First message role: {formatted_messages[0]['role'] if formatted_messages else 'none'}"
        )

        payload = {
            "model": self.model,
            "messages": formatted_messages,
            "stream": stream,
            "options": {"temperature": temperature},
        }

        if max_tokens:
            payload["options"]["num_predict"] = max_tokens

        start_time = time.time()

        if stream:
            return self._stream_response(url, headers, payload, start_time)
        else:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Ollama API error: {error_text}")
                        raise Exception(
                            f"Ollama API error: {response.status} - {error_text}"
                        )

                    result = await response.json()

                    # Extract content from the response
                    content = result.get("message", {}).get("content", "")

                    # Get token count from response if available
                    eval_count = result.get("eval_count", 0)
                    tokens = eval_count if eval_count > 0 else len(content.split())

                    # Calculate tokens per second
                    tokens_per_second = self.calculate_tokens_per_second(
                        start_time, tokens
                    )

                    return {
                        "content": content,
                        "model": self.model,
                        "provider": "ollama",
                        "tokens": tokens,
                        "tokens_per_second": tokens_per_second,
                    }

    async def _stream_response(
        self,
        url: str,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        start_time: float,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream response from Ollama.

        Args:
            url: API URL
            headers: Request headers
            payload: Request payload
            start_time: Start time for calculating tokens per second

        Yields:
            Chunks of the response
        """
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Ollama API error: {error_text}")
                    raise Exception(
                        f"Ollama API error: {response.status} - {error_text}"
                    )

                # Initialize variables for streaming
                content = ""
                token_count = 0

                # Process the stream
                async for line in response.content:
                    line = line.decode("utf-8").strip()

                    # Skip empty lines
                    if not line:
                        continue

                    try:
                        data = json.loads(line)

                        # Extract content from the response
                        delta_content = data.get("message", {}).get("content", "")

                        if delta_content:
                            content += delta_content
                            token_count += 1  # Approximate token count

                            # Calculate tokens per second
                            tokens_per_second = self.calculate_tokens_per_second(
                                start_time, token_count
                            )

                            yield {
                                "content": content,
                                "model": self.model,
                                "provider": "ollama",
                                "tokens": token_count,
                                "tokens_per_second": tokens_per_second,
                                "done": False,
                            }

                        # Check if this is the final message
                        if data.get("done", False):
                            # Final yield with done=True
                            tokens_per_second = self.calculate_tokens_per_second(
                                start_time, token_count
                            )
                            yield {
                                "content": content,
                                "model": self.model,
                                "provider": "ollama",
                                "tokens": token_count,
                                "tokens_per_second": tokens_per_second,
                                "done": True,
                            }
                            break

                    except json.JSONDecodeError:
                        logger.warning(f"Could not parse line: {line}")

    async def list_models(self) -> List[str]:
        """
        List available models from Ollama.

        Returns:
            List of available model names
        """
        try:
            url = f"{self.base_url}/api/tags"

            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Ollama API error: {error_text}")
                        return []

                    result = await response.json()

                    # Extract model names from the response
                    models = [model["name"] for model in result.get("models", [])]
                    return models
        except Exception as e:
            logger.error(f"Error listing Ollama models: {str(e)}")
            return []

    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for a list of texts using Ollama.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        url = f"{self.base_url}/api/embeddings"

        headers = {"Content-Type": "application/json"}

        embeddings = []

        # Log the number of texts to embed
        logger.info(f"Generating embeddings for {len(texts)} texts using Ollama")

        # Ollama API processes one text at a time
        for i, text in enumerate(texts):
            try:
                # Log progress for every 10th text
                if i % 10 == 0:
                    logger.info(f"Processing embedding {i+1}/{len(texts)}")

                payload = {
                    "model": self.embedding_model
                    or self.model,  # Use embedding model if set, else fall back to chat model
                    "prompt": text,
                }

                logger.debug(
                    f"Sending embedding request to Ollama for text {i+1}: {text[:50]}..."
                )

                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        url, headers=headers, json=payload
                    ) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            logger.error(
                                f"Ollama API error for text {i+1}: {error_text}"
                            )
                            # Instead of raising an exception, return an empty embedding
                            # This allows processing to continue even if one embedding fails
                            embeddings.append(
                                [0.0] * 768
                            )  # Default size for most embedding models
                            continue

                        result = await response.json()

                        # Extract embedding
                        embedding = result.get("embedding", [])

                        if not embedding:
                            logger.warning(
                                f"Ollama returned empty embedding for text {i+1}"
                            )
                            embeddings.append(
                                [0.0] * 768
                            )  # Default size for most embedding models
                        else:
                            logger.debug(
                                f"Successfully generated embedding for text {i+1} with dimension {len(embedding)}"
                            )
                            embeddings.append(embedding)
            except Exception as e:
                logger.error(f"Error generating embedding for text {i+1}: {str(e)}")
                # Add a dummy embedding to maintain the same length as the input texts
                embeddings.append([0.0] * 768)  # Default size for most embedding models

        logger.info(
            f"Completed embedding generation: {len(embeddings)}/{len(texts)} successful"
        )
        return embeddings
