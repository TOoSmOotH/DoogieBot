import logging
import time
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMClient(ABC):
    """
    Abstract base class for LLM clients.
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ):
        """
        Initialize the LLM client.

        Args:
            model: Chat model name
            api_key: API key (if required)
            base_url: Base URL for API (if required)
            embedding_model: Model to use for embeddings (if different from chat model)
        """
        self.model = model
        self.embedding_model = (
            embedding_model or model
        )  # Use chat model for embeddings if not specified
        self.api_key = api_key
        self.base_url = base_url

    @abstractmethod
    async def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
    ) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        """
        Generate a response from the LLM.

        Args:
            messages: List of messages in the conversation
            temperature: Temperature for generation
            max_tokens: Maximum number of tokens to generate
            stream: Whether to stream the response

        Returns:
            Response from the LLM or an async generator for streaming
        """
        pass

    @abstractmethod
    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for a list of texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        pass

    def calculate_tokens_per_second(self, start_time: float, tokens: int) -> float:
        """
        Calculate tokens per second.

        Args:
            start_time: Start time in seconds
            tokens: Number of tokens generated

        Returns:
            Tokens per second
        """
        elapsed_time = time.time() - start_time
        if elapsed_time > 0:
            return tokens / elapsed_time
        return 0.0

    def format_chat_message(self, role: str, content: str) -> Dict[str, str]:
        """
        Format a chat message.

        Args:
            role: Message role (user, assistant, system)
            content: Message content

        Returns:
            Formatted message
        """
        return {"role": role, "content": content}
