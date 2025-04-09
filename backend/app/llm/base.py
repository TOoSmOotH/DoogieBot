from typing import List, Dict, Any, Optional, Union, AsyncGenerator
from abc import ABC, abstractmethod
import time
import logging

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
        user_id: Optional[str] = None # Add user_id parameter
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
        self.embedding_model = embedding_model or model  # Use chat model for embeddings if not specified
        self.api_key = api_key
        self.base_url = base_url
        self.user_id = user_id # Store user_id

    @abstractmethod
    async def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        tools: Optional[List[Dict[str, Any]]] = None, # <-- Add tools parameter
        tool_choice: Optional[str] = None # <-- Add tool_choice parameter (optional)
    ) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        """
        Generate a response from the LLM.

        Args:
            messages: List of messages in the conversation
            temperature: Temperature for generation
            max_tokens: Maximum number of tokens to generate
            stream: Whether to stream the response
            tools: Optional list of tool definitions available to the model.
            tool_choice: Optional control over which tool is called (e.g., "auto", "none").

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

    def format_chat_message(
        self,
        role: str,
        content: Optional[str] = None, # Content can be None for assistant tool calls
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        tool_call_id: Optional[str] = None,
        name: Optional[str] = None # For tool result messages
    ) -> Dict[str, Any]:
        """
        Format a chat message, potentially including tool calls or results.

        Args:
            role: Message role (user, assistant, system, tool)
            content: Message content (can be None for assistant tool calls)
            tool_calls: List of tool calls made by the assistant.
            tool_call_id: ID of the tool call this message is a result for.
            name: The name of the tool whose result this is.

        Returns:
            Formatted message dictionary.
        """
        message: Dict[str, Any] = {"role": role}
        if content is not None:
            message["content"] = content
        if tool_calls:
            message["tool_calls"] = tool_calls
        if tool_call_id:
            message["tool_call_id"] = tool_call_id
        if name: # Include name if provided (typically for tool role)
             message["name"] = name
        # Ensure content is at least an empty string if not provided and not a tool call message
        if "content" not in message and not tool_calls:
             message["content"] = ""
        return message