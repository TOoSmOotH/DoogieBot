import logging
from typing import Any, Dict, Optional, Type

from app.core.config import settings
from app.llm.base import LLMClient
from app.llm.ollama_client import OllamaClient
from app.llm.openai_client import OpenAIClient

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import other LLM clients as needed
# Uncomment these when the actual client implementations are available
# from app.llm.anthropic_client import AnthropicClient
# from app.llm.openrouter_client import OpenRouterClient
# from app.llm.deepseek_client import DeepseekClient
# from app.llm.lmstudio_client import LMStudioClient


class LLMFactory:
    """
    Factory for creating LLM clients.
    """

    # Registry of LLM providers and their client classes
    _registry: Dict[str, Type[LLMClient]] = {
        "openai": OpenAIClient,
        "ollama": OllamaClient,
        # These are placeholders - they will use OpenAIClient until proper implementations are available
        "anthropic": OpenAIClient,  # Placeholder using OpenAIClient
        "openrouter": OpenAIClient,  # Placeholder using OpenAIClient
        # "deepseek": DeepseekClient,
        # "lmstudio": LMStudioClient,
    }

    @classmethod
    def create_client(
        cls,
        provider: str,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ) -> LLMClient:
        """
        Create an LLM client for the specified provider.

        Args:
            provider: LLM provider name
            model: Model name (optional, uses default if not provided)
            api_key: API key (optional, uses default if not provided)
            base_url: Base URL for API (optional, uses default if not provided)

        Returns:
            LLM client instance

        Raises:
            ValueError: If the provider is not supported
        """
        # Convert provider to lowercase for case-insensitive matching
        provider = provider.lower()

        # Check if provider is supported
        if provider not in cls._registry:
            raise ValueError(f"Unsupported LLM provider: {provider}")

        # Get client class
        client_class = cls._registry[provider]

        # Use default model if not provided
        if not model:
            model = cls._get_default_model(provider)

        # Create client instance
        try:
            client = client_class(
                model=model,
                api_key=api_key,
                base_url=base_url,
                embedding_model=embedding_model,
            )
            logger.info(
                f"Created {provider} client with model {model} and embedding model {embedding_model or model}"
            )
            return client
        except Exception as e:
            logger.error(f"Error creating {provider} client: {str(e)}")
            raise

    @classmethod
    def _get_default_model(cls, provider: str) -> str:
        """
        Get the default model for a provider.

        Args:
            provider: LLM provider name

        Returns:
            Default model name
        """
        if provider == "openai":
            return settings.DEFAULT_CHAT_MODEL
        elif provider == "ollama":
            return "llama2"
        elif provider == "anthropic":
            return "claude-2"
        elif provider == "openrouter":
            return "openai/gpt-3.5-turbo"
        elif provider == "deepseek":
            return "deepseek-chat"
        elif provider == "lmstudio":
            return "lmstudio-model"
        else:
            return "gpt-3.5-turbo"  # Default fallback

    @classmethod
    def get_available_providers(cls) -> Dict[str, Any]:
        """
        Get a list of available LLM providers.

        Returns:
            Dictionary of available providers and their status
        """
        providers = {}

        # Check OpenAI
        providers["openai"] = {
            "available": bool(settings.OPENAI_API_KEY),
            "default_model": settings.DEFAULT_CHAT_MODEL,
            "requires_api_key": True,
            "requires_base_url": False,
        }

        # Check Ollama
        providers["ollama"] = {
            "available": bool(settings.OLLAMA_BASE_URL),
            "default_model": "llama2",
            "requires_api_key": False,
            "requires_base_url": True,
        }

        # Check Anthropic
        providers["anthropic"] = {
            "available": bool(settings.ANTHROPIC_API_KEY),
            "default_model": "claude-2",
            "requires_api_key": True,
            "requires_base_url": False,
        }

        # Check OpenRouter
        providers["openrouter"] = {
            "available": bool(settings.OPENROUTER_API_KEY),
            "default_model": "openai/gpt-3.5-turbo",
            "requires_api_key": True,
            "requires_base_url": False,
        }

        # Check other providers as they are implemented
        # providers["deepseek"] = {
        #     "available": bool(settings.DEEPSEEK_API_KEY),
        #     "default_model": "deepseek-chat",
        #     "requires_api_key": True,
        #     "requires_base_url": False
        # }

        # providers["lmstudio"] = {
        #     "available": bool(settings.LM_STUDIO_BASE_URL),
        #     "default_model": "lmstudio-model",
        #     "requires_api_key": False,
        #     "requires_base_url": True
        # }

        return providers
