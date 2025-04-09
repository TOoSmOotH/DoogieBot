from typing import Dict, Any, Optional, Type, Union, Tuple
import logging

from app.llm.base import LLMClient
from app.llm.openai_client import OpenAIClient
from app.llm.ollama_client import OllamaClient
from app.llm.openrouter_client import OpenRouterClient
# Import newly added clients
from app.llm.anthropic_client import AnthropicClient
from app.llm.google_gemini_client import GoogleGeminiClient
from app.core.config import settings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import other LLM clients as needed
# Uncomment these when the actual client implementations are available
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
        "anthropic": AnthropicClient, # Use the actual Anthropic client
        "openrouter": OpenRouterClient,
        "google_gemini": GoogleGeminiClient, # Add Google Gemini client
        # "deepseek": DeepseekClient,
        # "lmstudio": LMStudioClient,
    }
    
    @classmethod
    def create_separate_clients(
        cls,
        chat_config: Dict[str, Any],
        embedding_config: Dict[str, Any],
        user_id: Optional[str] = None # Add user_id parameter
    ) -> Tuple[LLMClient, LLMClient]:
        """
        Create separate chat and embedding clients from their respective configs.
        
        Args:
            chat_config: Dictionary containing chat provider configuration
            embedding_config: Dictionary containing embedding provider configuration
            
        Returns:
            Tuple of (chat_client, embedding_client)
            
        Raises:
            ValueError: If any provider is not supported
        """
        chat_provider = chat_config.get('provider', '').lower()
        embedding_provider = embedding_config.get('provider', '').lower()
        
        # Check if providers are supported
        if chat_provider not in cls._registry:
            raise ValueError(f"Unsupported chat LLM provider: {chat_provider}")
        if embedding_provider not in cls._registry:
            raise ValueError(f"Unsupported embedding LLM provider: {embedding_provider}")
            
        try:
            # Create chat client
            chat_client = cls._create_single_client(
                provider=chat_provider,
                model=chat_config.get('model'),
                api_key=chat_config.get('api_key'),
                base_url=chat_config.get('base_url')
            )
            
            # Create embedding client
            embedding_client = cls._create_single_client(
                provider=embedding_provider,
                model=embedding_config.get('model'),
                api_key=embedding_config.get('api_key'),
                base_url=embedding_config.get('base_url'),
                user_id=user_id # Pass user_id
            )

            return (chat_client, embedding_client)
            
        except Exception as e:
            logger.error(f"Error creating separate clients: {str(e)}")
            raise

    @classmethod
    def _create_single_client(
        cls,
        provider: str,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        user_id: Optional[str] = None # Add user_id parameter
    ) -> LLMClient:
        """
        Internal method to create a single client instance.
        """
        provider = provider.lower()
        if provider not in cls._registry:
            raise ValueError(f"Unsupported provider: {provider}")
            
        if not model:
            model = cls._get_default_model(provider)
            
        # Set provider-specific base URLs
        if provider == "ollama":
            base_url = base_url if base_url else settings.OLLAMA_BASE_URL
        elif provider == "openrouter":
            base_url = "https://openrouter.ai/api/v1"
        elif provider == "openai":
             # OpenAI client uses default URL unless a specific one is provided
             # If base_url was incorrectly pulled from another provider's config, reset it.
             # Allow explicit override if base_url was intentionally passed for OpenAI (e.g., proxy)
             # We assume if base_url is passed *to this function*, it's intentional for OpenAI.
             # The issue arises when base_url is *derived* from the wrong DB config in the service layer.
             # Let's refine: only override if the base_url looks like a known local one.
             # A better fix might be in the service layer, but let's patch here for now.
             if base_url and ("localhost" in base_url or "127.0.0.1" in base_url or "ollama" in base_url or "lmstudio" in base_url):
                 # If a local-looking URL was passed for OpenAI, it's likely wrong. Reset it.
                 logger.warning(f"Potential incorrect base_url '{base_url}' provided for OpenAI client. Resetting to default.")
                 base_url = None # Use OpenAI client's default
             # Otherwise, trust the passed base_url if it exists (e.g. for Azure OpenAI, proxies)

        client_class = cls._registry[provider]
        client = client_class(
            model=model,
            api_key=api_key,
            base_url=base_url,
            user_id=user_id # Pass user_id to constructor
        )
        logger.info(f"Created {provider} client with model {model} for user {user_id}")
        return client

    @classmethod
    def create_client(
        cls,
        provider: str,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        embedding_model: Optional[str] = None,
        embedding_provider: Optional[str] = None,
        user_id: Optional[str] = None # Add user_id parameter
    ) -> Union[LLMClient, Tuple[LLMClient, LLMClient]]:
        """
        Create an LLM client for the specified provider (legacy method).
        If embedding_provider is different from provider, returns a tuple of (chat_client, embedding_client).
        
        Args:
            provider: Chat LLM provider name
            model: Chat model name (optional, uses default if not provided)
            api_key: API key (optional, uses default if not provided)
            base_url: Base URL for API (optional, uses default if not provided)
            embedding_model: Embedding model name (optional)
            embedding_provider: Embedding provider name (optional, uses chat provider if not provided)
            
        Returns:
            LLM client instance or tuple of (chat_client, embedding_client)
            
        Raises:
            ValueError: If any provider is not supported
        """
        # Convert providers to lowercase for case-insensitive matching
        provider = provider.lower()
        embedding_provider = (embedding_provider or provider).lower()
        
        # Check if providers are supported
        if provider not in cls._registry:
            raise ValueError(f"Unsupported chat LLM provider: {provider}")
        if embedding_provider not in cls._registry:
            raise ValueError(f"Unsupported embedding LLM provider: {embedding_provider}")
        
        # Use default models if not provided
        if not model:
            model = cls._get_default_model(provider)
        if not embedding_model:
            embedding_model = cls._get_default_model(embedding_provider)
        
        try:
            # Create chat client
            chat_client = cls._create_single_client(
                provider=provider,
                model=model,
                api_key=api_key,
                base_url=base_url,
                user_id=user_id # Pass user_id
            )

            # If same provider, return single client with embedding model set
            if provider == embedding_provider:
                chat_client.embedding_model = embedding_model
                logger.info(f"Using same client for embeddings with model {embedding_model}")
                return chat_client
                
            # Create separate embedding client
            embedding_client = cls._create_single_client(
                provider=embedding_provider,
                model=embedding_model,
                api_key=api_key,
                base_url=base_url,
                user_id=user_id # Pass user_id
            )

            return (chat_client, embedding_client)
            
        except Exception as e:
            logger.error(f"Error creating LLM client(s): {str(e)}")
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
            return "claude-3-opus-20240229" # Updated default for Anthropic
        elif provider == "openrouter":
            return "openai/gpt-3.5-turbo"
        elif provider == "google_gemini":
            return "gemini-pro" # Default for Google Gemini
        # elif provider == "deepseek":
        #     return "deepseek-chat"
        # elif provider == "lmstudio":
        #     return "lmstudio-model"
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
            "requires_base_url": False # Base URL is optional (e.g., for Azure)
        }
        
        # Check Ollama
        providers["ollama"] = {
            "available": bool(settings.OLLAMA_BASE_URL),
            "default_model": "llama2", # Consider making this configurable or dynamic
            "requires_api_key": False,
            "requires_base_url": True
        }
        
        # Check Anthropic
        providers["anthropic"] = {
            "available": bool(settings.ANTHROPIC_API_KEY),
            "default_model": cls._get_default_model("anthropic"), # Use the method
            "requires_api_key": True,
            "requires_base_url": False # Base URL is optional
        }
        
        # Check OpenRouter
        providers["openrouter"] = {
            "available": bool(settings.OPENROUTER_API_KEY),
            "default_model": cls._get_default_model("openrouter"), # Use the method
            "requires_api_key": True,
            "requires_base_url": False # Uses OpenRouter's base URL implicitly
        }
        
        # Check Google Gemini
        providers["google_gemini"] = {
            "available": bool(settings.GOOGLE_GEMINI_API_KEY),
            "default_model": cls._get_default_model("google_gemini"), # Use the method
            "requires_api_key": True,
            "requires_base_url": False
        }
        
        # Check other providers as they are implemented
        # providers["deepseek"] = {
        #     "available": bool(settings.DEEPSEEK_API_KEY),
        #     "default_model": cls._get_default_model("deepseek"),
        #     "requires_api_key": True,
        #     "requires_base_url": False
        # }
        
        # providers["lmstudio"] = {
        #     "available": bool(settings.LM_STUDIO_BASE_URL),
        #     "default_model": cls._get_default_model("lmstudio"),
        #     "requires_api_key": False,
        #     "requires_base_url": True
        # }
        
        return providers