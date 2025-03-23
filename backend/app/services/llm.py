import asyncio
import json
import logging
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

from app.core.config import settings
from app.llm.base import LLMClient
from app.llm.factory import LLMFactory
from app.rag.hybrid_retriever import HybridRetriever
from app.services.chat import ChatService
from app.services.llm_config import LLMConfigService
from sqlalchemy.orm import Session

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMService:
    """
    Service for interacting with LLMs.
    """

    def __init__(
        self,
        db: Session,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ):
        """
        Initialize the LLM service.

        Args:
            db: Database session
            provider: LLM provider name (optional, uses active config if not provided)
            model: Model name (optional, uses active config if not provided)
            system_prompt: Global system prompt (optional, uses active config if not provided)
                       This prompt is used for all LLM providers
            api_key: API key (optional, uses active config if not provided)
            base_url: Base URL (optional, uses active config if not provided)
            embedding_model: Embedding model name (optional, uses active config if not provided)
        """
        self.db = db

        # Get active configuration from database
        active_config = LLMConfigService.get_active_config(db)

        # Use provided values or fall back to active config or defaults
        self.provider = provider or (
            active_config.provider if active_config else settings.DEFAULT_LLM_PROVIDER
        )
        self.model = model or (
            active_config.model if active_config else settings.DEFAULT_CHAT_MODEL
        )
        self.system_prompt = system_prompt or (
            active_config.system_prompt
            if active_config
            else settings.DEFAULT_SYSTEM_PROMPT
        )
        self.api_key = api_key or (active_config.api_key if active_config else None)
        self.base_url = base_url or (active_config.base_url if active_config else None)
        self.embedding_model = embedding_model or (
            active_config.embedding_model if active_config else None
        )

        # Create LLM client
        self.client = LLMFactory.create_client(
            provider=self.provider,
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
            embedding_model=self.embedding_model,
        )

        # Create retriever for RAG
        self.retriever = HybridRetriever(db)

    async def chat(
        self,
        chat_id: str,
        user_message: str,
        use_rag: bool = True,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = True,
    ) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        """
        Send a message to the LLM and get a response.

        Args:
            chat_id: Chat ID
            user_message: User message
            use_rag: Whether to use RAG for context
            temperature: Temperature for generation
            max_tokens: Maximum number of tokens to generate
            stream: Whether to stream the response

        Returns:
            Response from the LLM or an async generator for streaming
        """
        # Get chat history
        messages = ChatService.get_messages(self.db, chat_id)

        # Prepare system prompt
        system_message = self.system_prompt
        logger.info(f"Using system prompt: {system_message[:100]}...")

        # Add RAG context if enabled
        context_documents = None
        if use_rag:
            # Get relevant documents
            context_documents = await self._get_rag_context(user_message)

            if context_documents:
                # Create context message and append to system prompt
                context_text = "\n\nHere is some relevant information that may help you answer the user's question:\n\n"
                for i, doc in enumerate(context_documents):
                    context_text += f"[{i+1}] {doc['content']}\n\n"

                context_text += "Please use this information to help answer the user's question. If the information doesn't contain the answer, just say so."

                # Combine with system prompt
                system_message += context_text
                logger.info(
                    f"Added RAG context to system prompt. Combined length: {len(system_message)}"
                )

        # Format messages for the LLM
        formatted_messages = [self.client.format_chat_message("system", system_message)]

        # Add chat history
        for msg in messages:
            formatted_messages.append(
                self.client.format_chat_message(msg.role, msg.content)
            )

        # Add user message
        formatted_messages.append(self.client.format_chat_message("user", user_message))

        # Log message count and roles for debugging
        roles = [msg["role"] for msg in formatted_messages]
        logger.info(
            f"Sending {len(formatted_messages)} messages to LLM. Roles: {roles}"
        )

        # Log the full prompt including context for debugging if LLM_DEBUG_LOGGING is enabled
        if settings.LLM_DEBUG_LOGGING:
            logger.info("Full prompt with context (LLM_DEBUG_LOGGING enabled):")
            for i, msg in enumerate(formatted_messages):
                # Log each message separately to avoid log size limitations
                logger.info(f"Message {i} - Role: {msg['role']}")

                # For system messages that might contain RAG context, log in detail
                if msg["role"] == "system" and context_documents:
                    # Log the system message in chunks to avoid log size limitations
                    content_chunks = [
                        msg["content"][i : i + 1000]
                        for i in range(0, len(msg["content"]), 1000)
                    ]
                    for j, chunk in enumerate(content_chunks):
                        logger.info(f"System message chunk {j}: {chunk}")

                    # Log RAG hits specifically
                    logger.info(
                        f"RAG context included in prompt: {len(context_documents)} documents"
                    )
                    for i, doc in enumerate(context_documents):
                        logger.info(
                            f"RAG hit {i+1}: ID={doc.get('id', 'unknown')}, Source={doc.get('source', 'unknown')}, Score={doc.get('score', 0)}"
                        )
                        # Log a preview of the content
                        content = doc.get("content", "")
                        logger.info(
                            f"RAG hit {i+1} content preview: {content[:200]}..."
                        )
                else:
                    # For non-system messages or when no RAG context, log a preview
                    content_preview = (
                        msg["content"][:200] + "..."
                        if len(msg["content"]) > 200
                        else msg["content"]
                    )
                    logger.info(f"Content preview: {content_preview}")
        else:
            # Even if detailed logging is disabled, still log basic RAG information
            if context_documents:
                logger.info(
                    f"RAG context included in prompt: {len(context_documents)} documents"
                )
                for i, doc in enumerate(context_documents):
                    logger.info(
                        f"RAG hit {i+1}: ID={doc.get('id', 'unknown')}, Source={doc.get('source', 'unknown')}, Score={doc.get('score', 0)}"
                    )

        # Note: We don't add the user message to the database here
        # because it's already added in the stream_from_llm_get function
        # in backend/app/api/routes/chats.py

        # Start time for calculating tokens per second
        start_time = time.time()

        # Generate response
        if stream:
            return self._stream_response(
                chat_id, formatted_messages, temperature, max_tokens, context_documents
            )
        else:
            # Get response from LLM
            response = await self.client.generate(
                formatted_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
            )

            # Save assistant message to database
            ChatService.add_message(
                self.db,
                chat_id,
                "assistant",
                response["content"],
                tokens=response.get("tokens"),
                tokens_per_second=response.get("tokens_per_second"),
                model=response.get("model"),
                provider=response.get("provider"),
                context_documents=(
                    [doc["id"] for doc in context_documents]
                    if context_documents
                    else None
                ),
            )

            return response

    async def _stream_response(
        self,
        chat_id: str,
        formatted_messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: Optional[int],
        context_documents: Optional[List[Dict[str, Any]]],
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream response from the LLM.

        Args:
            chat_id: Chat ID
            formatted_messages: Formatted messages for the LLM
            temperature: Temperature for generation
            max_tokens: Maximum number of tokens to generate
            context_documents: Context documents from RAG

        Yields:
            Chunks of the response
        """
        logger.debug(f"Starting streaming response for chat {chat_id}")

        # Initialize variables for tracking the response
        full_content = ""
        tokens = 0
        tokens_per_second = 0
        model = self.model or settings.DEFAULT_CHAT_MODEL
        provider = self.provider
        chunk_count = 0

        try:
            # Get streaming response from LLM with timeout
            logger.debug(f"Requesting streaming response from LLM client")
            timeout = settings.RAG_INDEX_BUILD_TIMEOUT

            try:
                response_stream = await asyncio.wait_for(
                    self.client.generate(
                        formatted_messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stream=True,
                    ),
                    timeout=60,  # 60 second timeout
                )

                logger.debug(f"Got response_stream from LLM client")
                logger.debug(f"Starting to iterate through response_stream chunks")

                async for chunk in response_stream:
                    chunk_count += 1

                    # Update tracking variables
                    full_content = chunk["content"]
                    tokens = chunk.get("tokens", 0)
                    tokens_per_second = chunk.get("tokens_per_second", 0)
                    model = chunk.get("model", model)
                    provider = chunk.get("provider", provider)
                    done = chunk.get("done", False)

                    # Log every 5th chunk to avoid excessive logging
                    if chunk_count % 5 == 0 or done:
                        logger.debug(
                            f"Streaming chunk {chunk_count} for chat {chat_id}: {full_content[-30:] if len(full_content) > 30 else full_content}... (done: {done})"
                        )

                    # Yield the chunk immediately
                    logger.debug(f"Yielding chunk {chunk_count} at {time.time()}")
                    yield chunk

                    # Add a very small delay to prevent overwhelming the client
                    if chunk_count % 10 == 0:
                        await asyncio.sleep(0.01)  # 10ms delay every 10 chunks
                    else:
                        await asyncio.sleep(0)  # Yield to event loop but don't delay

                    # Save the message when done
                    if done:
                        logger.debug(
                            f"Stream complete for chat {chat_id}, saving final message"
                        )
                        ChatService.add_message(
                            self.db,
                            chat_id,
                            "assistant",
                            full_content,
                            tokens=tokens,
                            tokens_per_second=tokens_per_second,
                            model=model,
                            provider=provider,
                            context_documents=(
                                [doc["id"] for doc in context_documents]
                                if context_documents
                                else None
                            ),
                        )
                        logger.debug(f"Final message saved for chat {chat_id}")

            except asyncio.TimeoutError:
                logger.error(
                    f"Timeout waiting for initial response from LLM for chat {chat_id}"
                )
                error_message = "The LLM took too long to start generating a response. Please try again."
                error_chunk = {"content": error_message, "error": True, "done": True}
                yield error_chunk

                # Save the error message to the chat
                ChatService.add_message(
                    self.db,
                    chat_id,
                    "assistant",
                    error_message,
                    model=model,
                    provider=provider,
                    context_documents=(
                        [doc["id"] for doc in context_documents]
                        if context_documents
                        else None
                    ),
                )
                return

        except asyncio.TimeoutError:
            logger.error(f"Timeout while streaming response for chat {chat_id}")
            error_message = "The response took too long to generate. This might be due to high server load or complexity of the query with RAG processing."
            error_chunk = {"content": error_message, "error": True, "done": True}
            yield error_chunk

            # Save the error message to the chat
            ChatService.add_message(
                self.db,
                chat_id,
                "assistant",
                error_message,
                model=model,
                provider=provider,
                context_documents=(
                    [doc["id"] for doc in context_documents]
                    if context_documents
                    else None
                ),
            )

        except Exception as e:
            logger.exception(
                f"Error in streaming response for chat {chat_id}: {str(e)}"
            )
            error_message = f"An error occurred while generating the response: {str(e)}"
            error_chunk = {"content": error_message, "error": True, "done": True}
            yield error_chunk

            # Save the error message to the chat
            ChatService.add_message(
                self.db,
                chat_id,
                "assistant",
                error_message,
                model=model,
                provider=provider,
                context_documents=(
                    [doc["id"] for doc in context_documents]
                    if context_documents
                    else None
                ),
            )

    async def _get_rag_context(
        self, query: str, top_k: int = 3
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Get relevant context for RAG.

        Args:
            query: User query
            top_k: Number of results to return

        Returns:
            List of relevant documents or None if no results
        """
        try:
            # Generate query embedding using the LLM client
            logger.info(f"Generating embedding for query: {query[:50]}...")
            query_embedding = None
            try:
                # Get embedding for the query
                embeddings = await self.client.get_embeddings([query])
                if embeddings and len(embeddings) > 0:
                    query_embedding = embeddings[0]
                    logger.info(
                        f"Successfully generated query embedding with dimension {len(query_embedding)}"
                    )
                else:
                    logger.warning("Failed to generate query embedding: empty result")
            except Exception as embed_error:
                logger.error(f"Error generating query embedding: {str(embed_error)}")
                # Continue with keyword search even if embedding fails

            # Retrieve relevant documents
            results = await self.retriever.retrieve(
                query=query, query_embedding=query_embedding, top_k=top_k
            )

            if not results:
                return None

            # Format results for context
            context = []
            for result in results:
                context.append(
                    {
                        "id": result.get("id"),
                        "content": result.get("content"),
                        "score": result.get("score", 0),
                        "source": result.get("source"),
                    }
                )

            return context
        except Exception as e:
            logger.error(f"Error getting RAG context: {str(e)}")
            return None

    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for a list of texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        return await self.client.get_embeddings(texts)

    async def get_available_models(self) -> tuple[List[str], List[str]]:
        """
        Get available models for the current provider.

        Returns:
            Tuple of (chat_models, embedding_models)
        """
        chat_models = []
        embedding_models = []

        try:
            # OpenAI models
            if self.provider == "openai":
                chat_models = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]
                embedding_models = [
                    "text-embedding-ada-002",
                    "text-embedding-3-small",
                    "text-embedding-3-large",
                ]

            # Ollama models - use the client to get actual models
            elif self.provider == "ollama":
                if hasattr(self.client, "list_models"):
                    models = await self.client.list_models()
                    # For Ollama, all models can be used for both chat and embeddings
                    chat_models = models
                    embedding_models = models
                else:
                    # Fallback if list_models is not implemented
                    chat_models = ["llama2", "llama3", "mistral", "mixtral"]
                    embedding_models = ["llama2", "llama3"]

            # Anthropic models
            elif self.provider == "anthropic":
                chat_models = [
                    "claude-2",
                    "claude-instant-1",
                    "claude-3-opus",
                    "claude-3-sonnet",
                ]
                embedding_models = []

            # OpenRouter models
            elif self.provider == "openrouter":
                chat_models = [
                    "openai/gpt-3.5-turbo",
                    "openai/gpt-4",
                    "anthropic/claude-2",
                ]
                embedding_models = ["openai/text-embedding-ada-002"]

            # Deepseek models
            elif self.provider == "deepseek":
                chat_models = ["deepseek-chat", "deepseek-coder"]
                embedding_models = ["deepseek-embedding"]

            return chat_models, embedding_models
        except Exception as e:
            logger.error(f"Error getting available models: {str(e)}")
            return [], []
