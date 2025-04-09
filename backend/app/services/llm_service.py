from typing import List, Dict, Any, Optional, Union, AsyncGenerator
import logging
from sqlalchemy.orm import Session
import time
from contextlib import suppress # For handling potential describe errors gracefully
import json
import asyncio
import google.generativeai as genai # Import google library

from app.llm.factory import LLMFactory
from app.llm.base import LLMClient
# Import specific clients for type checking
from app.llm.anthropic_client import AnthropicClient
from app.llm.google_gemini_client import GoogleGeminiClient
from app.services.chat import ChatService
from app.services.llm_config import LLMConfigService
from app.services.embedding_config import EmbeddingConfigService
from app.services.reranking_config import RerankingConfigService
# Import MCP config service and functions
from app.services.mcp_config_service import MCPConfigService, mcp_session_manager # Import manager
from app.rag.hybrid_retriever import HybridRetriever
from app.core.config import settings
# Import the extracted functions
from .llm_rag import get_rag_context
from .llm_stream import stream_llm_response

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


MAX_TOOL_TURNS = 5 # Maximum number of LLM <-> Tool execution cycles per user message

# --- Define KNOWN Schemas for specific MCP Servers ---
# Used for servers that don't support dynamic mcp.describe
# ALL MCP SERVERS SUPPORT mcp.describe
KNOWN_MCP_TOOL_SCHEMAS = {
    "fetch": [ # Server name matches config name
        {
            "name": "fetch", # Tool name
            "description": "Fetches a URL from the internet and optionally extracts its contents as markdown.",
            "input_schema": { # Use input_schema as per MCP spec
                "type": "object",
                "properties": {
                    "url": {
                        "description": "URL to fetch",
                        "format": "uri",
                        "minLength": 1,
                        "title": "Url",
                        "type": "string"
                    },
                    "max_length": {
                        "default": 5000,
                        "description": "Maximum number of characters to return.",
                        "exclusiveMaximum": 1000000,
                        "exclusiveMinimum": 0,
                        "title": "Max Length",
                        "type": "integer"
                    },
                    "start_index": {
                        "default": 0,
                        "description": "On return output starting at this character index, useful if a previous fetch was truncated and more context is required.",
                        "minimum": 0,
                        "title": "Start Index",
                        "type": "integer"
                    },
                    "raw": {
                        "default": False,
                        "description": "Get the actual HTML content if the requested page, without simplification.",
                        "title": "Raw",
                        "type": "boolean"
                    }
                },
                "required": ["url"],
                "title": "Fetch"
            }
        }
    ]
    # Add other known schemas here if needed
}
# ---

class LLMService:
    """
    Service for interacting with LLMs. Orchestrates RAG and streaming.
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
        temperature: Optional[float] = None, # Added temperature to init args (optional)
        user_id: Optional[str] = None # <-- Add user_id
    ):
        """
        Initialize the LLM service.
        """
        self.db = db
        self.user_id = user_id # <-- Store user_id

        # Get active configurations from database
        chat_config = LLMConfigService.get_active_config(db)
        embedding_config = EmbeddingConfigService.get_active_config(db)

        # Use provided values or fall back to active config or defaults
        self.provider = provider or (chat_config.chat_provider if chat_config else settings.DEFAULT_LLM_PROVIDER)
        self.model = model or (chat_config.model if chat_config else settings.DEFAULT_CHAT_MODEL)
        self.system_prompt = system_prompt or (chat_config.system_prompt if chat_config else settings.DEFAULT_SYSTEM_PROMPT)
        self.api_key = api_key or (chat_config.api_key if chat_config else None)
        self.base_url = base_url or (chat_config.base_url if chat_config else None)
        # Fetch temperature from config or use provided/default
        self.temperature = temperature if temperature is not None else (chat_config.temperature if chat_config and chat_config.temperature is not None else 0.7)

        # Embedding configuration
        self.embedding_model = embedding_model or (embedding_config.model if embedding_config else None)
        embedding_provider = embedding_config.provider if embedding_config else None
        embedding_api_key = embedding_config.api_key if embedding_config else None
        embedding_base_url = embedding_config.base_url if embedding_config else None

        # Determine correct base URLs to pass based on provider.
        chat_base_url_to_pass = None
        if self.provider == 'ollama':
            chat_base_url_to_pass = self.base_url
            logger.info(f"Using configured base_url '{chat_base_url_to_pass}' for Ollama chat client.")
        else:
            logger.info(f"Ignoring configured base_url for non-Ollama chat provider '{self.provider}'. Using default.")

        embedding_base_url_to_pass = None
        if embedding_provider == 'ollama':
            embedding_base_url_to_pass = embedding_base_url
            logger.info(f"Using configured base_url '{embedding_base_url_to_pass}' for Ollama embedding client.")
        else:
            logger.info(f"Ignoring configured base_url for non-Ollama embedding provider '{embedding_provider}'. Using default.")

        # Create LLM clients using separate configurations
        if chat_config and embedding_config:
            client_result = LLMFactory.create_separate_clients(
                chat_config={
                    'provider': self.provider,
                    'model': self.model,
                    'api_key': self.api_key,
                    'base_url': chat_base_url_to_pass,
                },
                embedding_config={
                    'provider': embedding_provider,
                    'model': self.embedding_model,
                    'api_key': embedding_api_key,
                    'base_url': embedding_base_url_to_pass,
                },
                user_id=self.user_id # Pass user_id here
            )
        else:
            client_result = LLMFactory.create_client(
                provider=self.provider,
                model=self.model,
                api_key=self.api_key,
                base_url=chat_base_url_to_pass,
                embedding_model=self.embedding_model,
                embedding_provider=embedding_provider,
                user_id=self.user_id # Pass user_id here
            )

        # Handle single client or separate clients
        if isinstance(client_result, tuple):
            self.chat_client, self.embedding_client = client_result
        else:
            self.chat_client = self.embedding_client = client_result

        # Log user_id immediately after client assignment
        logger.debug(f"LLMService.__init__: Assigned chat_client with user_id={getattr(self.chat_client, 'user_id', 'MISSING')}")

        # Create retriever for RAG
        self.retriever = HybridRetriever(db)

    def _format_tool_for_prompt(self, tool_name: str, description: str, input_schema: Dict[str, Any]) -> str:
        """Formats tool details for inclusion in the system prompt."""
        args_desc = []
        if input_schema and "properties" in input_schema:
            for param_name, param_info in input_schema.get("properties", {}).items():
                arg_desc = f'- {param_name}: {param_info.get("description", "No description")}'
                if param_name in input_schema.get("required", []):
                    arg_desc += " (required)"
                args_desc.append(arg_desc)

        formatted_args = "\\n".join(args_desc) if args_desc else "No arguments."

        return (
            f"\\nTool: {tool_name}\\n"
            f"Description: {description}\\n"
            f"Arguments:\\n{formatted_args}\\n"
        )

    async def chat(
        self,
        chat_id: str,
        user_message: str,
        use_rag: bool = True,
        max_tokens: Optional[int] = None,
        stream: bool = True,
        completion_state: Dict[str, Any] = None # Added state dict parameter
    ) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        """
        Send a message to the LLM and get a response orchestrating RAG and streaming.
        If streaming updates the provided completion_state dictionary.
        """
        # Get chat history
        messages = ChatService.get_messages(self.db, chat_id)

        # Prepare system prompt
        current_system_prompt = self.system_prompt # Use the instance's system prompt
        logger.info(f"Using system prompt: {current_system_prompt[:100]}...")

        # Add RAG context if enabled
        context_documents = None
        if use_rag:
            context_documents = await get_rag_context(
                db=self.db, embedding_client=self.embedding_client,
                retriever=self.retriever, query=user_message
            )
            if context_documents:
                context_text = "\\n\\nHere is some relevant information that may help you answer the user's question:\\n\\n"
                for i, doc in enumerate(context_documents): context_text += f"[{i+1}] {doc['content']}\\n\\n"
                context_text += "Please use this information to help answer the user's question. If the information doesn't contain the answer just say so."
                current_system_prompt += context_text
                logger.info(f"Added RAG context to system prompt. Combined length: {len(current_system_prompt)}")

        # --- Tool Handling for Prompt Injection ---
        prompt_tools_details = [] # List to hold dicts like {'name': ..., 'description': ..., 'schema': ...}
        tool_name_map = {} # Map unique_tool_name -> actual_tool_name for execution
        enabled_mcp_configs = [] # Keep track of configs for later execution

        if self.user_id:
            try:
                # Get all enabled MCP configs for this user
                enabled_mcp_configs = [c for c in MCPConfigService.get_configs_by_user(self.db, self.user_id) if c.config and c.config.get('enabled', False)]
                logger.info(f"Found {len(enabled_mcp_configs)} enabled MCP servers for user {self.user_id}")
                if settings.LLM_DEBUG_LOGGING:
                    for cfg in enabled_mcp_configs: logger.debug(f"  - Enabled MCP Config: ID={cfg.id}, Name={cfg.name}, Command={cfg.config.get('command')}")

                # Gather tool details from KNOWN_MCP_TOOL_SCHEMAS
                for config in enabled_mcp_configs:
                    server_name = config.name.lower()
                    if server_name in KNOWN_MCP_TOOL_SCHEMAS:
                        known_schemas = KNOWN_MCP_TOOL_SCHEMAS[server_name]
                        logger.info(f"Processing known schema for server '{config.name}'. Found {len(known_schemas)} tool(s).")
                        for tool_schema in known_schemas:
                            tool_name = tool_schema.get("name")
                            description = tool_schema.get("description")
                            input_schema = tool_schema.get("input_schema")
                            if tool_name and description and input_schema:
                                unique_tool_name = f"{config.name.replace('-', '_')}__{tool_name}"
                                prompt_tools_details.append({
                                    "name": unique_tool_name,
                                    "description": description,
                                    "schema": input_schema
                                })
                                tool_name_map[unique_tool_name] = tool_name
                                logger.debug(f"Gathered known tool for prompt: {unique_tool_name}")
                            else:
                                logger.warning(f"Skipping invalid known tool schema for server '{config.name}': {tool_schema}")
                    else:
                         # Dynamically fetch tools (if needed in future, add logic here)
                         logger.info(f"Server '{config.name}' not in KNOWN_MCP_TOOL_SCHEMAS, skipping dynamic fetch for now.")
                         pass # Placeholder for dynamic fetching if required

            except Exception as e:
                logger.error(f"Failed to fetch or format MCP tools: {e}")
                prompt_tools_details = []
                tool_name_map = {}
        else:
            logger.warning("No user_id provided, cannot fetch MCP tools.")

        # --- Generate Tool Descriptions and Instructions for System Prompt ---
        tools_description_for_prompt = ""
        if prompt_tools_details:
            logger.info(f"Formatting {len(prompt_tools_details)} tools for system prompt.")
            tools_description_for_prompt += "You have access to the following tools:\\n"
            for tool_detail in prompt_tools_details:
                tools_description_for_prompt += self._format_tool_for_prompt(
                    tool_detail["name"],
                    tool_detail["description"],
                    tool_detail["schema"]
                )
            tools_description_for_prompt += "\\nIMPORTANT: When you need to use a tool, you MUST respond ONLY with the following JSON structure, replacing placeholders:\\n"
            tools_description_for_prompt += '{\\n    "tool": "<tool_name_to_call>",\\n    "arguments": {\\n        "<argument_name>": "<value>",\\n        ...\\n    }\\n}\\n'
            tools_description_for_prompt += "Do not add any other text before or after the JSON object."

            # Append to the main system prompt
            current_system_prompt += "\\n\\n" + tools_description_for_prompt
            logger.info("Appended tool descriptions and instructions to system prompt.")
        else:
            logger.info("No tools available or gathered for system prompt.")
        # --- End Tool Prompt Generation ---

        # Format messages for the LLM including history
        formatted_messages = [self.chat_client.format_chat_message("system", current_system_prompt)]
        for msg in messages:
            # Format message based on role and add tool data if present
            # NOTE: We now rely on parsing content for tool calls, but still need to handle tool *results*
            if msg.role == "tool" and msg.tool_call_id:
                # Format tool result message
                formatted_messages.append(self.chat_client.format_chat_message(
                    "tool",
                    msg.content,
                    tool_call_id=msg.tool_call_id,
                    name=msg.name # Use the original tool name from the DB
                ))
            elif msg.role == "assistant" and msg.tool_calls:
                 # If assistant message had tool_calls saved (from previous API format or parsed JSON)
                 # Format it for history, but don't expect LLM to output this structure now
                 formatted_messages.append(self.chat_client.format_chat_message(
                     "assistant",
                     msg.content, # Include any text content
                     tool_calls=msg.tool_calls # Include the saved structure for context
                 ))
            else:
                # Regular message formatting
                formatted_messages.append(self.chat_client.format_chat_message(msg.role, msg.content))

        # Logging
        roles = [msg["role"] for msg in formatted_messages]
        logger.info(f"Sending {len(formatted_messages)} messages to LLM. Roles: {roles}")
        if settings.LLM_DEBUG_LOGGING:
            if context_documents: logger.info(f"RAG context included: {len(context_documents)} documents")
            logger.info(f"System prompt includes tool instructions: {tools_description_for_prompt != ''}")
        elif context_documents: logger.info(f"RAG context included: {len(context_documents)} documents")


        # Generate response
        if stream:
            # Log the user_id attribute of the client being passed
            logger.debug(f"LLMService.chat [STREAM]: Passing chat_client with user_id={getattr(self.chat_client, 'user_id', 'MISSING')}")
            logger.debug(f"LLMService.chat [STREAM]: Starting stream generation. Tools NOT passed via API.")
            # Return awaitable generator to be awaited by the caller
            # REMOVE 'tools' argument as we use prompt injection now
            stream_generator = stream_llm_response(
                db=self.db,
                chat_client=self.chat_client,
                chat_id=chat_id,
                formatted_messages=formatted_messages,
                temperature=self.temperature,
                max_tokens=max_tokens,
                context_documents=context_documents,
                system_prompt=current_system_prompt, # System prompt now contains tool info
                model=self.model,
                provider=self.provider,
                # tools=None # Explicitly None
            )
            logger.debug("LLMService.chat [STREAM]: Returning stream generator.")
            return stream_generator # Return the awaitable generator directly
        else:
            # --- Non-Streaming Multi-Turn Logic (Adapted for Prompt Injection) ---
            current_response_dict = None
            # Make a copy of messages to modify within the loop
            current_formatted_messages = list(formatted_messages)
            # Keep track of enabled configs for tool execution mapping
            configs_map = {cfg.name.replace('-', '_'): cfg.id for cfg in enabled_mcp_configs}

            for turn in range(MAX_TOOL_TURNS):
                logger.info(f"Non-Streaming Tool Turn {turn + 1}/{MAX_TOOL_TURNS}")
                start_time = time.time()

                try:
                    # REMOVE 'tools' and 'tool_choice' arguments
                    current_response_dict = await self.chat_client.generate(
                        current_formatted_messages, # Use the potentially updated message list
                        temperature=self.temperature, max_tokens=max_tokens,
                        stream=False,
                        # tools=None, # Explicitly None
                        # tool_choice=None # Explicitly None
                    )
                except Exception as llm_error:
                    logger.exception(f"LLM generation failed on turn {turn + 1}: {llm_error}")
                    error_content = f"An error occurred while communicating with the AI model: {str(llm_error)}"
                    # Save error message before returning
                    ChatService.add_message(self.db, chat_id, "assistant", error_content, finish_reason="error", model=self.model, provider=self.provider)
                    return {"content": error_content, "finish_reason": "error"}

                end_time = time.time()
                duration = end_time - start_time

                # --- Tool Call Handling via Prompt ---
                content = current_response_dict.get("content")
                tool_calls_to_execute = [] # List to hold validated tool calls for this turn
                is_tool_call_request = False

                if content:
                    try:
                        # Attempt to load the entire content as JSON
                        potential_tool_call = json.loads(content.strip()) # Strip whitespace
                        if isinstance(potential_tool_call, dict) and \
                           "tool" in potential_tool_call and \
                           "arguments" in potential_tool_call and \
                           isinstance(potential_tool_call.get("arguments"), dict):

                            logger.info(f"Detected tool call JSON in content: {potential_tool_call}")
                            is_tool_call_request = True

                            # Generate a unique ID for this call
                            tool_call_id = f"call_{int(time.time())}_{turn}"

                            # Reconstruct the tool_calls structure expected by downstream code/DB
                            # Use the unique name from the prompt (which the LLM should return)
                            unique_tool_name = potential_tool_call["tool"]
                            arguments_obj = potential_tool_call["arguments"]

                            # Map back to original tool name for execution if possible
                            original_tool_name = tool_name_map.get(unique_tool_name, unique_tool_name)
                            logger.info(f"Mapped unique name '{unique_tool_name}' back to '{original_tool_name}' for execution.")

                            parsed_tool_call_structure = {
                                "id": tool_call_id,
                                "type": "function", # Assume function type
                                "function": {
                                    "name": unique_tool_name, # Store the name LLM used
                                    "arguments": arguments_obj # Store arguments dict
                                }
                            }
                            tool_calls_to_execute.append({
                                "id": tool_call_id,
                                "unique_name": unique_tool_name, # Name LLM used
                                "original_name": original_tool_name, # Name MCP server expects
                                "arguments": arguments_obj # Arguments dict
                            })
                            # Save the assistant message that contained the JSON tool call
                            ChatService.add_message(
                                 self.db, chat_id, "assistant",
                                 content="", # Use empty string instead of None for content
                                 tool_calls=[parsed_tool_call_structure], # Save the structured call
                                 finish_reason="tool_calls",
                                 model=current_response_dict.get("model", self.model),
                                 provider=current_response_dict.get("provider", self.provider)
                                 # Add token usage if available in current_response_dict
                            )
                            # Add the assistant's "request" message to history for next turn
                            current_formatted_messages.append(self.chat_client.format_chat_message(
                                "assistant",
                                None, # No text content
                                tool_calls=[parsed_tool_call_structure] # Use the structure with ID
                            ))
                            content = None # Clear content as it was processed as a tool call

                        else:
                            logger.debug("Content parsed as JSON but doesn't match tool call structure.")
                    except json.JSONDecodeError:
                        logger.debug("Content is not valid JSON, treating as regular text response.")
                    except Exception as parse_err:
                        logger.error(f"Unexpected error parsing content for tool call: {parse_err}")
                # --- End Tool Call Handling via Prompt ---

                # Calculate usage stats (might need adjustment if not directly provided)
                usage = current_response_dict.get("usage", {})
                completion_tokens = usage.get("completion_tokens", len(content.split()) if content else 0) # Estimate if needed
                prompt_tokens = usage.get("prompt_tokens", 0)
                total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)
                tokens_per_second = completion_tokens / duration if completion_tokens and duration > 0 else 0.0
                finish_reason = current_response_dict.get("finish_reason", "stop") # Default to stop
                response_model = current_response_dict.get("model", self.model)
                response_provider = current_response_dict.get("provider", self.provider)

                if is_tool_call_request and tool_calls_to_execute:
                    logger.info(f"Executing {len(tool_calls_to_execute)} parsed tool calls.")
                    # 2. Execute Tools and Collect Results
                    tool_results_messages = []
                    if not self.user_id:
                         logger.error("Cannot execute tools: user_id missing.")
                         # Return the raw response dict as we can't proceed
                         return current_response_dict

                    tool_execution_tasks = []
                    for tool_call_data in tool_calls_to_execute:
                        tool_call_id = tool_call_data["id"]
                        unique_tool_name = tool_call_data["unique_name"] # Name LLM used
                        original_tool_name = tool_call_data["original_name"] # Name MCP server expects
                        arguments_obj = tool_call_data["arguments"]

                        # Find the corresponding server config_id using the unique name prefix
                        server_name_prefix = unique_tool_name.split("__")[0]
                        config_id = configs_map.get(server_name_prefix)

                        if not config_id:
                            logger.error(f"Could not find MCP config for tool prefix: {server_name_prefix} from unique name {unique_tool_name}")
                            tool_result_content_str = json.dumps({"error": {"message": f"Config for tool '{unique_tool_name}' not found."}})
                            # Save error message to DB
                            ChatService.add_message(self.db, chat_id, "tool", content=tool_result_content_str, tool_call_id=tool_call_id, name=unique_tool_name)
                            # Prepare error message for next LLM turn
                            tool_message_for_llm = self.chat_client.format_chat_message("tool", tool_result_content_str, tool_call_id=tool_call_id, name=unique_tool_name)
                            tool_results_messages.append(tool_message_for_llm)
                            continue # Skip execution for this tool

                        # Ensure arguments_str is always a JSON string for execute_mcp_tool
                        try:
                            arguments_str = json.dumps(arguments_obj)
                        except TypeError as e:
                            logger.error(f"Failed to dump tool arguments to JSON: {e}. Args: {arguments_obj}")
                            arguments_str = "{}" # Default on error

                        tool_execution_tasks.append(
                            asyncio.to_thread( # Run sync execute_mcp_tool in thread
                                MCPConfigService.execute_mcp_tool,
                                # Remove db=self.db as it's not expected by execute_mcp_tool
                                config_id=config_id, tool_call_id=tool_call_id,
                                tool_name=original_tool_name, # Use original name for execution
                                arguments_str=arguments_str
                            )
                        )

                    if tool_execution_tasks:
                        # Execute all tool calls concurrently
                        tool_results = await asyncio.gather(*tool_execution_tasks, return_exceptions=True)

                        # Process results and save them
                        for i, result in enumerate(tool_results):
                            # Get corresponding tool call data
                            executed_tool_data = tool_calls_to_execute[i]
                            tool_call_id = executed_tool_data["id"]
                            unique_tool_name = executed_tool_data["unique_name"] # Use the name LLM knows

                            if isinstance(result, Exception):
                                logger.error(f"Exception during tool execution for {unique_tool_name}: {result}")
                                tool_result_content_str = json.dumps({"error": {"message": f"Error executing tool {unique_tool_name}: {str(result)}"}})
                            elif isinstance(result, dict) and "result" in result:
                                # Extract the JSON string from the 'result' key
                                tool_result_content_str = result.get("result")
                                if not isinstance(tool_result_content_str, str):
                                    # If the inner result isn't a string, log error and create error JSON
                                    logger.error(f"Tool execution result's 'result' key for {unique_tool_name} is not a string: {result}")
                                    tool_result_content_str = json.dumps({"error": {"message": f"Tool {unique_tool_name} returned malformed result structure."}})
                            else:
                                # Handle cases where the dict doesn't have 'result' or it's not a dict
                                logger.error(f"Unexpected tool execution result structure for {unique_tool_name}: {result}")
                                tool_result_content_str = json.dumps({"error": {"message": f"Tool {unique_tool_name} returned unexpected result structure."}})


                            # Save tool result message to DB
                            ChatService.add_message(
                                self.db, chat_id, "tool", content=tool_result_content_str,
                                tool_call_id=tool_call_id, name=unique_tool_name # Use unique name LLM knows
                            )
                            # Prepare message for next LLM turn
                            tool_message_for_llm = self.chat_client.format_chat_message(
                                "tool",
                                tool_result_content_str,
                                tool_call_id=tool_call_id,
                                name=unique_tool_name # Use unique name LLM knows
                            )
                            tool_results_messages.append(tool_message_for_llm)

                    # Append all tool result messages to the history for the next turn
                    current_formatted_messages.extend(tool_results_messages)
                    # Continue to the next turn
                    continue
                else:
                    # No tool calls requested or detected, this is the final response
                    logger.info(f"No tool calls requested. Finish reason: {finish_reason}")
                    # Save the final assistant message
                    ChatService.add_message(
                        self.db, chat_id, "assistant", content=content,
                        tokens=total_tokens, prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens, tokens_per_second=tokens_per_second,
                        model=response_model, provider=response_provider,
                        context_documents=[doc["id"] for doc in context_documents] if context_documents else None,
                        finish_reason=finish_reason
                    )
                    # Return the final response dictionary
                    return current_response_dict

            # If loop finishes without returning (e.g., MAX_TOOL_TURNS reached)
            logger.warning(f"Reached maximum tool turns ({MAX_TOOL_TURNS}). Returning last response.")
            # Save the last assistant message if it wasn't saved already and wasn't a tool call request
            if current_response_dict and not is_tool_call_request:
                 ChatService.add_message(
                     self.db, chat_id, "assistant", content=current_response_dict.get("content"),
                     tokens=total_tokens, prompt_tokens=prompt_tokens,
                     completion_tokens=completion_tokens, tokens_per_second=tokens_per_second,
                     model=response_model, provider=response_provider,
                     context_documents=[doc["id"] for doc in context_documents] if context_documents else None,
                     finish_reason="length" # Indicate max turns reached
                 )
            return current_response_dict or {"content": "Maximum tool interaction limit reached.", "finish_reason": "length"}


    async def get_available_models(self) -> tuple[List[str], List[str]]:
        """
        Get available chat and embedding models from the factory.
        """
        try:
            chat_models = await LLMFactory.get_available_chat_models()
            embedding_models = await LLMFactory.get_available_embedding_models()
            return chat_models, embedding_models
        except Exception as e:
            logger.error(f"Error fetching available models: {e}")
            return [], []
