import time
import logging
from typing import List, Dict, Any, Optional, Union, AsyncGenerator
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
# Import Tool type for function calling - Schema/Type might not be needed directly
from google.generativeai.types import Tool, FunctionDeclaration # Removed Schema, Type
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import json # For logging and args stringify
import uuid # For generating tool call IDs
import asyncio # For to_thread

from app.llm.base import LLMClient
from app.core.config import settings

# Configure logging
logger = logging.getLogger(__name__)
if settings.LLM_DEBUG_LOGGING:
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)

# Define retry mechanism for Google API calls (common transient errors)
retry_decorator = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((
        google_exceptions.ServerError,
        google_exceptions.ServiceUnavailable,
        google_exceptions.TooManyRequests,
        google_exceptions.ResourceExhausted, # Can sometimes be transient
    )),
    reraise=True
)

class GoogleGeminiClient(LLMClient):
    """
    Client for interacting with the Google Gemini API.
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None, # Gemini doesn't use base_url
        embedding_model: Optional[str] = None,
        user_id: Optional[str] = None # Add user_id parameter
    ):
        # Use a default embedding model if not provided, specific to Gemini
        default_embedding_model = "models/embedding-001"
        effective_embedding_model = embedding_model or default_embedding_model

        # Pass user_id to base class constructor
        super().__init__(model=model, api_key=api_key, base_url=base_url, embedding_model=effective_embedding_model, user_id=user_id)

        if not self.api_key:
            raise ValueError("Google Gemini API key is required.")

        genai.configure(api_key=self.api_key)

        # Verify the chat model exists
        try:
            # Add safety_settings to potentially reduce blocking
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            ]
            self.chat_model_instance = genai.GenerativeModel(
                self.model,
                safety_settings=safety_settings
            )
            logger.info(f"Google Gemini client initialized with chat model: {self.model}")
        except Exception as e:
            logger.error(f"Failed to initialize Google Gemini chat model '{self.model}': {e}", exc_info=True)
            raise ValueError(f"Invalid Google Gemini chat model specified: {self.model}") from e

        # Verify the embedding model exists
        try:
            # Check if embedding model is valid by trying to get info (no direct instance needed for embeddings API)
            genai.get_model(self.embedding_model)
            logger.info(f"Using Google Gemini embedding model: {self.embedding_model}")
        except Exception as e:
            logger.error(f"Failed to validate Google Gemini embedding model '{self.embedding_model}': {e}", exc_info=True)
            # Fallback or raise error? Let's raise for clarity.
            raise ValueError(f"Invalid Google Gemini embedding model specified: {self.embedding_model}") from e

        if self.base_url:
            logger.warning("base_url is provided but not typically used by the Google Gemini client.")


    def _convert_messages_to_gemini_format(self, messages: List[Dict[str, str]], system_prompt: Optional[str] = None) -> List[Dict[str, Any]]:
        """Converts standard message format to Gemini's format."""
        gemini_messages = []

        # Handle system prompt if provided (Gemini prefers it as the first message or part of the first user message)
        if system_prompt:
             # Prepend system prompt as a separate 'user' message if no user message exists yet,
             # or combine with the first user message if it exists.
             if not messages or messages[0].get("role") != "user":
                 gemini_messages.append({'role': 'user', 'parts': [system_prompt]})
             else:
                 # Combine with the first user message
                 first_user_message_content = f"{system_prompt}\n\n{messages[0].get('content', '')}"
                 gemini_messages.append({'role': 'user', 'parts': [first_user_message_content]})
                 messages = messages[1:] # Remove the first message as it's now combined

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")
            # Handle tool call/result messages
            if role == "assistant" and msg.get("tool_calls"):
                gemini_role = "model" # Assistant is 'model' in Gemini
                parts = []
                
                # Include text content if it exists alongside tool calls
                if content:
                    parts.append({'text': content})
                
                # Process each tool call
                for tool_call in msg.get("tool_calls", []):
                    if tool_call.get("type") == "function":
                        func = tool_call.get("function", {})
                        func_name = func.get("name")
                        
                        # Parse arguments from JSON string to dict
                        try:
                            func_args = json.loads(func.get("arguments", "{}"))
                        except json.JSONDecodeError as e:
                            logger.warning(f"Could not decode tool call arguments for {func_name}: {e}")
                            func_args = {}
                            
                        # Add function call part
                        if func_name:
                            parts.append({'function_call': {'name': func_name, 'args': func_args}})
                
                # Add the message with all parts
                if parts:
                    gemini_messages.append({'role': gemini_role, 'parts': parts})
                continue # Skip further processing of this message
                
            # Handle tool response messages (role='tool')
            elif role == "tool":
                # Tool response becomes a 'function' role in Gemini
                func_name = msg.get("name", "unknown_function") 
                tool_call_id = msg.get("tool_call_id", "unknown_id")
                
                # Format response properly
                try:
                    # Try to parse content if it's JSON
                    response_obj = json.loads(content) if isinstance(content, str) else content
                except json.JSONDecodeError:
                    # If not valid JSON, wrap as a string result
                    response_obj = {"result": content}
                    
                gemini_messages.append({
                    'role': 'function',
                    'parts': [{
                        'function_response': {
                            'name': func_name,
                            'response': response_obj
                        }
                    }]
                })
                continue # Skip further processing
            if not role: continue # Skip if no role

            # Gemini uses 'user' and 'model' roles
            gemini_role = "user" if role == "user" else "model"

            # Handle potential tool results message (role='tool') -> Gemini 'function' role
            if role == "tool":
                 gemini_role = "function" # Gemini uses 'function' role for tool results
                 # Content needs to be formatted as FunctionResponse part
                 # Assuming content is the result string and we need the tool_call_id
                 tool_call_id = msg.get("tool_call_id", "unknown_tool_call_id") # Need tool_call_id from original request
                 function_name = msg.get("name", "unknown_function") # Need function name
                 gemini_messages.append({
                     'role': gemini_role,
                     'parts': [{
                         'function_response': {
                             'name': function_name,
                             'response': {'result': content} # Wrap content in response dict
                         }
                     }]
                 })
                 continue # Skip normal content processing for tool results

            # Handle assistant message with tool calls -> Gemini 'function_call' part
            if role == "assistant" and msg.get("tool_calls"):
                 gemini_role = "model" # Assistant is 'model'
                 parts = []
                 if content: # Include text part if it exists alongside tool calls
                     parts.append({'text': content})
                 for tool_call in msg.get("tool_calls", []):
                     if tool_call.get("type") == "function":
                         func = tool_call.get("function", {})
                         func_name = func.get("name")
                         try:
                             # Gemini expects args as dict, not string
                             func_args = json.loads(func.get("arguments", "{}"))
                         except json.JSONDecodeError:
                             logger.warning(f"Could not decode tool call arguments for {func_name}: {func.get('arguments')}")
                             func_args = {}
                         if func_name:
                             parts.append({'function_call': {'name': func_name, 'args': func_args}})
                 if parts:
                     gemini_messages.append({'role': gemini_role, 'parts': parts})
                 continue # Skip normal content processing

            # Normal user/assistant message with content
            if content:
                # Ensure alternating roles if necessary (Gemini can be strict)
                if gemini_messages and gemini_messages[-1]['role'] == gemini_role:
                    # If consecutive messages have the same role, merge content
                    logger.warning(f"Consecutive messages with role '{gemini_role}'. Merging content.")
                    gemini_messages[-1]['parts'].append({'text': content}) # Ensure it's a text part
                else:
                     gemini_messages.append({'role': gemini_role, 'parts': [{'text': content}]}) # Ensure it's a text part

        return gemini_messages

    # --- Helper to convert OpenAI tool schema to Gemini FunctionDeclaration ---
    def _convert_tools_to_gemini_format(self, tools: List[Dict[str, Any]]) -> Optional[List[Tool]]:
        if not tools:
            return None

        gemini_tools = []
        for tool in tools:
            if tool.get("type") == "function":
                func_data = tool.get("function", {})
                name = func_data.get("name")
                description = func_data.get("description")
                parameters_schema = func_data.get("parameters") # This is already a dict

                if name and description and parameters_schema:
                    try:
                        # Pass the parameter dictionary directly to FunctionDeclaration
                        # Ensure properties are correctly structured if needed
                        # Note: Gemini's internal types might differ slightly, direct dict passing is often okay
                        func_decl = FunctionDeclaration(
                            name=name,
                            description=description,
                            parameters=parameters_schema # Pass the dict directly
                        )
                        gemini_tools.append(Tool(function_declarations=[func_decl]))
                    except Exception as e:
                        logger.error(f"Failed to create Gemini FunctionDeclaration for tool '{name}': {e}. Schema: {parameters_schema}. Skipping tool.")
                else:
                    logger.warning(f"Skipping tool due to missing name, description, or parameters: {tool}")
            else:
                logger.warning(f"Skipping non-function tool type: {tool.get('type')}")

        return gemini_tools if gemini_tools else None
    # ---

    @retry_decorator
    async def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None, # Gemini uses max_output_tokens
        stream: bool = False,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None, # <-- Add tools
        tool_choice: Optional[str] = None # <-- Add tool_choice (maps to tool_config)
    ) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        """
        Generate a response from the Google Gemini model.
        """
        start_time = time.time()
        # Note: _convert_messages needs to handle tool results/calls before this point
        gemini_messages = self._convert_messages_to_gemini_format(messages, system_prompt)

        generation_config = genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens
        )

        gemini_tools = self._convert_tools_to_gemini_format(tools) if tools else None
        tool_config = None
        if gemini_tools:
            # Simplified tool_choice mapping for Gemini
            if tool_choice == "none":
                tool_config = {"function_calling_config": {"mode": "NONE"}}
            elif isinstance(tool_choice, str) and tool_choice != "auto":
                 # Attempt to force a specific function if requested
                 tool_config = {"function_calling_config": {"mode": "ANY", "allowed_function_names": [tool_choice]}}
                 logger.warning(f"Attempting to force Gemini tool '{tool_choice}'. Model support varies.")
            else: # Default to AUTO if tools are present
                tool_config = {"function_calling_config": {"mode": "AUTO"}}


        request_params: Dict[str, Any] = {
            "contents": gemini_messages,
            "generation_config": generation_config,
            "stream": stream
        }
        if gemini_tools: request_params["tools"] = gemini_tools
        if tool_config: request_params["tool_config"] = tool_config

        # Logging (simplified)
        logger.debug(f"Google Gemini request params (contents omitted): { {k:v for k,v in request_params.items() if k != 'contents'} }")
        if settings.LLM_DEBUG_LOGGING:
             for i, msg in enumerate(gemini_messages): logger.info(f"Gemini Message {i}: {msg}")


        try:
            if stream:
                return self._generate_stream(request_params, start_time)
            else:
                # Non-streaming response handling
                response = await self.chat_model_instance.generate_content_async(**request_params)
                logger.debug(f"Google Gemini non-stream response: {response}")

                prompt_tokens = response.usage_metadata.prompt_token_count if response.usage_metadata else 0
                completion_tokens = response.usage_metadata.candidates_token_count if response.usage_metadata else 0
                total_tokens = response.usage_metadata.total_token_count if response.usage_metadata else 0
                tokens_per_second = self.calculate_tokens_per_second(start_time, completion_tokens)
                finish_reason = response.candidates[0].finish_reason.name if response.candidates else "UNKNOWN"

                content = None
                tool_calls = None # Standardized format

                if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                    part = response.candidates[0].content.parts[0]
                    if hasattr(part, 'text'):
                        content = part.text
                    elif hasattr(part, 'function_call'):
                        # Format Gemini function_call to OpenAI tool_calls
                        fc = part.function_call
                        try:
                             args_dict = dict(fc.args)
                             args_str = json.dumps(args_dict)
                        except Exception as e:
                             logger.warning(f"Could not stringify Gemini function call args: {fc.args}. Error: {e}")
                             args_str = "{}"

                        tool_calls = [{
                            "id": f"call_{uuid.uuid4()}", # Generate an ID
                            "type": "function",
                            "function": {"name": fc.name, "arguments": args_str}
                        }]
                        logger.info(f"Gemini returned function call: {tool_calls}")
                        finish_reason = "tool_calls" # Standardized reason

                elif response.prompt_feedback and response.prompt_feedback.block_reason:
                     content = f"Response blocked due to: {response.prompt_feedback.block_reason.name}"
                     logger.warning(f"Gemini response blocked: {response.prompt_feedback.block_reason.name}")
                     finish_reason = "BLOCKED"

                response_data = {
                    "role": "assistant", "content": content, "tool_calls": tool_calls,
                    "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": total_tokens},
                    "tokens_per_second": tokens_per_second, "finish_reason": finish_reason,
                    "model": self.model, "provider": "google_gemini"
                }
                return {k: v for k, v in response_data.items() if v is not None}

        except (google_exceptions.GoogleAPIError, ValueError) as e:
            logger.error(f"Google Gemini API error: {e}", exc_info=True)
            raise
        except Exception as e:
             logger.error(f"Unexpected error during Google Gemini generation: {e}", exc_info=True)
             raise

    async def _generate_stream(
        self,
        request_params: Dict[str, Any],
        start_time: float
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Helper for generating streaming response."""
        completion_tokens = 0
        prompt_tokens = 0
        total_tokens = 0
        finish_reason = None
        accumulated_content = ""
        # --- Tool Call Streaming State ---
        current_tool_calls: Dict[int, Dict[str, Any]] = {} # Accumulate tool calls by index
        # ---

        try:
            response_stream = await self.chat_model_instance.generate_content_async(**request_params)
            first_chunk_processed = False

            async for chunk in response_stream:
                # logger.debug(f"Google Gemini stream chunk: {chunk}") # Very verbose
                yield_chunk: Dict[str, Any] = {"type": "delta", "done": False}
                has_update = False

                # Extract usage if available (usually only in first/last chunk)
                if hasattr(chunk, 'usage_metadata') and chunk.usage_metadata:
                    if chunk.usage_metadata.prompt_token_count > 0 and not first_chunk_processed:
                         prompt_tokens = chunk.usage_metadata.prompt_token_count
                         yield { "type": "start", "role": "assistant", "model": self.model, "usage": {"prompt_tokens": prompt_tokens}}
                         first_chunk_processed = True
                    if chunk.usage_metadata.candidates_token_count > 0: completion_tokens = chunk.usage_metadata.candidates_token_count
                    if chunk.usage_metadata.total_token_count > 0: total_tokens = chunk.usage_metadata.total_token_count

                # Extract content delta or function call delta
                if chunk.candidates and chunk.candidates[0].content and chunk.candidates[0].content.parts:
                    part = chunk.candidates[0].content.parts[0]
                    if hasattr(part, 'text'):
                        delta_content = part.text
                        if delta_content:
                            accumulated_content += delta_content
                            yield_chunk["content"] = delta_content
                            has_update = True
                    elif hasattr(part, 'function_call'):
                         # --- Handle Streaming Function Calls ---
                         fc = part.function_call
                         # Gemini streams function calls differently. It might send name first, then args.
                         # We need to accumulate based on an assumed index (usually 0 for Gemini's current implementation).
                         index = 0 # Assume index 0 for Gemini function calls in stream
                         if index not in current_tool_calls:
                             current_tool_calls[index] = {
                                 "id": f"call_{uuid.uuid4()}", # Generate ID once
                                 "type": "function",
                                 "function": {"name": None, "arguments": ""}
                             }

                         call_part = current_tool_calls[index]
                         delta_update = {}
                         if fc.name and call_part["function"]["name"] is None:
                             call_part["function"]["name"] = fc.name
                             delta_update = {"index": index, "id": call_part["id"], "type": "function", "function": {"name": fc.name}}
                             has_update = True
                         if fc.args:
                             # Args stream chunk by chunk, append them
                             args_str_part = json.dumps(dict(fc.args))[1:-1] # Get partial args string without {}
                             call_part["function"]["arguments"] += args_str_part
                             if "function" not in delta_update: # Ensure function structure exists
                                 delta_update = {"index": index, "id": call_part["id"], "type": "function", "function": {}}
                             delta_update["function"]["arguments"] = args_str_part # Yield only the delta part
                             has_update = True

                         if delta_update:
                             yield_chunk["tool_calls_delta"] = [delta_update]
                         # ---

                # Check for finish reason
                if chunk.candidates and chunk.candidates[0].finish_reason:
                    finish_reason = chunk.candidates[0].finish_reason.name
                    if finish_reason == "FUNCTION_CALLING": finish_reason = "tool_calls"

                # Check for blocking
                if chunk.prompt_feedback and chunk.prompt_feedback.block_reason:
                     block_reason_name = chunk.prompt_feedback.block_reason.name
                     logger.warning(f"Gemini stream blocked: {block_reason_name}")
                     yield_chunk["content"] = f"\n[STREAM BLOCKED DUE TO: {block_reason_name}]"
                     finish_reason = "BLOCKED"
                     yield yield_chunk # Yield block message
                     break # Stop stream

                # Yield delta chunk if it has content or tool delta
                if has_update:
                    yield yield_chunk

            # Ensure start event is sent if missed
            if not first_chunk_processed:
                 yield {"type": "start", "role": "assistant", "model": self.model, "usage": {"prompt_tokens": 0}}
                 logger.warning("Could not determine prompt tokens from Gemini stream.")

            # Yield final message after stream ends
            if total_tokens == 0 and prompt_tokens > 0 and completion_tokens > 0: total_tokens = prompt_tokens + completion_tokens
            tokens_per_second = self.calculate_tokens_per_second(start_time, completion_tokens)

            # Finalize accumulated tool calls arguments string
            final_tool_calls = []
            for index in sorted(current_tool_calls.keys()):
                call = current_tool_calls[index]
                # Ensure arguments is a valid JSON string, wrap if needed
                if call["function"]["arguments"] and not (call["function"]["arguments"].startswith("{") and call["function"]["arguments"].endswith("}")):
                     call["function"]["arguments"] = "{" + call["function"]["arguments"] + "}"
                # Attempt to parse/re-serialize for validation, fallback to raw string
                try:
                    json.loads(call["function"]["arguments"])
                except json.JSONDecodeError:
                    logger.warning(f"Could not parse final tool call arguments for call {index}: {call['function']['arguments']}")
                    # Keep the potentially partial string as is, or set to "{}" ?
                final_tool_calls.append(call)


            final_yield: Dict[str, Any] = {
                "type": "final", "done": True,
                "content": accumulated_content if accumulated_content else None,
                "tool_calls": final_tool_calls if final_tool_calls else None,
                "model": self.model, "provider": "google_gemini",
                "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": total_tokens},
                "tokens_per_second": tokens_per_second, "finish_reason": finish_reason or "UNKNOWN"
            }
            yield {k: v for k, v in final_yield.items() if v is not None}

        except (google_exceptions.GoogleAPIError, ValueError) as e:
            logger.error(f"Google Gemini API stream error: {e}", exc_info=True)
            yield {"type": "error", "error": str(e), "done": True}
        except Exception as e:
            logger.error(f"Unexpected error during Google Gemini stream: {e}", exc_info=True)
            yield {"type": "error", "error": "An unexpected error occurred during streaming.", "done": True}

    @retry_decorator
    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """ Get embeddings for a list of texts using the configured Gemini embedding model. """
        logger.debug(f"Requesting Gemini embeddings for {len(texts)} texts using model {self.embedding_model}")
        start_time = time.time()
        try:
            batch_size = 100
            all_embeddings = []
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                response = await asyncio.to_thread(
                    genai.embed_content,
                    model=self.embedding_model, content=batch_texts, task_type="retrieve_document"
                )
                all_embeddings.extend(response['embedding'])
            elapsed_time = time.time() - start_time
            logger.info(f"Generated Gemini embeddings for {len(texts)} texts in {elapsed_time:.2f} seconds.")
            return all_embeddings
        except (google_exceptions.GoogleAPIError, ValueError) as e:
            logger.error(f"Google Gemini embedding error: {e}", exc_info=True)
            dimension = 768
            return [[0.0] * dimension for _ in range(len(texts))]
        except Exception as e:
             logger.error(f"Unexpected error during Google Gemini embedding: {e}", exc_info=True)
             dimension = 768
             return [[0.0] * dimension for _ in range(len(texts))]