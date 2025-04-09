import time
import logging
from typing import List, Dict, Any, Optional, Union, AsyncGenerator
from anthropic import AsyncAnthropic, APIError, APIStatusError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import json
import uuid

from app.llm.base import LLMClient
from app.core.config import settings

# Configure logging
logger = logging.getLogger(__name__)
if settings.LLM_DEBUG_LOGGING:
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)

# Define retry mechanism for Anthropic API calls
retry_decorator = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((APIError, APIStatusError)),
    reraise=True
)

class AnthropicClient(LLMClient):
    """
    Client for interacting with the Anthropic API (Claude models).
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None, # Anthropic doesn't typically use base_url
        embedding_model: Optional[str] = None,
        user_id: Optional[str] = None # Add user_id parameter
    ):
        # Pass user_id to base class constructor
        super().__init__(model=model, api_key=api_key, base_url=base_url, embedding_model=embedding_model, user_id=user_id)
        if not self.api_key:
            raise ValueError("Anthropic API key is required.")

        # Base URL is not typically used for Anthropic's main API, but allow if provided
        client_args = {"api_key": self.api_key}
        if self.base_url:
            client_args["base_url"] = self.base_url

        self.async_client = AsyncAnthropic(**client_args)
        logger.info(f"Anthropic client initialized with model: {self.model}")
        if self.base_url:
             logger.info(f"Using custom Anthropic base URL: {self.base_url}")
             
    def _params_to_xml(self, params_schema: Dict[str, Any]) -> str:
        """
        Convert JSON Schema parameters to Anthropic XML format.
        
        Args:
            params_schema: JSON Schema object with properties, required, etc.
            
        Returns:
            XML string representation of parameters
        """
        xml_parts = []
        
        # Get required params if present
        required_params = params_schema.get('required', [])
        
        # Process properties
        for param_name, param_info in params_schema.get('properties', {}).items():
            param_type = param_info.get('type', 'string')
            param_desc = param_info.get('description', '')
            param_required = param_name in required_params
            
            # Start parameter tag
            param_xml = f'<parameter name="{param_name}" type="{param_type}"'
            if param_required:
                param_xml += ' required="true"'
            param_xml += '>\n'
            
            # Add description if present
            if param_desc:
                param_xml += f'<description>{param_desc}</description>\n'
            
            # Handle enum values
            if 'enum' in param_info:
                param_xml += '<enum>\n'
                for value in param_info['enum']:
                    param_xml += f'<option>{value}</option>\n'
                param_xml += '</enum>\n'
            
            # Close parameter tag
            param_xml += '</parameter>\n'
            xml_parts.append(param_xml)
        
        return ''.join(xml_parts)

    @retry_decorator
    async def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = 1024, # Default for Claude
        stream: bool = False,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None, # <-- Add tools
        tool_choice: Optional[str] = None # <-- Add tool_choice
    ) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        """
        Generate a response from the Anthropic model.

        Args:
            messages: List of messages (user/assistant roles).
            temperature: Temperature for generation.
            max_tokens: Maximum number of tokens to generate.
            stream: Whether to stream the response.
            system_prompt: The system prompt to use (required by Anthropic).
            tools: Optional list of tool definitions.
            tool_choice: Optional control over tool calling (currently ignored for Anthropic).

        Returns:
            Response dictionary or an async generator for streaming.
        """
        start_time = time.time()

        if not system_prompt:
             # Use default if not provided, but log a warning as it's important for Claude
            system_prompt = settings.DEFAULT_SYSTEM_PROMPT
            logger.warning("System prompt not explicitly provided for Anthropic, using default.")

        # Filter out system messages from the main list if present, use the dedicated param
        conversation_messages = [msg for msg in messages if msg.get("role") != "system"]

        request_params: Dict[str, Any] = { # Define type
            "model": self.model,
            "messages": conversation_messages,
            "system": system_prompt,
            "temperature": temperature,
            "max_tokens": max_tokens or 1024, # Ensure a default if None
        }

        # Handle tools for Anthropic (convert to XML tool format)
        if tools:
            logger.info(f"Implementing Anthropic tool use for {len(tools)} tools")
            if system_prompt:
                # Format tools in XML <tools> format within the system prompt
                tools_xml = "<tools>\n"
                for tool in tools:
                    if tool.get("type") == "function":
                        func = tool.get("function", {})
                        func_name = func.get("name")
                        func_desc = func.get("description", "")
                        params_schema = func.get("parameters", {})
                        
                        # Start tool definition
                        tools_xml += f"<tool_description>\n"
                        tools_xml += f"<tool_name>{func_name}</tool_name>\n"
                        if func_desc:
                            tools_xml += f"<description>{func_desc}</description>\n"
                        
                        # Add parameters
                        if params_schema:
                            tools_xml += "<parameters>\n"
                            params_xml = self._params_to_xml(params_schema)
                            tools_xml += params_xml
                            tools_xml += "</parameters>\n"
                        
                        tools_xml += "</tool_description>\n"
                
                tools_xml += "</tools>\n\n"
                
                # Add tool use instructions
                tools_xml += """To use a tool, respond in this format: <tool_use>
<tool_name>$TOOL_NAME</tool_name>
<parameters>
<$PARAM_NAME>$PARAM_VALUE</$PARAM_NAME>
</parameters>
</tool_use>

Only use the listed tools. Only include parameters defined for each tool."""
                
                # Append the tools XML to the system prompt
                request_params["system"] = system_prompt + "\n\n" + tools_xml
                
                if settings.LLM_DEBUG_LOGGING:
                    logger.debug(f"Anthropic tools formatted as XML in system prompt")
            else:
                logger.warning("Cannot add tools to Anthropic request: system prompt is required")
            
            # Note that tool_choice is still ignored for Anthropic
            if tool_choice and tool_choice != "auto":
                logger.warning(f"Anthropic does not support tool_choice={tool_choice}. Using default behavior.")
        

        logger.debug(f"Anthropic request params: {request_params}")

        try:
            if stream:
                # Pass request_params which now includes tools/tool_choice if they were added
                return self._generate_stream(request_params, start_time)
            else:
                response = await self.async_client.messages.create(**request_params)

                logger.debug(f"Anthropic non-stream response: {response}")

                # Handle tool_use block in non-streaming response
                if response.stop_reason == "tool_use":
                    logger.info("Detected tool_use in Anthropic non-streaming response")
                    content = None
                    tool_calls = []
                    
                    # Extract tool use block from content
                    tool_text = response.content[0].text if response.content and hasattr(response.content[0], 'text') else None
                    
                    if tool_text and "<tool_use>" in tool_text:
                        # Extract tool name and parameters from XML
                        import re
                        
                        # Extract tool name
                        name_match = re.search(r'<tool_name>(.*?)</tool_name>', tool_text, re.DOTALL)
                        tool_name = name_match.group(1).strip() if name_match else None
                        
                        # Extract parameters block
                        params_match = re.search(r'<parameters>(.*?)</parameters>', tool_text, re.DOTALL)
                        params_block = params_match.group(1).strip() if params_match else ""
                        
                        # Parse parameters into JSON
                        params = {}
                        param_matches = re.findall(r'<([^>]+)>(.*?)</\\1>', params_block, re.DOTALL)
                        for param_name, param_value in param_matches:
                            # Convert values to appropriate types if needed
                            try:
                                # Try to parse as number or boolean
                                import json
                                parsed_value = json.loads(param_value.strip())
                                params[param_name] = parsed_value
                            except json.JSONDecodeError:
                                # Keep as string if not valid JSON
                                params[param_name] = param_value.strip()
                        
                        if tool_name:
                            # Generate a unique ID for the tool call
                            import uuid
                            tool_id = f"call_{uuid.uuid4()}"
                            
                            # Format as OpenAI-compatible tool call
                            tool_calls.append({
                                "id": tool_id,
                                "type": "function",
                                "function": {
                                    "name": tool_name,
                                    "arguments": json.dumps(params)
                                }
                            })
                            
                            logger.info(f"Extracted tool call for {tool_name} with ID {tool_id}")
                    
                    return {
                        "role": "assistant",
                        "content": content,  # Will be None if tool use
                        "tool_calls": tool_calls,
                        "usage": {
                            "prompt_tokens": response.usage.input_tokens,
                            "completion_tokens": response.usage.output_tokens,
                            "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
                        },
                        "tokens_per_second": self.calculate_tokens_per_second(start_time, response.usage.output_tokens),
                        "finish_reason": "tool_calls"  # Convert Anthropic's stop_reason to OpenAI-compatible
                    }

                total_tokens = response.usage.input_tokens + response.usage.output_tokens
                tokens_per_second = self.calculate_tokens_per_second(start_time, response.usage.output_tokens)

                # Assuming text response for now
                content = response.content[0].text if response.content and hasattr(response.content[0], 'text') else None

                return {
                    "role": "assistant",
                    "content": content,
                    "usage": {
                        "prompt_tokens": response.usage.input_tokens,
                        "completion_tokens": response.usage.output_tokens,
                        "total_tokens": total_tokens,
                    },
                    "tokens_per_second": tokens_per_second,
                    "finish_reason": response.stop_reason
                }
        except (APIError, APIStatusError) as e:
            logger.error(f"Anthropic API error: {e}", exc_info=True)
            raise  # Re-raise after logging and retries

    async def _generate_stream(
        self,
        request_params: Dict[str, Any], # Includes tools if added
        start_time: float
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Helper for generating streaming response."""
        # Initialize state variables
        completion_tokens = 0
        prompt_tokens = 0 # Anthropic sends usage stats at the end or in message_start
        finish_reason = None
        accumulated_content = "" # Accumulate content across deltas
        
        # Tool handling state
        current_tool_name = None
        current_tool_id = None
        current_tool_params = {}
        final_tool_calls = []

        try:
            async with self.async_client.messages.stream(**request_params) as stream_obj:
                async for event in stream_obj:
                    logger.debug(f"Anthropic stream event: {event.type}")
                    yield_chunk: Dict[str, Any] = {"type": "delta", "done": False} # Default chunk

                    if event.type == "message_start":
                        prompt_tokens = event.message.usage.input_tokens
                        yield { # Yield start event separately
                            "type": "start",
                            "role": "assistant",
                            "model": event.message.model,
                            "usage": {"prompt_tokens": prompt_tokens}
                        }
                        continue # Don't yield delta for start event

                    elif event.type == "content_block_start":
                         # Handle tool_use block start
                         if event.content_block.type == "tool_use":
                              logger.info(f"Anthropic stream started tool_use block: {event.content_block.name}")
                              
                              # Generate a unique ID for this tool call
                              current_tool_id = f"call_{uuid.uuid4()}"
                              current_tool_name = event.content_block.name
                              current_tool_params = {}
                              
                              # Yield a delta for the tool call start
                              yield {
                                  "tool_calls_delta": [
                                      {
                                          "index": 0,
                                          "id": current_tool_id,
                                          "type": "function",
                                          "function": {
                                              "name": current_tool_name
                                          }
                                      }
                                  ]
                              }
                         continue # Don't yield delta

                    elif event.type == "content_block_delta":
                        if event.delta.type == "text_delta":
                            delta_content = event.delta.text
                            accumulated_content += delta_content
                            yield_chunk["content"] = delta_content # Yield delta
                        elif event.delta.type == "input_json_delta":
                             # Accumulate tool input JSON
                             logger.info(f"Anthropic stream tool input delta: {event.delta.partial_json}")
                             
                             if current_tool_id and current_tool_name:
                                 # Extract parameter name and value from partial JSON
                                 partial_json = event.delta.partial_json
                                 
                                 if isinstance(partial_json, dict):
                                     # Update the parameters dictionary
                                     current_tool_params.update(partial_json)
                                     
                                     # Convert to JSON string for arguments delta
                                     json_str = json.dumps(partial_json).strip('{}').strip()
                                     if json_str:
                                         # Yield a delta for the argument update
                                         yield {
                                             "tool_calls_delta": [
                                                 {
                                                     "index": 0,
                                                     "function": {
                                                         "arguments": json_str
                                                     }
                                                 }
                                             ]
                                         }
                        else:
                             # Handle other potential delta types if necessary
                             logger.warning(f"Unhandled content_block_delta type: {event.delta.type}")

                    elif event.type == "content_block_stop":
                         # Handle tool_use block stop
                         logger.info(f"Anthropic stream stopped content block index: {event.index}")
                         
                         # If we have an active tool call, finalize it
                         if current_tool_id and current_tool_name:
                             try:
                                 # Convert accumulated parameters to JSON string
                                 arguments_json = json.dumps(current_tool_params)
                                 
                                 # Create the complete tool call
                                 tool_call = {
                                     "id": current_tool_id,
                                     "type": "function",
                                     "function": {
                                         "name": current_tool_name,
                                         "arguments": arguments_json
                                     }
                                 }
                                 
                                 # Add to final tool calls list
                                 final_tool_calls.append(tool_call)
                                 
                                 # Set finish reason to tool_calls for the final response
                                 finish_reason = "tool_calls"
                                 
                                 # Yield a complete tool call
                                 yield {
                                     "tool_calls": [tool_call],
                                     "type": "tool_calls_done"
                                 }
                                 
                                 # Reset current tool state
                                 current_tool_id = None
                                 current_tool_name = None
                                 current_tool_params = {}
                                 
                             except Exception as e:
                                 logger.error(f"Error finalizing tool call: {e}")
                         continue # Don't yield delta

                    elif event.type == "message_delta":
                         # Contains final usage stats sometimes and stop reason
                        if hasattr(event, 'usage') and hasattr(event.usage, 'output_tokens'):
                            completion_tokens = event.usage.output_tokens # Update with actual count
                        finish_reason = event.delta.stop_reason
                        # Don't yield this event directly as delta, wait for message_stop

                    elif event.type == "message_stop":
                        # Final event, get the complete message to extract final usage
                        final_message = await stream_obj.get_final_message()
                        if final_message and final_message.usage:
                             prompt_tokens = final_message.usage.input_tokens
                             completion_tokens = final_message.usage.output_tokens
                        # finish_reason should have been set in message_delta
                        break # Exit loop, final chunk yielded below

                    else:
                         logger.warning(f"Unhandled Anthropic stream event type: {event.type}")
                         continue # Skip unknown event types for now

                    # Yield the delta chunk if it has content or tool delta
                    if "content" in yield_chunk or "tool_input_delta" in yield_chunk:
                         yield yield_chunk


                # Yield final message after stream ends
                total_tokens = prompt_tokens + completion_tokens
                tokens_per_second = self.calculate_tokens_per_second(start_time, completion_tokens)

                final_yield: Dict[str, Any] = {
                    "type": "final",
                    "done": True,
                    "content": accumulated_content if accumulated_content else None,
                    "tool_calls": final_tool_calls if final_tool_calls else None,
                    "model": request_params["model"], # Use model from request
                    "provider": "anthropic",
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens,
                    },
                    "tokens_per_second": tokens_per_second,
                    "finish_reason": finish_reason
                }
                yield {k: v for k, v in final_yield.items() if v is not None}


        except (APIError, APIStatusError) as e:
            logger.error(f"Anthropic API stream error: {e}", exc_info=True)
            yield {
                "type": "error",
                "error": str(e),
                "done": True
            }
        except Exception as e:
            logger.error(f"Unexpected error during Anthropic stream: {e}", exc_info=True)
            yield {
                "type": "error",
                "error": "An unexpected error occurred during streaming.",
                "done": True
            }


    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for a list of texts.
        NOTE: Anthropic's primary Python client library does not support embeddings directly.
              This method raises NotImplementedError. Configure a different embedding_provider.
        """
        logger.warning("Anthropic client does not support embeddings directly. Configure a different embedding_provider.")
        raise NotImplementedError("Anthropic client does not support embeddings directly.")