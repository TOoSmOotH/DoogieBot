# Doogie Chat Bot - Tool Calling Implementation

This document details the internal workflow for how Doogie Chat Bot enables and handles tool calls made by the Language Model (LLM) using connected Model Context Protocol (MCP) servers.

## Overview

The tool calling feature allows the LLM to request the execution of functions provided by external MCP servers during a conversation. This enables the LLM to access real-time information, interact with external systems, or perform actions beyond its built-in knowledge. The process involves multiple steps orchestrated primarily by the `LLMService` and related components.

## Workflow

The following steps outline the end-to-end process when a user sends a message:

1.  **Tool Discovery and Formatting (`LLMService`)**:
    *   When `LLMService.chat` is called, it retrieves all enabled `MCPConfig` entries associated with the current `user_id` from the database using `MCPConfigService`.
    *   For each enabled config, it attempts to find a matching predefined schema in `LLMService.CONNECTED_SERVER_SCHEMAS` based on the server's configured name (e.g., a config named `filesystem-data` matches the `filesystem` schema key).
    *   If a schema is found, the tools defined within that schema are formatted according to the requirements of the target LLM provider (currently targeting OpenAI's function calling format). A unique prefix derived from the MCP server's name (e.g., `filesystem_data__`) is added to each tool name within that server (e.g., `filesystem_data__read_file`) to avoid naming collisions and allow mapping back to the correct server later.
    *   If no predefined schema matches, a generic tool schema is created based on the server name.
    *   The collected list of formatted tool schemas is prepared.

2.  **Initial LLM Call (`LLMService` -> `LLMClient`)**:
    *   The user message, conversation history, system prompt (potentially including RAG context), and the formatted `tools` list are passed to the appropriate `LLMClient` implementation (`generate` method).
    *   The `tool_choice` parameter is typically set to `"auto"` for the initial call, allowing the LLM to decide whether to use a tool or respond directly.

3.  **LLM Response Parsing (`LLMClient` -> `LLMService`/`llm_stream.py`)**:
    *   The `LLMClient` implementation receives the response from the LLM provider's API.
    *   It parses the response, specifically looking for tool usage indicators (e.g., `tool_calls` in OpenAI/Gemini, `tool_use` content blocks in Anthropic).
    *   The client standardizes the parsed output, ensuring that if tools were called, the result contains a `tool_calls` key with a list of requested calls (including `id`, `function.name`, `function.arguments`).
    *   The `finish_reason` is checked (e.g., `tool_calls`, `stop`).

4.  **Handling Tool Call Request (`LLMService`/`llm_stream.py`)**:
    *   The calling service (`LLMService.chat` for non-streaming, `stream_llm_response` for streaming) receives the parsed response from the `LLMClient`.
    *   It checks if the `tool_calls` key is present and the `finish_reason` indicates tool usage.
    *   **Save Assistant Message:** The assistant's message (which might contain preliminary text *before* the tool call request) and the `tool_calls` data are saved to the database as a single `Message` record with `role='assistant'` using `ChatService.add_message`. The `finish_reason` is stored as `tool_calls`.
    *   The assistant message (including `tool_calls`) is added to the current message history list for context.

5.  **Tool Execution (`MCPConfigService.execute_mcp_tool`)**:
    *   For each tool call in the `tool_calls` list:
        *   The unique tool name (e.g., `filesystem_data__read_file`) is parsed to extract the server name prefix (`filesystem_data`).
        *   The prefix is used to look up the corresponding `MCPConfig.id` from the configurations fetched earlier.
        *   `MCPConfigService.execute_mcp_tool` is called with the `config_id`, `tool_call_id`, the original full tool name, and the arguments string.
        *   Inside `execute_mcp_tool`:
            *   The running Docker container associated with the `config_id` is located.
            *   *(Implementation Detail: Likely uses `container.attach_socket()` or a similar mechanism, not just `exec_run`, to interact with the container's stdio).*
            *   A JSON-RPC request containing the *original* tool name (without the prefix, e.g., `read_file`) and parsed arguments is constructed and sent to the container's stdin.
            *   The service waits for a JSON-RPC response (containing the result or an error) from the container's stdout.
            *   The result content (as a JSON string) is extracted. Errors during execution are caught and formatted as a JSON error object string.
        *   *(Concurrency: Tool executions for multiple calls within the same LLM response are typically run concurrently using `asyncio.gather` and `asyncio.to_thread`)*.

6.  **Handling Tool Result (`LLMService`/`llm_stream.py`)**:
    *   The result string (or error string) obtained from `execute_mcp_tool` is received.
    *   **Save Tool Message:** A new `Message` record with `role='tool'` is created and saved using `ChatService.add_message`. This message includes the `tool_call_id` (linking it back to the assistant's request), the `name` of the tool called, and the `content` (the JSON result string).
    *   The tool result message (formatted for the LLM with `role='tool'`, `tool_call_id`, and `content`) is added to the current message history list.

7.  **Subsequent LLM Call (`LLMService`/`llm_stream.py`)**:
    *   The `LLMClient.generate` method is called *again*, this time with the updated message history (which now includes the assistant's tool request and the corresponding tool result messages).
    *   **Crucially, the `tools` and `tool_choice` parameters are *not* sent on this subsequent call** to prevent the LLM from immediately trying to call another tool based only on the first tool's result. The LLM should generate a natural language response based on the tool's output.

8.  **Processing Final Response (`LLMService`/`llm_stream.py`)**:
    *   The response from the second (or subsequent, in non-streaming) LLM call is received. This response should contain the final natural language answer for the user.
    *   **Save Final Assistant Message:** This final content is saved as a new `Message` record with `role='assistant'` using `ChatService.add_message`. The `finish_reason` should now be `stop` (or similar).

## Multi-Turn Limits

*   **Non-Streaming (`LLMService.chat`)**: A `MAX_TOOL_TURNS` constant (e.g., 5) limits the number of cycles (LLM requests tool -> execute tool -> LLM processes result -> LLM requests tool...) within a single user turn to prevent infinite loops. If the limit is reached, an error message is returned.
*   **Streaming (`stream_llm_response`)**: The current implementation simplifies the flow by only performing **one round** of tool execution per user message. If the LLM were to request tools again immediately after processing the first tool result, those subsequent requests would not be executed in the streaming context. The final content stream is sent after the single tool execution round.

## Error Handling

Errors can occur at various stages:
*   LLM API errors (rate limits, context length, etc.).
*   Failure to find a configured MCP server.
*   Docker container errors (not running, failed to start).
*   Errors during tool execution within the MCP server.
*   Errors parsing LLM responses or tool results.
These errors are generally logged, and often an error message is saved to the chat history and/or returned to the user.