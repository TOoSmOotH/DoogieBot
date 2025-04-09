# Doogie Chat Bot - Model Context Protocol (MCP) Configuration Guide

This guide explains how administrators can configure and manage Model Context Protocol (MCP) servers within the Doogie Chat Bot application. MCP allows the chatbot to interact with external tools and data sources.

## Overview

Doogie Chat Bot supports connecting to MCP servers, enabling the underlying Language Model (LLM) to utilize external tools during chat interactions. All MCP servers are managed and run securely within Docker containers orchestrated by the Doogie backend.

## Accessing MCP Configuration

1.  Log in to the Doogie Chat Bot application as an administrator.
2.  Navigate to the **Admin Dashboard**.
3.  Select **MCP Servers** from the sidebar menu.

## Managing MCP Servers

The MCP Servers dashboard displays a list of all configured servers, their status (Running, Stopped, Error), and provides options to manage them.

### Adding a New MCP Server

1.  Click the "**Add New MCP Server**" button on the dashboard.
2.  Fill in the following details:
    *   **Name:** A unique, descriptive name for the server (e.g., `filesystem-projectA`, `github-personal`). This name is used internally and helps the LLM identify the tool source.
    *   **Command:** The command used to start the server.
        *   **Important:** Even if the original command uses `npx` or `uvx`, the backend will translate this to run within a Docker container. You typically specify the Docker image and any necessary arguments here.
        *   Example (Filesystem Server): `docker run -i --rm mcp/filesystem /path/to/allowed/dir`
        *   Example (GitHub Server): `docker run -i --rm mcp/github`
    *   **Environment Variables (Optional):** Provide any necessary environment variables (e.g., API keys) as key-value pairs. For sensitive values like API keys, ensure proper security practices are followed. Example: `GITHUB_PERSONAL_ACCESS_TOKEN=your_token_here`.
    *   **Enabled:** Check this box to make the server available for the LLM to use.

3.  Click "**Save Configuration**".

### Editing an MCP Server

1.  From the MCP Servers dashboard, click the "**Edit**" button next to the server you want to modify.
2.  Update the configuration details as needed.
3.  Click "**Save Configuration**". Changes might require restarting the server container.

### Enabling/Disabling a Server

You can quickly enable or disable a server directly from the dashboard using the toggle switch. Disabling a server prevents the LLM from seeing or using its tools.

### Starting/Stopping a Server

*   Use the "**Start**" or "**Stop**" buttons on the dashboard to control the underlying Docker container for the MCP server.
*   The status indicator will update to reflect the container's state.

### Deleting a Server

1.  Click the "**Delete**" button next to the server you want to remove.
2.  Confirm the deletion. This will stop the container (if running) and remove the configuration from the database.

## Tool Usage by the LLM

*   When an MCP server is **enabled** and **running**, the Doogie backend makes its tools available to the LLM during chat generation.
*   The backend automatically formats the tool schemas provided by the *connected* MCP servers for the LLM.
*   If the LLM decides to use a tool, the backend handles:
    1.  Parsing the tool call request from the LLM.
    2.  Identifying the correct MCP server based on the tool name prefix (derived from the server's configured **Name**).
    3.  Executing the tool command within the corresponding Docker container via `docker exec`.
    4.  Sending the tool arguments to the MCP server's standard input.
    5.  Reading the tool's result (JSON) from the server's standard output.
    6.  Formatting the result and sending it back to the LLM.
    7.  Receiving and displaying the LLM's final response, which incorporates the tool's output.

## Important Notes

*   **Security:** Exposing the Docker socket requires caution. Ensure the Doogie application itself is secured and access is restricted. In development, the container runs as root for simplicity, but production deployments should implement stricter permissions.
*   **Docker Images:** Ensure the Docker images specified in the server commands are accessible to the Docker daemon running within the Doogie container (e.g., pulled beforehand or available in a registry).
*   **Error Handling:** If an MCP server container fails to start or encounters an error during execution, the status will be updated on the dashboard, and errors might be logged in the backend. Tool execution failures will be reported back to the LLM.
*   **Resource Management:** Be mindful of the system resources consumed by running multiple MCP server containers.