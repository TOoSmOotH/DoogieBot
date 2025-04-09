# Doogie Chat Bot - Docker API Interaction

This document describes how the Doogie Chat Bot backend interacts with the Docker API to manage the lifecycle of Model Context Protocol (MCP) server containers.

## Overview

To enable MCP servers, Doogie Chat Bot needs to start, stop, monitor, and execute commands within Docker containers based on administrator configurations. This interaction is handled primarily by the `MCPConfigService` using the official `docker` Python library.

## Setup (Docker-in-Docker)

Running Docker commands from within the main Doogie Chat Bot container requires a "Docker-in-Docker" setup:

1.  **Dockerfile:** The main `Dockerfile` installs the Docker client CLI and the `docker` Python library.
2.  **Docker Socket Binding:** The `docker-compose.yml` file mounts the host's Docker socket (`/var/run/docker.sock`) into the Doogie container. This allows the Docker client inside the container to communicate with the Docker daemon running on the host machine.
    ```yaml
    # Example snippet from docker-compose.yml
    services:
      doogie-chat:
        # ... other config ...
        volumes:
          - /var/run/docker.sock:/var/run/docker.sock
          # Bind mount for project code
          - ./backend:/app/backend
          - ./frontend:/app/frontend
          # ... other potential mounts ...
        # user: root # Running as root in dev for socket permissions; USE CAUTION
    ```
3.  **Permissions:** Accessing the Docker socket typically requires specific permissions. In the development `docker-compose.yml`, the container is often run as `root` for simplicity. **In production, this is a significant security risk.** A dedicated Docker group or more granular permissions should be configured on the host and container user.

## Core Service (`MCPConfigService`)

The `backend/app/services/mcp_config_service.py` contains the core logic for Docker interactions related to MCP servers. It initializes a Docker client connected to the host daemon via the mounted socket.

```python
# Example initialization in MCPConfigService
import docker
# ...
try:
    self.docker_client = docker.from_env()
    # Verify connection
    self.docker_client.ping()
    logger.info("Successfully connected to Docker daemon.")
except Exception as e:
    logger.error(f"Failed to connect to Docker daemon: {e}")
    self.docker_client = None
```

## Key Docker Operations

The service implements the following operations using the `docker` library:

1.  **Start Server (`_start_container`)**:
    *   Takes an `MCPConfig` object.
    *   Parses the `command` string (see Command Translation).
    *   Uses `docker_client.containers.run()` to start a new container.
    *   Key parameters used:
        *   `image`: The Docker image name.
        *   `command`: Arguments to pass to the container entrypoint/command.
        *   `environment`: Dictionary of environment variables.
        *   `detach=True`: Runs the container in the background.
        *   `auto_remove=True`: Automatically removes the container when stopped (suitable for `-i --rm` style MCP servers).
        *   `stdin_open=True`, `tty=False`: Configured for interactive stdio communication used by MCP.
        *   `name`: A unique name for the container (e.g., `mcp-server-<config_id>`).
    *   Stores the container ID in the `MCPConfig` database record.

2.  **Stop Server (`_stop_container`)**:
    *   Takes an `MCPConfig` object.
    *   Retrieves the container using `docker_client.containers.get(container_id)`.
    *   Stops the container using `container.stop()`.
    *   Handles `docker.errors.NotFound` if the container is already gone.
    *   Removes the container ID from the `MCPConfig` record.

3.  **Get Server Status (`get_container_status`)**:
    *   Takes an `MCPConfig` object.
    *   If a `container_id` is stored, attempts to get the container via `docker_client.containers.get()`.
    *   Returns the `container.status` (e.g., 'running', 'exited', 'created').
    *   Handles `docker.errors.NotFound` and returns 'stopped' or 'error'.

4.  **Execute Tool (`execute_mcp_tool`)**:
    *   Identifies the running container for the target `MCPConfig`.
    *   Uses `container.exec_run()` to execute a command *inside* the running container.
    *   **Crucially, for MCP stdio communication:** Instead of `exec_run`, it likely needs to attach to the container's stdio streams. This is more complex and might involve:
        *   Using `container.attach_socket()` to get a socket connection.
        *   Sending the JSON-RPC tool request over the socket's stdin stream.
        *   Reading the JSON-RPC response from the socket's stdout stream.
        *   Handling potential errors and timeouts during communication.
    *   *(Self-correction: The initial implementation might have used `exec_run` incorrectly; attaching via sockets is the standard way to interact with `-i` containers like MCP servers).*

## Command Translation (`npx`/`uvx` to `docker run`)

The service includes logic (likely in helper functions or within `_start_container`) to translate commands like `npx @mcp/server ...` or `uvx mcp-server ...` into equivalent `docker run ...` commands. This typically involves:

*   Identifying the package name (e.g., `@mcp/server`).
*   Mapping it to a known Docker image (e.g., `mcp/server`).
*   Extracting arguments and environment variables.
*   Constructing the arguments for `docker_client.containers.run()`.

## Error Handling

The service wraps Docker API calls in `try...except` blocks to catch potential errors:

*   `docker.errors.APIError`: General errors from the Docker daemon.
*   `docker.errors.NotFound`: Container or image not found.
*   `docker.errors.ContainerError`: Error originating from within the container.
*   Connection errors during client initialization.

These errors are logged and often result in the MCP server status being set to 'error'.

## Security Considerations

**Binding the Docker socket into the container is inherently risky.** Any process within the container that can access the socket effectively has root-level control over the host machine's Docker daemon.

*   **Development:** Running the container as `root` simplifies permissions but is insecure.
*   **Production:**
    *   Run the Doogie container as a non-root user.
    *   Add this user to a specific group on the host that has permissions to access the Docker socket (e.g., the `docker` group).
    *   Consider alternative, more secure methods if possible, although they add complexity (e.g., a dedicated proxy service for Docker commands).
    *   Strictly limit administrator access to the Doogie application.