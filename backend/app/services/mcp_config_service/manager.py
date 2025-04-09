# backend/app/services/mcp_config_service/manager.py
"""
Manages active MCP server connections and sessions.
"""
import asyncio
import logging
import os
import shutil
from contextlib import AsyncExitStack
from typing import Any, Dict, Optional, Tuple

# Conditional MCP SDK imports
try:
    from mcp import ClientSession, StdioServerParameters, types
    from mcp.client.stdio import stdio_client
    MCP_SDK_AVAILABLE = True
except ImportError:
    MCP_SDK_AVAILABLE = False
    # Define dummy types if SDK is not available to prevent runtime errors on import
    class StdioServerParameters: pass
    class ClientSession: pass
    class types: pass
    def stdio_client(*args, **kwargs): pass


logger = logging.getLogger(__name__)

# Type Alias for managed connection state
ManagedConnection = Tuple[ClientSession, AsyncExitStack]

class MCPSessionManager:
    """
    Manages lifecycle and access to MCP ClientSessions for configured servers.
    Ensures only one connection/process per server config is active.
    """
    def __init__(self):
        if not MCP_SDK_AVAILABLE:
            logger.error("MCP SDK (modelcontextprotocol) not installed. MCP functionality will be disabled.")
            # Raise an error or handle appropriately based on application requirements
            # raise ImportError("MCP SDK not found. Please install 'modelcontextprotocol'.")
        self._sessions: Dict[str, ManagedConnection] = {} # config_id -> (ClientSession, AsyncExitStack)
        self._locks: Dict[str, asyncio.Lock] = {} # config_id -> Lock

    async def _get_lock(self, config_id: str) -> asyncio.Lock:
        """Gets or creates a lock for a specific config_id."""
        if config_id not in self._locks:
            self._locks[config_id] = asyncio.Lock()
        return self._locks[config_id]

    def _create_server_parameters(self, config: Dict[str, Any]) -> Optional[StdioServerParameters]:
        """Creates StdioServerParameters from a config dictionary."""
        if not MCP_SDK_AVAILABLE: return None

        command_name = config.get("command")
        if not command_name:
            logger.error("Missing 'command' in MCP config.")
            return None

        command_path = shutil.which(command_name)
        if not command_path:
            # Allow common commands directly if not explicitly in PATH
            if command_name in ["npx", "docker", "uvx"]: # Added uvx
                 command_path = command_name
            else:
                logger.error(f"Command '{command_name}' not found in PATH for MCP server.")
                return None

        args = config.get("args", [])
        # Ensure docker run uses -i if applicable (important for stdio)
        if command_path == "docker" or command_path.endswith("/docker"):
            # Add --host flag to connect to dind-server
            if "--host" not in args and "-H" not in args:
                args.insert(0, "--host=tcp://dind-server:2375")
                logger.debug("Added '--host=tcp://dind-server:2375' flag to docker command")
            
            # Ensure -i flag is present for docker run
            if "run" in args and "-i" not in args:
                try:
                    run_idx = args.index("run")
                    # Insert '-i' right after 'run'
                    args.insert(run_idx + 1, "-i")
                    logger.debug("Ensured '-i' flag is present for 'docker run'")
                except ValueError:
                    pass # 'run' not found

        # Use os.environ as base, override with config env, ensuring config env is a dict
        config_env = config.get("env") or {} # Ensure we have a dict, even if env is None/null
        env = {**os.environ, **config_env}
        # Optionally add debug flags if needed globally
        # env['DEBUG'] = '1'

        try:
            params = StdioServerParameters(command=command_path, args=args, env=env)
            logger.debug(f"Created StdioServerParameters: {command_path} {' '.join(args)}")
            return params
        except Exception as e:
            logger.error(f"Failed to create StdioServerParameters: {e}")
            return None

    async def get_session(self, config_id: str, config: Dict[str, Any]) -> Optional[ClientSession]:
        """
        Gets an active ClientSession for the given config_id.
        Establishes a new connection if one doesn't exist or is unhealthy.
        """
        if not MCP_SDK_AVAILABLE:
             logger.warning("MCP SDK not available, cannot get session.")
             return None

        lock = await self._get_lock(config_id)
        async with lock:
            # Check if session exists and is healthy
            if config_id in self._sessions:
                session, _ = self._sessions[config_id]
                if await self._is_session_healthy(session, config_id):
                    logger.info(f"Reusing existing healthy session for config_id: {config_id}")
                    return session
                else:
                    logger.warning(f"Session for config_id {config_id} found unhealthy. Reconnecting.")
                    await self._close_session(config_id) # Clean up old one

            # Create new session
            logger.info(f"Establishing new MCP session for config_id: {config_id}")
            server_params = self._create_server_parameters(config)
            if not server_params:
                logger.error(f"Failed to create server parameters for config_id: {config_id}")
                return None

            try:
                exit_stack = AsyncExitStack()
                stdio_transport = await exit_stack.enter_async_context(stdio_client(server_params))
                read, write = stdio_transport
                session = await exit_stack.enter_async_context(ClientSession(read, write))
                await session.initialize() # Perform MCP handshake
                logger.info(f"Successfully initialized new session for config_id: {config_id}")
                self._sessions[config_id] = (session, exit_stack) # Store session and stack
                return session
            except Exception as e:
                logger.exception(f"Failed to establish MCP session for config_id {config_id}: {e}")
                # Ensure partial cleanup if error occurred during setup
                if 'exit_stack' in locals() and exit_stack:
                     await exit_stack.aclose() # Attempt to clean up resources managed by the stack so far
                return None

    async def _is_session_healthy(self, session: ClientSession, config_id: str) -> bool:
        """Checks if a session is responsive using a ping."""
        if not MCP_SDK_AVAILABLE: return False
        try:
            # Use a timeout for the ping to prevent hanging indefinitely
            async with asyncio.timeout(5): # 5-second timeout for ping
                 await session.send_ping()
            logger.debug(f"Session health check PASSED for config_id: {config_id}")
            return True
        except asyncio.TimeoutError:
             logger.warning(f"Session health check TIMEOUT for config_id: {config_id}")
             return False
        except Exception as e:
            # Log specific error type and message
            logger.warning(f"Session health check FAILED for config_id {config_id}: {type(e).__name__} - {e}")
            return False

    async def _close_session(self, config_id: str) -> None:
        """Closes the session and cleans up resources for a specific config_id."""
        if config_id in self._sessions:
            session, exit_stack = self._sessions.pop(config_id)
            logger.info(f"Closing MCP session and resources for config_id: {config_id}")
            try:
                await exit_stack.aclose() # This closes ClientSession and stdio_client contexts
                logger.info(f"Successfully closed resources for config_id: {config_id}")
            except Exception as e:
                logger.exception(f"Error closing AsyncExitStack for config_id {config_id}: {e}")
        else:
             logger.debug(f"No active session found to close for config_id: {config_id}")


    async def close_all_sessions(self) -> None:
        """Closes all managed sessions, typically called on application shutdown."""
        logger.info("Closing all managed MCP sessions...")
        # Create a list of keys to avoid modifying dict while iterating
        config_ids = list(self._sessions.keys())
        for config_id in config_ids:
             # Use a lock for each session closure for safety, though less critical on shutdown
             lock = await self._get_lock(config_id)
             async with lock:
                  await self._close_session(config_id)
        logger.info("Finished closing all managed MCP sessions.")

# Global instance (consider dependency injection for larger apps)
mcp_session_manager = MCPSessionManager()

async def get_mcp_session_manager() -> MCPSessionManager:
     """Dependency function for FastAPI."""
     return mcp_session_manager