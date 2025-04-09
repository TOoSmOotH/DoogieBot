# backend/app/services/mcp_config_service/__init__.py
"""
MCP Configuration Service Package.

This package handles the management of MCP server configurations, including:
1. CRUD operations for MCP server configurations
2. Docker container management for running MCP servers
3. Generating MCP configuration JSON for clients
4. Executing tool calls via MCP servers
5. Describing tools provided by MCP servers
"""

# Import functions/classes from submodules to expose them at the package level
from .crud import (
    create_config,
    get_config_by_id,
    get_configs_by_user,
    update_config,
    delete_config,
    get_all_enabled_configs,
)
from .docker_utils import (
    _get_docker_client,
    _get_container_name,
    _transform_command_to_docker,
)
from .status import get_config_status
from .lifecycle import start_server, stop_server, restart_server
from .execution import execute_mcp_tool
from .config_gen import generate_mcp_config_json
from .manager import mcp_session_manager, get_mcp_session_manager, MCPSessionManager

# Define a facade class or simply expose functions directly
# For simplicity, let's expose functions directly for now.
# If a class structure is preferred later, we can refactor __init__.py

__all__ = [
    "create_config",
    "get_config_by_id",
    "get_configs_by_user",
    "update_config",
    "delete_config",
    "get_all_enabled_configs",
    "get_config_status",
    "start_server",
    "stop_server",
    "restart_server",
    "execute_mcp_tool",
    "generate_mcp_config_json",
]

# Define the MCPConfigService class that's imported elsewhere
class MCPConfigService:
    @staticmethod
    def create_config(*args, **kwargs): 
        return create_config(*args, **kwargs)
    
    @staticmethod
    def get_config_by_id(*args, **kwargs):
        return get_config_by_id(*args, **kwargs)
    
    @staticmethod
    def get_configs_by_user(*args, **kwargs):
        return get_configs_by_user(*args, **kwargs)
    
    @staticmethod
    def update_config(*args, **kwargs):
        return update_config(*args, **kwargs)
    
    @staticmethod
    def delete_config(*args, **kwargs):
        return delete_config(*args, **kwargs)
    
    @staticmethod
    def get_all_enabled_configs(*args, **kwargs):
        return get_all_enabled_configs(*args, **kwargs)
    
    @staticmethod
    def get_config_status(*args, **kwargs):
        return get_config_status(*args, **kwargs)
    
    @staticmethod
    def start_server(*args, **kwargs):
        return start_server(*args, **kwargs)
    
    @staticmethod
    def stop_server(*args, **kwargs):
        return stop_server(*args, **kwargs)
    
    @staticmethod
    def restart_server(*args, **kwargs):
        return restart_server(*args, **kwargs)
    
    @staticmethod
    def execute_mcp_tool(*args, **kwargs):
        return execute_mcp_tool(*args, **kwargs)
    
    @staticmethod
    def generate_mcp_config_json(*args, **kwargs):
        return generate_mcp_config_json(*args, **kwargs)

# Update the __all__ list to include the MCPConfigService class and manager components
__all__ = [
    "create_config",
    "get_config_by_id",
    "get_configs_by_user",
    "update_config",
    "delete_config",
    "get_all_enabled_configs",
    "get_config_status",
    "start_server",
    "stop_server",
    "restart_server",
    "execute_mcp_tool",
    "generate_mcp_config_json",
    "MCPConfigService",  # Add the class to __all__
    "mcp_session_manager",
    "get_mcp_session_manager",
    "MCPSessionManager",
]
