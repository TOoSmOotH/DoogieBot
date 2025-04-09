"""
Sample MCP configurations for testing.

This module provides sample MCP server configurations that can be used for testing
MCP-related functionality without requiring actual MCP servers.
"""

from typing import Dict, List, Any

# Sample MCP server configuration for filesystem server
sample_filesystem_config = {
    "id": "mcp-fs-12345",
    "name": "filesystem",
    "command": "docker",
    "args": ["run", "-i", "--rm", "mcp/filesystem", "/path/to/allowed/files"],
    "env": None,
    "enabled": True,
    "user_id": "user1"
}

# Sample MCP server configuration for git server
sample_git_config = {
    "id": "mcp-git-12345",
    "name": "git",
    "command": "docker",
    "args": ["run", "-i", "--rm", "mcp/git", "/path/to/git/repo"],
    "env": None,
    "enabled": True,
    "user_id": "user1"
}

# Sample MCP server configuration for github server
sample_github_config = {
    "id": "mcp-github-12345",
    "name": "github",
    "command": "docker",
    "args": ["run", "-i", "--rm", "mcp/github"],
    "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "github_pat_12345"
    },
    "enabled": True,
    "user_id": "user1"
}

# Sample MCP server configuration for postgres server
sample_postgres_config = {
    "id": "mcp-postgres-12345",
    "name": "postgres",
    "command": "docker",
    "args": ["run", "-i", "--rm", "mcp/postgres", "postgresql://localhost/mydb"],
    "env": None,
    "enabled": True,
    "user_id": "user1"
}

# Sample MCP server configuration with npx command
sample_npx_config = {
    "id": "mcp-npx-12345",
    "name": "npx-server",
    "command": "npx",
    "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/allowed/files"],
    "env": None,
    "enabled": True,
    "user_id": "user1"
}

# Sample MCP server configuration with uvx command
sample_uvx_config = {
    "id": "mcp-uvx-12345",
    "name": "uvx-server",
    "command": "uvx",
    "args": ["mcp-server-git", "--repository", "path/to/git/repo"],
    "env": None,
    "enabled": True,
    "user_id": "user1"
}

# Sample MCP server configuration in disabled state
sample_disabled_config = {
    "id": "mcp-disabled-12345",
    "name": "disabled-server",
    "command": "docker",
    "args": ["run", "-i", "--rm", "mcp/disabled"],
    "env": None,
    "enabled": False,
    "user_id": "user1"
}

# Sample MCP configuration JSON format for Claude Desktop
sample_mcp_config_json = {
    "mcpServers": {
        "filesystem": {
            "command": "docker",
            "args": ["run", "-i", "--rm", "mcp/filesystem", "/path/to/allowed/files"]
        },
        "git": {
            "command": "docker",
            "args": ["run", "-i", "--rm", "mcp/git", "/path/to/git/repo"]
        },
        "github": {
            "command": "docker",
            "args": ["run", "-i", "--rm", "mcp/github"],
            "env": {
                "GITHUB_PERSONAL_ACCESS_TOKEN": "github_pat_12345"
            }
        },
        "postgres": {
            "command": "docker",
            "args": ["run", "-i", "--rm", "mcp/postgres", "postgresql://localhost/mydb"]
        }
    }
}

# Sample list of MCP server configurations
sample_mcp_configs = [
    sample_filesystem_config,
    sample_git_config,
    sample_github_config,
    sample_postgres_config,
    sample_npx_config,
    sample_uvx_config,
    sample_disabled_config
]

# Sample MCP server status for running container
sample_running_status = {
    "id": "mcp-fs-12345",
    "name": "filesystem",
    "enabled": True,
    "status": "running",
    "container_id": "container1",
    "error_message": None
}

# Sample MCP server status for stopped container
sample_stopped_status = {
    "id": "mcp-postgres-12345",
    "name": "postgres",
    "enabled": True,
    "status": "exited",
    "container_id": "container3",
    "error_message": None
}

# Sample MCP server status with error
sample_error_status = {
    "id": "mcp-error-12345",
    "name": "error-server",
    "enabled": True,
    "status": "error",
    "container_id": None,
    "error_message": "Failed to start container: Port is already in use"
}

# Sample MCP server statuses
sample_mcp_statuses = [
    sample_running_status,
    sample_stopped_status,
    sample_error_status
]


def get_sample_mcp_config(config_id: str) -> Dict[str, Any]:
    """
    Get a sample MCP server configuration by ID.
    
    Args:
        config_id: The ID of the configuration to get
        
    Returns:
        A sample MCP server configuration
    """
    for config in sample_mcp_configs:
        if config["id"] == config_id:
            return config
    
    return sample_filesystem_config


def get_sample_mcp_configs_by_user(user_id: str) -> List[Dict[str, Any]]:
    """
    Get sample MCP server configurations by user ID.
    
    Args:
        user_id: The ID of the user
        
    Returns:
        A list of sample MCP server configurations
    """
    return [config for config in sample_mcp_configs if config["user_id"] == user_id]


def get_sample_mcp_status(config_id: str) -> Dict[str, Any]:
    """
    Get a sample MCP server status by configuration ID.
    
    Args:
        config_id: The ID of the configuration
        
    Returns:
        A sample MCP server status
    """
    for status in sample_mcp_statuses:
        if status["id"] == config_id:
            return status
    
    return sample_running_status
