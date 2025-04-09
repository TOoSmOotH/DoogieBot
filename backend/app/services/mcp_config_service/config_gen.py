# backend/app/services/mcp_config_service/config_gen.py
"""
Functionality for generating MCP client configuration JSON.
"""
import logging
from typing import Dict, Any

from sqlalchemy.orm import Session

from .crud import get_configs_by_user # Use relative import

logger = logging.getLogger(__name__)

def generate_mcp_config_json(db: Session, user_id: str) -> Dict[str, Any]:
    """ Generate the MCP configuration JSON for Claude Desktop or other MCP clients. """
    configs = get_configs_by_user(db, user_id)
    mcp_servers = {}
    for config in configs:
        if config.config and config.config.get("enabled", False):
            server_config = {"command": config.config.get("command"), "args": config.config.get("args")}
            if config.config.get("env"): server_config["env"] = config.config.get("env")
            mcp_servers[config.name] = server_config
    return {"mcpServers": mcp_servers}