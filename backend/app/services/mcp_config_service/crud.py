# backend/app/services/mcp_config_service/crud.py
"""
CRUD operations for MCP Server Configurations.
"""
import uuid
import logging
from typing import List, Optional

from sqlalchemy.orm import Session
from fastapi import HTTPException, status as fastapi_status

from app.models.mcp_config import MCPServerConfig
from app.schemas.mcp import MCPServerConfigCreate, MCPServerConfigUpdate


logger = logging.getLogger(__name__)

# --- CRUD Functions ---

def create_config(db: Session, config: MCPServerConfigCreate, user_id: str) -> MCPServerConfig:
    """ Create a new MCP server configuration. """
    config_id = str(uuid.uuid4())
    db_config = MCPServerConfig(
        id=config_id,
        name=config.name,
        server_type="custom", # Default type
        status="stopped",
        user_id=user_id,
        config={
            "command": config.command,
            "args": config.args,
            "env": config.env,
            "enabled": config.enabled
        }
    )
    db.add(db_config)
    db.commit()
    db.refresh(db_config)
    logger.info(f"Created MCP server configuration: {db_config.name} (ID: {db_config.id})")
    return db_config

def get_config_by_id(db: Session, config_id: str) -> Optional[MCPServerConfig]:
    """ Get an MCP server configuration by ID. """
    return db.query(MCPServerConfig).filter(MCPServerConfig.id == config_id).first()

def get_configs_by_user(db: Session, user_id: str) -> List[MCPServerConfig]:
    """ Get all MCP server configurations for a user. """
    return db.query(MCPServerConfig).filter(MCPServerConfig.user_id == user_id).all()

def get_all_enabled_configs(db: Session) -> List[MCPServerConfig]:
    """ Get all MCP server configurations that are marked as enabled. """
    # Query based on the 'enabled' key within the JSONB 'config' field
    # Note: JSONB operators might differ slightly based on DB dialect (e.g., PostgreSQL vs SQLite)
    # This uses standard SQLAlchemy JSON access which should work for SQLite >= 3.38.0
    # and PostgreSQL.
    return db.query(MCPServerConfig).filter(MCPServerConfig.config["enabled"].as_boolean() == True).all() # noqa

def update_config(db: Session, config_id: str, config_update: MCPServerConfigUpdate) -> Optional[MCPServerConfig]:
    """ Update an MCP server configuration. """
    db_config = get_config_by_id(db, config_id) # Use local function
    if not db_config: return None
    update_data = config_update.model_dump(exclude_unset=True)
    # Convert update fields to config fields
    config_fields = {}
    direct_fields = {}

    for key, value in update_data.items():
        if key in ['name', 'description', 'is_active', 'server_type', 'base_url', 'api_key', 'port']:
            direct_fields[key] = value
        elif key in ['command', 'args', 'env', 'enabled']:
            config_fields[key] = value

    # Update direct fields
    for key, value in direct_fields.items():
        setattr(db_config, key, value)

    # Update config fields
    if config_fields:
        # Create a copy to ensure SQLAlchemy detects the change
        new_config = (db_config.config or {}).copy()
        for key, value in config_fields.items():
            new_config[key] = value
        db_config.config = new_config # Assign the new dictionary
    db.commit()
    db.refresh(db_config)
    logger.info(f"Updated MCP server configuration: {db_config.name} (ID: {db_config.id})")
    return db_config

def delete_config(db: Session, config_id: str) -> bool:
    """ Delete an MCP server configuration. Does NOT stop the server first. """
    db_config = get_config_by_id(db, config_id) # Use local function
    if not db_config: return False
    # Removed stop_server call to break circular dependency
    # Stopping should be handled by the caller (e.g., API route) before deleting.
    db.delete(db_config)
    db.commit()
    logger.info(f"Deleted MCP server configuration: {db_config.name} (ID: {db_config.id})")
    return True