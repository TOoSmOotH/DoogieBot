"""
MCP (Model Context Protocol) API routes.

This module defines the API endpoints for managing MCP server configurations
and controlling MCP servers.
"""

from typing import List
# Import status with an alias to avoid conflicts
from fastapi import APIRouter, Depends, HTTPException, status as fastapi_status
from sqlalchemy.orm import Session

from app.db.base import get_db
from app.models.user import User, UserRole
from app.schemas.mcp import (
    MCPServerConfigCreate,
    MCPServerConfigUpdate,
    MCPServerConfigResponse,
    MCPServerStatus,
    MCPConfigJSON
)
# Import functions directly from the new package
from app.services.mcp_config_service import (
    create_config,
    get_configs_by_user,
    get_config_by_id,
    update_config,
    delete_config,
    stop_server,
    get_config_status,
    start_server,
    restart_server,
    generate_mcp_config_json
)
from app.utils.deps import get_current_user

router = APIRouter()

@router.post("/configs", response_model=MCPServerConfigResponse)
async def create_mcp_config(
    config: MCPServerConfigCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Create a new MCP server configuration.

    Only admin users can create MCP configurations.
    """
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=fastapi_status.HTTP_403_FORBIDDEN, # Use alias
            detail="Only admin users can create MCP configurations"
        )

    db_config = create_config(db, config, current_user.id) # Use imported function
    return db_config

@router.get("/configs", response_model=List[MCPServerConfigResponse])
async def get_mcp_configs(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get all MCP server configurations for the current user.
    """
    return get_configs_by_user(db, current_user.id) # Use imported function

@router.get("/configs/{config_id}", response_model=MCPServerConfigResponse)
async def get_mcp_config(
    config_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get an MCP server configuration by ID.
    """
    db_config = get_config_by_id(db, config_id) # Use imported function
    if not db_config or db_config.user_id != current_user.id:
        raise HTTPException(
            status_code=fastapi_status.HTTP_404_NOT_FOUND, # Use alias
            detail="MCP configuration not found"
        )
    return db_config

@router.put("/configs/{config_id}", response_model=MCPServerConfigResponse)
async def update_mcp_config(
    config_id: str,
    config_update: MCPServerConfigUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Update an MCP server configuration.

    Only admin users can update MCP configurations.
    """
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=fastapi_status.HTTP_403_FORBIDDEN, # Use alias
            detail="Only admin users can update MCP configurations"
        )

    db_config = get_config_by_id(db, config_id) # Use imported function
    if not db_config or db_config.user_id != current_user.id:
        raise HTTPException(
            status_code=fastapi_status.HTTP_404_NOT_FOUND, # Use alias
            detail="MCP configuration not found"
        )

    updated_config = update_config(db, config_id, config_update) # Use imported function
    return updated_config

@router.delete("/configs/{config_id}", status_code=fastapi_status.HTTP_204_NO_CONTENT) # Use alias
async def delete_mcp_config(
    config_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Delete an MCP server configuration.

    Only admin users can delete MCP configurations.
    """
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=fastapi_status.HTTP_403_FORBIDDEN, # Use alias
            detail="Only admin users can delete MCP configurations"
        )

    db_config = get_config_by_id(db, config_id) # Use imported function
    if not db_config or db_config.user_id != current_user.id:
        raise HTTPException(
            status_code=fastapi_status.HTTP_404_NOT_FOUND, # Use alias
            detail="MCP configuration not found"
        )

    # Stop the server if it's running
    try:
        stop_server(db, config_id) # Use imported function
    except Exception as e:
        raise HTTPException(
            status_code=fastapi_status.HTTP_500_INTERNAL_SERVER_ERROR, # Use alias
            detail=f"Failed to stop MCP server: {str(e)}"
        )

    # Delete the configuration
    success = delete_config(db, config_id) # Use imported function
    if not success:
        raise HTTPException(
            status_code=fastapi_status.HTTP_500_INTERNAL_SERVER_ERROR, # Use alias
            detail="Failed to delete MCP configuration"
        )

@router.get("/configs/{config_id}/status", response_model=MCPServerStatus)
async def get_mcp_config_status(
    config_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get the status of an MCP server.
    """
    db_config = get_config_by_id(db, config_id) # Use imported function
    if not db_config or db_config.user_id != current_user.id:
        raise HTTPException(
            status_code=fastapi_status.HTTP_404_NOT_FOUND, # Use alias
            detail="MCP configuration not found"
        )

    status_result = get_config_status(db, config_id) # Use imported function
    if not status_result:
        raise HTTPException(
            status_code=fastapi_status.HTTP_500_INTERNAL_SERVER_ERROR, # Use alias
            detail="Failed to get MCP server status"
        )

    return status_result

@router.post("/configs/{config_id}/start", response_model=MCPServerStatus)
async def start_mcp_server(
    config_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Start an MCP server.

    Only admin users can start MCP servers.
    """
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=fastapi_status.HTTP_403_FORBIDDEN, # Use alias
            detail="Only admin users can start MCP servers"
        )

    db_config = get_config_by_id(db, config_id) # Use imported function
    if not db_config or db_config.user_id != current_user.id:
        raise HTTPException(
            status_code=fastapi_status.HTTP_404_NOT_FOUND, # Use alias
            detail="MCP configuration not found"
        )

    if not db_config.config or not db_config.config.get('enabled', False):
        raise HTTPException(
            status_code=fastapi_status.HTTP_400_BAD_REQUEST, # Use alias
            detail="Cannot start disabled MCP server"
        )

    status_result = start_server(db, config_id) # Use imported function
    if not status_result:
        raise HTTPException(
            status_code=fastapi_status.HTTP_500_INTERNAL_SERVER_ERROR, # Use alias
            detail="Failed to start MCP server"
        )

    if status_result.status == "error":
        raise HTTPException(
            status_code=fastapi_status.HTTP_500_INTERNAL_SERVER_ERROR, # Use alias
            detail=f"Failed to start MCP server: {status_result.error_message}"
        )

    return status_result

@router.post("/configs/{config_id}/stop", response_model=MCPServerStatus)
async def stop_mcp_server(
    config_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Stop an MCP server.

    Only admin users can stop MCP servers.
    """
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=fastapi_status.HTTP_403_FORBIDDEN, # Use alias
            detail="Only admin users can stop MCP servers"
        )

    db_config = get_config_by_id(db, config_id) # Use imported function
    if not db_config or db_config.user_id != current_user.id:
        raise HTTPException(
            status_code=fastapi_status.HTTP_404_NOT_FOUND, # Use alias
            detail="MCP configuration not found"
        )

    status_result = MCPConfigService.stop_server(db, config_id) # Renamed variable
    if not status_result:
        raise HTTPException(
            status_code=fastapi_status.HTTP_500_INTERNAL_SERVER_ERROR, # Use alias
            detail="Failed to stop MCP server"
        )

    if status_result.status == "error":
        raise HTTPException(
            status_code=fastapi_status.HTTP_500_INTERNAL_SERVER_ERROR, # Use alias
            detail=f"Failed to stop MCP server: {status_result.error_message}"
        )

    return status_result

@router.post("/configs/{config_id}/restart", response_model=MCPServerStatus)
async def restart_mcp_server(
    config_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Restart an MCP server.

    Only admin users can restart MCP servers.
    """
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=fastapi_status.HTTP_403_FORBIDDEN, # Use alias
            detail="Only admin users can restart MCP servers"
        )

    db_config = get_config_by_id(db, config_id) # Use imported function
    if not db_config or db_config.user_id != current_user.id:
        raise HTTPException(
            status_code=fastapi_status.HTTP_404_NOT_FOUND, # Use alias
            detail="MCP configuration not found"
        )

    status_result = restart_server(db, config_id) # Use imported function
    if not status_result:
        raise HTTPException(
            status_code=fastapi_status.HTTP_500_INTERNAL_SERVER_ERROR, # Use alias
            detail="Failed to restart MCP server"
        )

    if status_result.status == "error":
        raise HTTPException(
            status_code=fastapi_status.HTTP_500_INTERNAL_SERVER_ERROR, # Use alias
            detail=f"Failed to restart MCP server: {status_result.error_message}"
        )

    return status_result

@router.get("/configs/export/json", response_model=MCPConfigJSON) # Changed path
async def get_mcp_config_json(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get the MCP configuration JSON for Claude Desktop.
    """
    return generate_mcp_config_json(db, current_user.id) # Use imported function
