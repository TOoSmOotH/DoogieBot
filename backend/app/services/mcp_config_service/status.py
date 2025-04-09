# backend/app/services/mcp_config_service/status.py
"""
Functionality related to checking MCP server status.
"""
import logging
import time # Import time for sleep
from typing import Optional

from sqlalchemy.orm import Session
from docker.errors import APIError, NotFound as DockerNotFound

from app.schemas.mcp import MCPServerStatus
from .crud import get_config_by_id # Use relative import
from .docker_utils import _get_docker_client, _get_container_name # Use relative import

logger = logging.getLogger(__name__)

def get_config_status(db: Session, config_id: str) -> Optional[MCPServerStatus]:
    """ Get the status of an MCP server container. """
    db_config = get_config_by_id(db, config_id)
    if not db_config: return None

    container_id, status_str, error_message = None, "stopped", None
    max_retries = 2 # Try initial + 1 retry
    retry_delay = 1 # seconds

    for attempt in range(max_retries):
        container_id, status_str, error_message = None, "stopped", None # Reset for retry
        try:
            docker_client = _get_docker_client()
            container_name = _get_container_name(config_id)
            logger.debug(f"Checking status for container '{container_name}' (Attempt {attempt + 1}/{max_retries})")
            try:
                containers = docker_client.containers.list(all=True, filters={"name": container_name})
                if containers:
                    container = containers[0]
                    container_id = container.id
                    current_docker_status = container.status
                    logger.debug(f"Found container {container_id} with Docker status: {current_docker_status}")

                    # Normalize status
                    if current_docker_status == "running":
                        status_str = "running"
                    elif current_docker_status in ["exited", "removing", "created", "paused"]: # Treat created/paused as stopped for simplicity
                        status_str = "stopped"
                    else: # Treat other statuses like 'dead' as error
                        error_message = f"Unexpected container status: {current_docker_status}"
                        status_str = "error"
                else:
                    logger.debug(f"Container '{container_name}' not found by list.")
                    status_str = "stopped" # Explicitly stopped if not found

            except DockerNotFound:
                 logger.debug(f"Container '{container_name}' not found (DockerNotFound).")
                 status_str = "stopped" # Explicitly stopped if not found
            except APIError as e:
                 logger.error(f"Docker API error checking status for {container_name}: {e}")
                 status_str, error_message = "error", f"Docker API error: {str(e)}"
                 break # Don't retry on API errors

            # If running, break retry loop immediately
            if status_str == "running":
                logger.debug(f"Container '{container_name}' confirmed running.")
                break

            # If not running and more retries left, wait and retry
            if attempt < max_retries - 1:
                logger.warning(f"Container '{container_name}' not running (status: {status_str}). Retrying after {retry_delay}s...")
                time.sleep(retry_delay)
            else:
                 logger.warning(f"Container '{container_name}' not running after {max_retries} attempts (final status: {status_str}).")


        except Exception as e:
            logger.exception(f"Unexpected error getting container status for {config_id}: {e}")
            status_str, error_message = "error", str(e)
            break # Don't retry on unexpected errors

    # Final status after retries (or break)
    return MCPServerStatus(
        id=config_id, name=db_config.name, enabled=db_config.config.get("enabled", False),
        status=status_str, container_id=container_id, error_message=error_message
    )