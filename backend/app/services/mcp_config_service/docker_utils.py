# backend/app/services/mcp_config_service/docker_utils.py
"""
Docker utility functions for MCP Service.
"""
import logging
from typing import List
import docker
from docker.errors import DockerException
from fastapi import HTTPException, status as fastapi_status

logger = logging.getLogger(__name__)

def _get_docker_client():
    """ Get a Docker client instance. """
    try:
        # Connect to the Docker-in-Docker sidecar service via TCP
        client = docker.DockerClient(base_url='tcp://dind-server:2375', timeout=10)
        # Verify connection by pinging the daemon
        client.ping()
        logger.info("Successfully connected to Docker daemon via tcp://dind-server:2375")
        return client
    except DockerException as e:
        logger.error(f"Failed to initialize Docker client using tcp://dind-server:2375: {e}")
        # Optionally, you could try falling back to docker.from_env() here if needed,
        # but the explicit path is usually required for Docker Desktop mounts.
        raise HTTPException(status_code=fastapi_status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Docker connection error: {str(e)}")
    except Exception as e: # Catch other potential errors like file not found if mount failed
        logger.error(f"Unexpected error getting Docker client: {e}")
        raise HTTPException(status_code=fastapi_status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Docker setup error: {str(e)}")


def _get_container_name(config_id: str) -> str:
    """ Generate a standardized container name. """
    return f"mcp-{config_id}"

def _transform_command_to_docker(command: str, args: List[str]) -> List[str]:
    """ Transform npx/uvx commands to docker run commands if needed. """
    if command == "docker": return args
    if command == "npx" and "run" not in args:
        return ["run", "--rm", "-i", "node:latest", "npx"] + args
    if command == "uvx" and "run" not in args:
        return ["run", "--rm", "-i", "python:latest", "pip", "install", "-q", "uvx", "&&", "uvx"] + args
    if "run" not in args:
        # Assume the command itself is the image if 'run' is missing
        logger.warning(f"Assuming '{command}' is the image name as 'run' is missing from args: {args}")
        return ["run", "--rm", "-i", command] + args # Prepend run, rm, i
    return args # Return original args if 'run' is already present