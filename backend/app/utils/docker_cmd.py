"""
Docker command utilities.

This module provides utilities for building and validating Docker commands.
"""

import os
import re
import shlex
import subprocess
from typing import Dict, List, Any, Optional, Tuple, Union
import logging

# Configure logging
logger = logging.getLogger(__name__)


def validate_docker_command(args: List[str]) -> Tuple[bool, Optional[str]]:
    """
    Validate a Docker command for security.
    
    Args:
        args: Docker command arguments
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Basic validation
    if not args:
        return False, "Empty command"
    
    # Check for dangerous options
    dangerous_options = [
        "--privileged",
        "--cap-add=all",
        "--security-opt=seccomp=unconfined",
        "--device=/dev/mem",
        "-v", "/:/host"  # Mount host root
    ]
    
    for i, arg in enumerate(args):
        # Check for dangerous options
        if arg in dangerous_options:
            return False, f"Forbidden option: {arg}"
        
        # Check for volume mounts that access sensitive directories
        if arg == "-v" or arg == "--volume":
            if i + 1 < len(args):
                volume_arg = args[i + 1]
                if ":" in volume_arg:
                    host_path, container_path = volume_arg.split(":", 1)
                    
                    # Check for sensitive host paths
                    sensitive_paths = [
                        "/etc/shadow",
                        "/etc/passwd",
                        "/etc/ssh",
                        "/root/.ssh",
                        "/var/run/docker.sock"  # Exception: this might be needed for Docker-in-Docker
                    ]
                    
                    for path in sensitive_paths:
                        if path != "/var/run/docker.sock" and (host_path == path or host_path.startswith(path + "/")):
                            return False, f"Forbidden volume mount: {volume_arg}"
    
    return True, None


def build_docker_run_command(
    image: str,
    container_name: Optional[str] = None,
    env_vars: Optional[Dict[str, str]] = None,
    volumes: Optional[List[str]] = None,
    ports: Optional[List[str]] = None,
    network: Optional[str] = None,
    command: Optional[List[str]] = None,
    interactive: bool = False,
    detach: bool = True,
    remove: bool = False,
    extra_args: Optional[List[str]] = None
) -> List[str]:
    """
    Build a 'docker run' command with the given options.
    
    Args:
        image: Docker image name
        container_name: Name for the container
        env_vars: Environment variables
        volumes: Volume mounts
        ports: Port mappings
        network: Network to join
        command: Command to run in the container
        interactive: Whether to run in interactive mode
        detach: Whether to run in detached mode
        remove: Whether to remove the container when it exits
        extra_args: Additional arguments
        
    Returns:
        Docker run command as a list of arguments
    """
    cmd = ["docker", "run"]
    
    # Add options
    if detach:
        cmd.append("-d")
    
    if interactive:
        cmd.append("-i")
    
    if remove:
        cmd.append("--rm")
    
    if container_name:
        cmd.extend(["--name", container_name])
    
    # Add environment variables
    if env_vars:
        for key, value in env_vars.items():
            cmd.extend(["-e", f"{key}={value}"])
    
    # Add volumes
    if volumes:
        for volume in volumes:
            cmd.extend(["-v", volume])
    
    # Add ports
    if ports:
        for port in ports:
            cmd.extend(["-p", port])
    
    # Add network
    if network:
        cmd.extend(["--network", network])
    
    # Add extra arguments
    if extra_args:
        cmd.extend(extra_args)
    
    # Add image
    cmd.append(image)
    
    # Add command
    if command:
        cmd.extend(command)
    
    return cmd


def build_mcp_server_command(
    server_type: str,
    args: List[str],
    env_vars: Optional[Dict[str, str]] = None,
    container_name: Optional[str] = None
) -> List[str]:
    """
    Build a command for running an MCP server with Docker.
    
    Args:
        server_type: Type of MCP server (filesystem, git, github, postgres, etc.)
        args: Server-specific arguments
        env_vars: Environment variables
        container_name: Name for the Docker container
        
    Returns:
        Docker command as a list of arguments
    """
    # Base options
    docker_opts = {
        "interactive": True,
        "detach": True,
        "remove": True,
        "container_name": container_name
    }
    
    # Standard MCP server images
    server_images = {
        "filesystem": "mcp/filesystem",
        "git": "mcp/git",
        "github": "mcp/github",
        "postgres": "mcp/postgres",
        "memory": "mcp/memory",
        "slack": "mcp/slack",
        "brave": "mcp/brave",
        "google-drive": "mcp/google-drive",
        "sequential-thinking": "mcp/sequential-thinking"
    }
    
    # Get image for server type
    image = server_images.get(server_type, f"mcp/{server_type}")
    
    # Build docker run command
    return build_docker_run_command(
        image=image,
        container_name=container_name,
        env_vars=env_vars,
        command=args,
        interactive=docker_opts["interactive"],
        detach=docker_opts["detach"],
        remove=docker_opts["remove"]
    )


def execute_docker_command(cmd: List[str], check: bool = True) -> subprocess.CompletedProcess:
    """
    Execute a Docker command.
    
    Args:
        cmd: Docker command as a list of arguments
        check: Whether to check the return code
        
    Returns:
        Completed process
        
    Raises:
        subprocess.CalledProcessError: If the command fails and check is True
    """
    logger.info(f"Executing Docker command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            check=check,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        if result.returncode != 0:
            logger.error(f"Docker command failed: {result.stderr}")
        
        return result
    except subprocess.CalledProcessError as e:
        logger.error(f"Docker command failed: {e}")
        raise


def parse_docker_container_id(output: str) -> Optional[str]:
    """
    Parse a Docker container ID from command output.
    
    Args:
        output: Command output
        
    Returns:
        Container ID or None if not found
    """
    # Docker container IDs are 64-character hexadecimal strings
    # But Docker commands usually return the first 12 characters
    pattern = r"([0-9a-f]{12}|[0-9a-f]{64})"
    match = re.search(pattern, output)
    
    if match:
        return match.group(1)
    
    return None


def get_docker_container_status(container_id: str) -> Dict[str, Any]:
    """
    Get the status of a Docker container.
    
    Args:
        container_id: Container ID or name
        
    Returns:
        Container status information
        
    Raises:
        subprocess.CalledProcessError: If the command fails
    """
    cmd = ["docker", "inspect", "--format", "{{json .State}}", container_id]
    result = execute_docker_command(cmd)
    
    if result.returncode == 0:
        try:
            return {"status": "success", "data": json.loads(result.stdout)}
        except json.JSONDecodeError:
            return {"status": "error", "error": "Failed to parse container status"}
    else:
        return {"status": "error", "error": result.stderr}
