"""
Docker configuration utilities.

This module provides utilities for parsing and validating Docker configurations.
"""

import os
import json
import yaml
from typing import Dict, List, Any, Optional, Union
import re
from pathlib import Path


def parse_docker_config(config_path: str) -> Dict[str, Any]:
    """
    Parse a Docker configuration file (YAML or JSON).
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Parsed configuration as a dictionary
        
    Raises:
        ValueError: If the file format is not supported or the file cannot be parsed
    """
    if not os.path.exists(config_path):
        raise ValueError(f"Configuration file not found: {config_path}")
    
    file_ext = os.path.splitext(config_path)[1].lower()
    
    try:
        with open(config_path, "r") as f:
            if file_ext in [".yaml", ".yml"]:
                return yaml.safe_load(f)
            elif file_ext == ".json":
                return json.load(f)
            else:
                raise ValueError(f"Unsupported configuration file format: {file_ext}")
    except Exception as e:
        raise ValueError(f"Failed to parse configuration file: {str(e)}")


def validate_docker_config(config: Dict[str, Any]) -> List[str]:
    """
    Validate a Docker configuration.
    
    Args:
        config: Docker configuration dictionary
        
    Returns:
        List of validation errors, empty if valid
    """
    errors = []
    
    # Check version
    if "version" not in config:
        errors.append("Missing 'version' field")
    
    # Check services
    if "services" not in config:
        errors.append("Missing 'services' field")
    else:
        services = config["services"]
        
        # Check if there's at least one service
        if not services:
            errors.append("No services defined")
        
        # Check each service
        for service_name, service_config in services.items():
            # Check image or build
            if "image" not in service_config and "build" not in service_config:
                errors.append(f"Service '{service_name}' must specify 'image' or 'build'")
            
            # Check volumes
            if "volumes" in service_config:
                for volume in service_config["volumes"]:
                    if ":" not in volume:
                        continue  # Named volume, no validation needed
                    
                    # Check bind mount
                    host_path = volume.split(":")[0]
                    if "${" in host_path:
                        # Environment variable used, can't validate
                        continue
                    
                    # Absolute path validation
                    if not os.path.isabs(host_path):
                        errors.append(f"Service '{service_name}' volume '{volume}' must use absolute path for host directory")
    
    return errors


def generate_docker_run_args(image: str, options: Dict[str, Any]) -> List[str]:
    """
    Generate 'docker run' arguments from options.
    
    Args:
        image: Docker image name
        options: Docker run options
        
    Returns:
        List of 'docker run' arguments
    """
    args = ["run"]
    
    # Add options
    if options.get("detach", False):
        args.append("-d")
    
    if options.get("interactive", False):
        args.append("-i")
    
    if options.get("tty", False):
        args.append("-t")
    
    if options.get("rm", False):
        args.append("--rm")
    
    # Add name
    if "name" in options:
        args.extend(["--name", options["name"]])
    
    # Add environment variables
    if "env" in options:
        for key, value in options["env"].items():
            args.extend(["-e", f"{key}={value}"])
    
    # Add ports
    if "ports" in options:
        for port in options["ports"]:
            args.extend(["-p", port])
    
    # Add volumes
    if "volumes" in options:
        for volume in options["volumes"]:
            args.extend(["-v", volume])
    
    # Add networks
    if "networks" in options:
        for network in options["networks"]:
            args.extend(["--network", network])
    
    # Add other options
    if "extra_options" in options:
        args.extend(options["extra_options"])
    
    # Add image
    args.append(image)
    
    # Add command
    if "command" in options:
        args.extend(options["command"] if isinstance(options["command"], list) else [options["command"]])
    
    return args


def transform_mcp_command(command: str, args: List[str]) -> Dict[str, Any]:
    """
    Transform an MCP server command (like npx or uvx) to Docker configuration.
    
    Args:
        command: The command (e.g., 'npx', 'uvx')
        args: Command arguments
        
    Returns:
        Docker run options dictionary
    """
    # Default options
    options = {
        "detach": True,
        "interactive": True,
        "rm": True
    }
    
    # Handle different commands
    if command == "npx":
        # Use Node.js image for npx
        options["image"] = "node:latest"
        options["command"] = ["npx"] + args
    elif command == "uvx":
        # Use Python image for uvx
        options["image"] = "python:latest"
        options["command"] = ["pip", "install", "-q", "uvx", "&&", "uvx"] + args
    else:
        # For other commands, use a minimal image
        options["image"] = "alpine:latest"
        options["command"] = [command] + args
    
    return options


def parse_docker_image_tag(image_spec: str) -> Dict[str, str]:
    """
    Parse a Docker image tag specification.
    
    Args:
        image_spec: Image specification (e.g., 'mcp/filesystem:latest')
        
    Returns:
        Dictionary with 'repository' and 'tag'
    """
    if ":" in image_spec:
        repository, tag = image_spec.split(":", 1)
    else:
        repository = image_spec
        tag = "latest"
    
    return {"repository": repository, "tag": tag}


def generate_mcp_config_json(configs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate MCP configuration JSON from a list of configurations.
    
    Args:
        configs: List of MCP server configurations
        
    Returns:
        MCP configuration JSON for Claude Desktop and other MCP clients
    """
    mcp_servers = {}
    
    for config in configs:
        if config.get("enabled", True):
            server_config = {
                "command": config["command"],
                "args": config["args"]
            }
            
            if "env" in config and config["env"]:
                server_config["env"] = config["env"]
            
            mcp_servers[config["name"]] = server_config
    
    return {"mcpServers": mcp_servers}
