"""
Tests for Docker configuration validation.

This module contains test cases for Docker configuration validation
and command transformation.
"""

import pytest
from unittest.mock import patch, MagicMock
import os
import json

from app.services.mcp_config_service import MCPConfigService


class TestDockerConfigValidation:
    """Test cases for Docker configuration validation."""
    
    def test_transform_docker_command(self):
        """Test transforming a regular Docker command."""
        command = "docker"
        args = ["run", "-i", "--rm", "mcp/filesystem", "/path/to/allowed/files"]
        
        result = MCPConfigService._transform_command_to_docker(command, args)
        
        # For docker command, args should be returned as-is
        assert result == args
    
    def test_transform_npx_command(self):
        """Test transforming an npx command to a Docker command."""
        command = "npx"
        args = ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/allowed/files"]
        
        result = MCPConfigService._transform_command_to_docker(command, args)
        
        # For npx command, args should be transformed to docker run with node
        assert "run" in result
        assert "--rm" in result
        assert "-i" in result
        assert "node:latest" in result
        assert "npx" in result
        assert "-y" in result
        assert "@modelcontextprotocol/server-filesystem" in result
        assert "/path/to/allowed/files" in result
    
    def test_transform_uvx_command(self):
        """Test transforming a uvx command to a Docker command."""
        command = "uvx"
        args = ["mcp-server-git", "--repository", "path/to/git/repo"]
        
        result = MCPConfigService._transform_command_to_docker(command, args)
        
        # For uvx command, args should be transformed to docker run with python
        assert "run" in result
        assert "--rm" in result
        assert "-i" in result
        assert "python:latest" in result
        assert "pip" in result
        assert "install" in result
        assert "uvx" in result
        assert "mcp-server-git" in result
        assert "--repository" in result
        assert "path/to/git/repo" in result
    
    def test_transform_command_with_run(self):
        """Test transforming a command that already includes 'run'."""
        command = "npx"
        args = ["run", "-i", "--rm", "node:latest", "npx", "-y", "@modelcontextprotocol/server-filesystem", "/path/to/allowed/files"]
        
        result = MCPConfigService._transform_command_to_docker(command, args)
        
        # For command with run, args should be returned as-is
        assert result == args
    
    def test_get_container_name(self):
        """Test generating a standardized container name."""
        config_id = "mcp-fs-12345"
        
        result = MCPConfigService._get_container_name(config_id)
        
        # Container name should be mcp-<config_id>
        assert result == "mcp-mcp-fs-12345"
    
    @patch("docker.from_env")
    def test_get_docker_client(self, mock_from_env):
        """Test getting a Docker client."""
        mock_from_env.return_value = "mock_docker_client"
        
        result = MCPConfigService._get_docker_client()
        
        # Should call docker.from_env() and return the result
        mock_from_env.assert_called_once()
        assert result == "mock_docker_client"
    
    @patch("docker.from_env")
    def test_get_docker_client_exception(self, mock_from_env):
        """Test getting a Docker client with an exception."""
        # Setup mock to raise an exception
        mock_from_env.side_effect = Exception("Docker error")
        
        # Call the method and check that it raises an HTTPException
        with pytest.raises(Exception) as excinfo:
            MCPConfigService._get_docker_client()
        
        # Check the exception details
        assert "Docker error" in str(excinfo.value)


class TestDockerConfigIntegration:
    """Test cases for Docker configuration integration with MCP service."""
    
    def test_docker_args_extraction(self):
        """Test extracting Docker image name and command from args."""
        # Using a private method indirectly through start_server
        # This is testing the logic of parsing Docker run arguments
        args = ["run", "-i", "--rm", "mcp/filesystem", "/path/to/allowed/files"]
        
        # Mock Docker client and container
        with patch("app.services.mcp_config_service.MCPConfigService._get_docker_client") as mock_get_client:
            # Create a mock Docker client
            mock_client = MagicMock()
            mock_get_client.return_value = mock_client

            # Create a mock containers collection
            mock_containers = MagicMock()
            mock_client.containers = mock_containers

            # Configure the list method to return an empty list
            mock_containers.list.return_value = []

            # Configure the run method to return a mock container
            mock_container = MagicMock()
            mock_container.id = "mock_container_id"
            mock_container.status = "running"
            mock_containers.run.return_value = mock_container
            
            # Configure get_config_by_id to return a mock config
            with patch("app.services.mcp_config_service.MCPConfigService.get_config_by_id") as mock_get_config:
                mock_config = MagicMock()
                mock_config.id = "mock_config_id"
                mock_config.name = "mock_config"
                mock_config.command = "docker"
                mock_config.args = args
                mock_config.env = None
                mock_config.enabled = True
                mock_get_config.return_value = mock_config
                
                # Configure get_config_status to return a mock status
                with patch("app.services.mcp_config_service.MCPConfigService.get_config_status") as mock_get_status:
                    mock_status = MagicMock()
                    mock_status.id = "mock_config_id"
                    mock_status.name = "mock_config"
                    mock_status.enabled = True
                    mock_status.status = "running"
                    mock_status.container_id = "mock_container_id"
                    mock_status.error_message = None
                    mock_get_status.return_value = mock_status

                    # Call the method
                    MCPConfigService.start_server(MagicMock(), "mock_config_id")

                    # Check that containers.run was called with the correct arguments
                    # Note: This is verifying that the image and command parsing works correctly
                    mock_containers.run.assert_called_once()
                    run_args, run_kwargs = mock_containers.run.call_args

                    # Check the image keyword argument
                    assert run_kwargs["image"] == "mcp/filesystem"

                    # The command keyword argument should contain the command
                    assert run_kwargs["command"] == ["/path/to/allowed/files"]
    
    def test_environment_variables_handling(self):
        """Test handling of environment variables in Docker commands."""
        # Setup
        args = ["run", "-i", "--rm", "mcp/github"]
        env = {"GITHUB_PERSONAL_ACCESS_TOKEN": "github_pat_12345"}
        
        # Mock Docker client and container
        with patch("app.services.mcp_config_service.MCPConfigService._get_docker_client") as mock_get_client:
            # Create a mock Docker client
            mock_client = MagicMock()
            mock_get_client.return_value = mock_client

            # Create a mock containers collection
            mock_containers = MagicMock()
            mock_client.containers = mock_containers

            # Configure the list method to return an empty list
            mock_containers.list.return_value = []

            # Configure the run method to return a mock container
            mock_container = MagicMock()
            mock_container.id = "mock_container_id"
            mock_container.status = "running"
            mock_containers.run.return_value = mock_container
            
            # Configure get_config_by_id to return a mock config
            with patch("app.services.mcp_config_service.MCPConfigService.get_config_by_id") as mock_get_config:
                mock_config = MagicMock()
                mock_config.id = "mock_config_id"
                mock_config.name = "mock_config"
                mock_config.command = "docker"
                mock_config.args = args
                mock_config.env = env
                mock_config.enabled = True
                mock_get_config.return_value = mock_config
                
                # Configure get_config_status to return a mock status
                with patch("app.services.mcp_config_service.MCPConfigService.get_config_status") as mock_get_status:
                    mock_status = MagicMock()
                    mock_status.id = "mock_config_id"
                    mock_status.name = "mock_config"
                    mock_status.enabled = True
                    mock_status.status = "running"
                    mock_status.container_id = "mock_container_id"
                    mock_status.error_message = None
                    mock_get_status.return_value = mock_status

                    # Call the method
                    MCPConfigService.start_server(MagicMock(), "mock_config_id")

                    # Check that containers.run was called with the correct environment variables
                    mock_containers.run.assert_called_once()
                    _, run_kwargs = mock_containers.run.call_args
                    
                    # The environment keyword argument should contain the environment variables
                    assert run_kwargs["environment"] == env
