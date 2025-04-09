"""
Tests for the MCP configuration API and service.

This module contains test cases for the MCP configuration API and service.
"""

import pytest
import json
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session # Import Session
from datetime import datetime, timezone

from main import app
from app.models.user import User, UserRole
from app.utils.deps import get_current_user, get_db
from app.models.mcp_config import MCPServerConfig
from app.schemas.mcp import (
    MCPServerConfigCreate,
    MCPServerConfigUpdate,
    MCPServerConfigResponse,
    MCPServerStatus,
    MCPConfigJSON
)
from app.services.mcp_config_service import MCPConfigService
from tests.mocks.docker_client import MockDockerClient, mock_docker_from_env
from tests.mocks.mcp_config_samples import sample_mcp_config_json # Keep only needed import


# Create a test client
client = TestClient(app)

# Define mock users
mock_admin = User(id="user1", email="admin@example.com", role=UserRole.ADMIN, status="active")
mock_user = User(id="user2", email="user@example.com", role=UserRole.USER, status="active")


# Mock database session fixture
@pytest.fixture
def mock_db():
    """Mock database session."""
    db = MagicMock(spec=Session) # Use spec=Session
    query = MagicMock()
    db.query.return_value = query
    filtered_query = MagicMock()
    query.filter.return_value = filtered_query
    filtered_query.first.return_value = None
    filtered_query.all.return_value = []
    return db


@pytest.fixture
def mock_docker():
    """Mock Docker client."""
    with patch("docker.from_env", mock_docker_from_env):
        with patch("app.services.mcp_config_service.docker.from_env", mock_docker_from_env):
            yield


# Test cases for MCP Configuration service
class TestMCPConfigService:
    """Test cases for MCP Configuration service."""

    @patch("uuid.uuid4", return_value="new-mcp-config-id")
    def test_create_config(self, mock_uuid, mock_db):
        config_data = MCPServerConfigCreate(
            name="test-config", command="docker", args=["run", "mcp/test"], env={"TEST": "value"}, enabled=True
        )
        user_id = "user1"
        # Mock the refresh operation to avoid issues with MagicMock attributes
        mock_db.refresh = MagicMock()
        result = MCPConfigService.create_config(mock_db, config_data, user_id)
        mock_db.add.assert_called_once()
        added_obj = mock_db.add.call_args[0][0]
        assert isinstance(added_obj, MCPServerConfig)
        assert added_obj.name == "test-config"
        assert added_obj.user_id == user_id
        mock_db.commit.assert_called_once()
        mock_db.refresh.assert_called_once_with(added_obj)
        # Return the object passed to add, as refresh is mocked
        assert result is added_obj

    def test_get_config_by_id(self, mock_db):
        mock_config = MagicMock(spec=MCPServerConfig, id="mcp-config-id", name="test-config", user_id="user1")
        filtered_query = mock_db.query.return_value.filter.return_value
        filtered_query.first.return_value = mock_config
        result = MCPConfigService.get_config_by_id(mock_db, "mcp-config-id")
        mock_db.query.assert_called_once_with(MCPServerConfig)
        filtered_query.first.assert_called_once()
        assert result == mock_config

    def test_get_configs_by_user(self, mock_db):
        mock_config1 = MagicMock(spec=MCPServerConfig, id="mcp-config-id-1")
        mock_config2 = MagicMock(spec=MCPServerConfig, id="mcp-config-id-2")
        filtered_query = mock_db.query.return_value.filter.return_value
        filtered_query.all.return_value = [mock_config1, mock_config2]
        result = MCPConfigService.get_configs_by_user(mock_db, "user1")
        mock_db.query.assert_called_once_with(MCPServerConfig)
        filtered_query.all.assert_called_once()
        assert len(result) == 2

    def test_update_config(self, mock_db):
        mock_config = MCPServerConfig(id="mcp-config-id", name="old", enabled=True, command="cmd", args=["a"], user_id="u1") # Use real object
        filtered_query = mock_db.query.return_value.filter.return_value
        filtered_query.first.return_value = mock_config
        update_data = MCPServerConfigUpdate(name="new", enabled=False, args=["b"])
        # Mock refresh
        mock_db.refresh = MagicMock()
        result = MCPConfigService.update_config(mock_db, "mcp-config-id", update_data)
        filtered_query.first.assert_called_once()
        mock_db.commit.assert_called_once()
        mock_db.refresh.assert_called_once_with(mock_config)
        assert result.name == "new"
        assert result.enabled is False
        assert result.args == ["b"]

    def test_delete_config(self, mock_db):
        mock_config = MagicMock(spec=MCPServerConfig, id="mcp-config-id")
        filtered_query = mock_db.query.return_value.filter.return_value
        filtered_query.first.return_value = mock_config
        with patch.object(MCPConfigService, "stop_server") as mock_stop_server:
            result = MCPConfigService.delete_config(mock_db, "mcp-config-id")
            filtered_query.first.assert_called_once()
            mock_db.delete.assert_called_once_with(mock_config)
            mock_db.commit.assert_called_once()
            assert result is True
            mock_stop_server.assert_called_once_with(mock_db, "mcp-config-id")

    def test_get_config_status(self, mock_db, mock_docker):
        # Use MagicMock and set attributes
        mock_config = MagicMock(spec=MCPServerConfig)
        mock_config.id = "mcp-fs-12345"
        mock_config.name = "filesystem" # Set as string
        mock_config.enabled = True
        mock_config.container_id = "container1" # Model doesn't have this, but service uses it
        filtered_query = mock_db.query.return_value.filter.return_value
        filtered_query.first.return_value = mock_config
        # Mock the docker client's container status directly if needed,
        # but the error was in Pydantic validation due to mock attributes.
        # Let's assume the mock docker client works correctly now.
        result = MCPConfigService.get_config_status(mock_db, "mcp-fs-12345")
        filtered_query.first.assert_called_once()
        assert result.id == "mcp-fs-12345"
        assert isinstance(result.name, str) # Ensure name is string
        assert result.name == "filesystem"
        assert result.enabled is True
        assert result.status in ["running", "exited", "stopped", "error"]

    def test_start_server(self, mock_db, mock_docker):
        mock_config = MagicMock(spec=MCPServerConfig, id="mcp-fs-12345", name="filesystem", command="docker", args=["run", "img"], env=None, enabled=True, container_id=None)
        filtered_query = mock_db.query.return_value.filter.return_value
        filtered_query.first.return_value = mock_config
        with patch.object(MCPConfigService, "get_config_status") as mock_get_status:
            # Return a valid Pydantic model
            mock_get_status.return_value = MCPServerStatus(id="mcp-fs-12345", name="filesystem", enabled=True, status="running", container_id="container1", error_message=None)
            result = MCPConfigService.start_server(mock_db, "mcp-fs-12345")
            filtered_query.first.assert_called_once()
            assert result.status == "running"
            assert result.container_id == "container1"

    def test_stop_server(self, mock_db, mock_docker):
        mock_config = MagicMock(spec=MCPServerConfig, id="mcp-fs-12345", name="filesystem", enabled=True, container_id="container1")
        filtered_query = mock_db.query.return_value.filter.return_value
        filtered_query.first.return_value = mock_config
        with patch.object(MCPConfigService, "get_config_status") as mock_get_status:
            # Return a valid Pydantic model
            mock_get_status.return_value = MCPServerStatus(id="mcp-fs-12345", name="filesystem", enabled=True, status="exited", container_id=None, error_message=None)
            result = MCPConfigService.stop_server(mock_db, "mcp-fs-12345")
            filtered_query.first.assert_called_once()
            assert result.status == "exited"
            assert result.container_id is None

    def test_restart_server(self, mock_db, mock_docker):
        with patch.object(MCPConfigService, "stop_server") as mock_stop_server, \
             patch.object(MCPConfigService, "start_server") as mock_start_server:
            # Return valid Pydantic models
            mock_stop_server.return_value = MCPServerStatus(id="mcp-fs-12345", name="filesystem", enabled=True, status="exited", container_id=None, error_message=None)
            mock_start_server.return_value = MCPServerStatus(id="mcp-fs-12345", name="filesystem", enabled=True, status="running", container_id="container1", error_message=None)
            result = MCPConfigService.restart_server(mock_db, "mcp-fs-12345")
            mock_stop_server.assert_called_once_with(mock_db, "mcp-fs-12345")
            mock_start_server.assert_called_once_with(mock_db, "mcp-fs-12345")
            assert result.status == "running"

    def test_generate_mcp_config_json(self, mock_db):
        # Use real objects with string names
        mock_config1 = MCPServerConfig(name="filesystem", command="docker", args=["run", "fs"], env=None, enabled=True, id="id1", user_id="u1")
        mock_config2 = MCPServerConfig(name="github", command="docker", args=["run", "gh"], env={"TOKEN": "123"}, enabled=True, id="id2", user_id="u1")
        mock_config3 = MCPServerConfig(name="disabled", command="docker", args=["run", "dis"], env=None, enabled=False, id="id3", user_id="u1")
        with patch.object(MCPConfigService, "get_configs_by_user") as mock_get_configs:
            mock_get_configs.return_value = [mock_config1, mock_config2, mock_config3]
            result = MCPConfigService.generate_mcp_config_json(mock_db, "user1")
            assert "mcpServers" in result
            assert "filesystem" in result["mcpServers"] # Check with string key
            assert "github" in result["mcpServers"]     # Check with string key
            assert "disabled" not in result["mcpServers"]
            assert result["mcpServers"]["github"]["env"] == {"TOKEN": "123"}


# Test cases for MCP Configuration API
class TestMCPConfigAPI:
    """Test cases for MCP Configuration API."""

    @patch("app.api.routes.mcp.MCPConfigService.create_config")
    def test_create_mcp_config(self, mock_create_config, mock_db):
        mock_config_id="new-mcp-config-id"
        mock_user_id="user1"
        # Mock service return value with Pydantic model
        mock_create_config.return_value = MCPServerConfigResponse(
            id=mock_config_id, name="test-config", command="docker",
            args=["run", "-i", "--rm", "mcp/test"], env={"TEST": "value"},
            enabled=True, user_id=mock_user_id, container_id=None,
            created_at=datetime.now(timezone.utc), updated_at=datetime.now(timezone.utc)
        )
        request_data = {
            "name": "test-config", "command": "docker",
            "args": ["run", "-i", "--rm", "mcp/test"], "env": {"TEST": "value"},
            "enabled": True
        }
        app.dependency_overrides[get_db] = lambda: mock_db
        app.dependency_overrides[get_current_user] = lambda: mock_admin
        response = client.post("/api/v1/mcp/configs", json=request_data)
        app.dependency_overrides = {}
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == mock_config_id
        mock_create_config.assert_called_once()
        call_args, _ = mock_create_config.call_args
        assert call_args[0] is mock_db # Check db arg passed to service

    @patch("app.api.routes.mcp.MCPConfigService.get_configs_by_user")
    def test_get_mcp_configs(self, mock_get_configs, mock_db):
        mock_user_id = "user1"
        # Mock service return value with Pydantic models
        mock_config1 = MCPServerConfigResponse(id="id1", name="cfg1", command="docker", args=["run"], enabled=True, user_id=mock_user_id, container_id=None, created_at=datetime.now(timezone.utc), updated_at=datetime.now(timezone.utc))
        mock_config2 = MCPServerConfigResponse(id="id2", name="cfg2", command="docker", args=["run"], enabled=False, user_id=mock_user_id, container_id=None, created_at=datetime.now(timezone.utc), updated_at=datetime.now(timezone.utc))
        mock_get_configs.return_value = [mock_config1, mock_config2]
        app.dependency_overrides[get_db] = lambda: mock_db
        app.dependency_overrides[get_current_user] = lambda: mock_admin # Or mock_user, endpoint allows both
        response = client.get("/api/v1/mcp/configs")
        app.dependency_overrides = {}
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        mock_get_configs.assert_called_once_with(mock_db, mock_admin.id)

    @patch("app.api.routes.mcp.MCPConfigService.get_config_by_id")
    def test_get_mcp_config(self, mock_get_config, mock_db):
        mock_config_id = "mcp-config-id"
        mock_user_id = "user1"
        # Mock service return value with Pydantic model
        mock_get_config.return_value = MCPServerConfigResponse(
            id=mock_config_id, name="test-config", command="docker", args=["run"],
            enabled=True, user_id=mock_user_id, container_id=None,
            created_at=datetime.now(timezone.utc), updated_at=datetime.now(timezone.utc)
        )
        app.dependency_overrides[get_db] = lambda: mock_db
        app.dependency_overrides[get_current_user] = lambda: mock_admin # Or mock_user if they own it
        response = client.get(f"/api/v1/mcp/configs/{mock_config_id}")
        app.dependency_overrides = {}
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == mock_config_id
        mock_get_config.assert_called_once_with(mock_db, mock_config_id)
        call_args, _ = mock_get_config.call_args
        assert call_args[0] is mock_db

    @patch("app.api.routes.mcp.MCPConfigService.update_config")
    @patch("app.api.routes.mcp.MCPConfigService.get_config_by_id")
    def test_update_mcp_config(self, mock_get_config, mock_update_config, mock_db):
        mock_config_id = "mcp-config-id"
        mock_user_id = "user1"
        mock_get_config.return_value = MagicMock(id=mock_config_id, user_id=mock_user_id)
        # Mock update_config return value with Pydantic model
        mock_update_config.return_value = MCPServerConfigResponse(
            id=mock_config_id, name="updated-config", command="docker", args=["new"],
            enabled=False, user_id=mock_user_id, container_id=None,
            created_at=datetime.now(timezone.utc), updated_at=datetime.now(timezone.utc)
        )
        request_data = {"name": "updated-config", "args": ["new"], "enabled": False}
        app.dependency_overrides[get_db] = lambda: mock_db
        app.dependency_overrides[get_current_user] = lambda: mock_admin
        response = client.put(f"/api/v1/mcp/configs/{mock_config_id}", json=request_data)
        app.dependency_overrides = {}
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "updated-config"
        mock_get_config.assert_called_once_with(mock_db, mock_config_id)
        mock_update_config.assert_called_once()
        call_args, _ = mock_update_config.call_args
        assert call_args[0] is mock_db

    @patch("app.api.routes.mcp.MCPConfigService.delete_config")
    @patch("app.api.routes.mcp.MCPConfigService.get_config_by_id")
    @patch("app.api.routes.mcp.MCPConfigService.stop_server") # stop_server is called by delete route
    def test_delete_mcp_config(self, mock_stop_server, mock_get_config, mock_delete_config, mock_db):
        mock_config_id = "mcp-config-id"
        mock_user_id = "user1"
        mock_get_config.return_value = MagicMock(id=mock_config_id, user_id=mock_user_id)
        mock_delete_config.return_value = True
        app.dependency_overrides[get_db] = lambda: mock_db
        app.dependency_overrides[get_current_user] = lambda: mock_admin
        response = client.delete(f"/api/v1/mcp/configs/{mock_config_id}")
        app.dependency_overrides = {}
        assert response.status_code == 204
        mock_get_config.assert_called_once_with(mock_db, mock_config_id)
        mock_delete_config.assert_called_once_with(mock_db, mock_config_id)
        mock_stop_server.assert_called_once_with(mock_db, mock_config_id)
        call_args_get, _ = mock_get_config.call_args
        assert call_args_get[0] is mock_db
        call_args_delete, _ = mock_delete_config.call_args
        assert call_args_delete[0] is mock_db

    @patch("app.api.routes.mcp.MCPConfigService.get_config_status")
    @patch("app.api.routes.mcp.MCPConfigService.get_config_by_id")
    def test_get_mcp_config_status(self, mock_get_config, mock_get_status, mock_db):
        mock_config_id = "mcp-fs-12345"
        mock_user_id = "user1"
        mock_get_config.return_value = MagicMock(id=mock_config_id, user_id=mock_user_id)
        # Mock get_config_status return value with Pydantic model
        mock_get_status.return_value = MCPServerStatus(
            id=mock_config_id, name="filesystem", enabled=True, status="running",
            container_id="container1", error_message=None
        )
        app.dependency_overrides[get_db] = lambda: mock_db
        app.dependency_overrides[get_current_user] = lambda: mock_admin # Or mock_user if they own it
        response = client.get(f"/api/v1/mcp/configs/{mock_config_id}/status")
        app.dependency_overrides = {}
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "running"
        mock_get_config.assert_called_once_with(mock_db, mock_config_id)
        mock_get_status.assert_called_once_with(mock_db, mock_config_id)
        call_args_get, _ = mock_get_config.call_args
        assert call_args_get[0] is mock_db
        call_args_status, _ = mock_get_status.call_args
        assert call_args_status[0] is mock_db

    @patch("app.api.routes.mcp.MCPConfigService.start_server")
    @patch("app.api.routes.mcp.MCPConfigService.get_config_by_id")
    def test_start_mcp_server(self, mock_get_config, mock_start_server, mock_db):
        mock_config_id = "mcp-fs-12345"
        mock_user_id = "user1"
        mock_get_config.return_value = MagicMock(id=mock_config_id, user_id=mock_user_id, enabled=True)
        # Mock start_server return value with Pydantic model
        mock_start_server.return_value = MCPServerStatus(
            id=mock_config_id, name="filesystem", enabled=True, status="running",
            container_id="container1", error_message=None
        )
        app.dependency_overrides[get_db] = lambda: mock_db
        app.dependency_overrides[get_current_user] = lambda: mock_admin
        response = client.post(f"/api/v1/mcp/configs/{mock_config_id}/start")
        app.dependency_overrides = {}
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "running"
        mock_get_config.assert_called_once_with(mock_db, mock_config_id)
        mock_start_server.assert_called_once_with(mock_db, mock_config_id)
        call_args_get, _ = mock_get_config.call_args
        assert call_args_get[0] is mock_db
        call_args_start, _ = mock_start_server.call_args
        assert call_args_start[0] is mock_db

    @patch("app.api.routes.mcp.MCPConfigService.stop_server")
    @patch("app.api.routes.mcp.MCPConfigService.get_config_by_id")
    def test_stop_mcp_server(self, mock_get_config, mock_stop_server, mock_db):
        mock_config_id = "mcp-fs-12345"
        mock_user_id = "user1"
        mock_get_config.return_value = MagicMock(id=mock_config_id, user_id=mock_user_id)
        # Mock stop_server return value with Pydantic model
        mock_stop_server.return_value = MCPServerStatus(
            id=mock_config_id, name="filesystem", enabled=True, status="exited",
            container_id=None, error_message=None
        )
        app.dependency_overrides[get_db] = lambda: mock_db
        app.dependency_overrides[get_current_user] = lambda: mock_admin
        response = client.post(f"/api/v1/mcp/configs/{mock_config_id}/stop")
        app.dependency_overrides = {}
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "exited"
        mock_get_config.assert_called_once_with(mock_db, mock_config_id)
        mock_stop_server.assert_called_once_with(mock_db, mock_config_id)
        call_args_get, _ = mock_get_config.call_args
        assert call_args_get[0] is mock_db
        call_args_stop, _ = mock_stop_server.call_args
        assert call_args_stop[0] is mock_db

    @patch("app.api.routes.mcp.MCPConfigService.restart_server")
    @patch("app.api.routes.mcp.MCPConfigService.get_config_by_id")
    def test_restart_mcp_server(self, mock_get_config, mock_restart_server, mock_db):
        mock_config_id = "mcp-fs-12345"
        mock_user_id = "user1"
        mock_get_config.return_value = MagicMock(id=mock_config_id, user_id=mock_user_id)
        # Mock restart_server return value with Pydantic model
        mock_restart_server.return_value = MCPServerStatus(
            id=mock_config_id, name="filesystem", enabled=True, status="running",
            container_id="container1", error_message=None
        )
        app.dependency_overrides[get_db] = lambda: mock_db
        app.dependency_overrides[get_current_user] = lambda: mock_admin
        response = client.post(f"/api/v1/mcp/configs/{mock_config_id}/restart")
        app.dependency_overrides = {}
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "running"
        mock_get_config.assert_called_once_with(mock_db, mock_config_id)
        mock_restart_server.assert_called_once_with(mock_db, mock_config_id)
        call_args_get, _ = mock_get_config.call_args
        assert call_args_get[0] is mock_db
        call_args_restart, _ = mock_restart_server.call_args
        assert call_args_restart[0] is mock_db

    @patch("app.api.routes.mcp.MCPConfigService.generate_mcp_config_json")
    def test_get_mcp_config_json(self, mock_generate_json, mock_db):
        mock_user_id = "user1"
        mock_generate_json.return_value = sample_mcp_config_json
        app.dependency_overrides[get_db] = lambda: mock_db
        app.dependency_overrides[get_current_user] = lambda: mock_admin # Or mock_user
        response = client.get("/api/v1/mcp/configs/export/json")
        app.dependency_overrides = {}
        assert response.status_code == 200 # Should now pass
        data = response.json()
        assert "mcpServers" in data
        mock_generate_json.assert_called_once_with(mock_db, mock_admin.id)
        call_args, _ = mock_generate_json.call_args
        assert call_args[0] is mock_db

    def test_unauthorized_access(self, mock_db):
        app.dependency_overrides[get_db] = lambda: mock_db
        app.dependency_overrides[get_current_user] = lambda: mock_user # Use regular user

        # Test create endpoint (requires admin) - send valid body
        response_post = client.post(
            "/api/v1/mcp/configs",
            json={"name": "test", "command": "docker", "args": ["run"]} # Minimal valid body
        )
        assert response_post.status_code == 403 # Expect Forbidden

        # Mock get_config_by_id for subsequent checks
        mock_config_id = "mcp-config-id"
        with patch("app.api.routes.mcp.MCPConfigService.get_config_by_id") as mock_get_config:
             mock_get_config.return_value = MagicMock(id=mock_config_id, user_id=mock_user.id) # Belongs to user

             response_put = client.put(f"/api/v1/mcp/configs/{mock_config_id}", json={"name": "new"})
             assert response_put.status_code == 403

             response_delete = client.delete(f"/api/v1/mcp/configs/{mock_config_id}")
             assert response_delete.status_code == 403

             response_start = client.post(f"/api/v1/mcp/configs/{mock_config_id}/start")
             assert response_start.status_code == 403

             response_stop = client.post(f"/api/v1/mcp/configs/{mock_config_id}/stop")
             assert response_stop.status_code == 403

             response_restart = client.post(f"/api/v1/mcp/configs/{mock_config_id}/restart")
