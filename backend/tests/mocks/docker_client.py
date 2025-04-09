"""
Mock Docker client for testing the MCP implementation.

This module provides a mock Docker client that can be used for testing Docker-related
functionality without requiring an actual Docker installation.
"""

from typing import Dict, List, Optional, Any
from unittest.mock import MagicMock
from .docker_api_responses import (
    sample_container_list,
    sample_container_inspect,
    sample_image_list,
    sample_image_inspect,
    sample_volume_list,
    sample_volume_inspect,
    sample_network_list,
    sample_network_inspect,
)


class MockContainer:
    """
    Mock class for a Docker container.
    """
    
    def __init__(self, container_data: Dict[str, Any]):
        """
        Initialize a mock container with sample data.
        
        Args:
            container_data: Dictionary with container attributes
        """
        self.id = container_data.get("Id", "mock_container_id")
        self.name = container_data.get("Names", ["/mock_container"])[0].lstrip("/")
        self.status = container_data.get("State", {}).get("Status", "running")
        self.image = container_data.get("Image", "mock_image")
        self.labels = container_data.get("Labels", {})
        self.attrs = container_data
        
        # Mock methods
        self.start = MagicMock(return_value=None)
        self.stop = MagicMock(return_value=None)
        self.restart = MagicMock(return_value=None)
        self.remove = MagicMock(return_value=None)
        self.logs = MagicMock(return_value=b"Mock logs")
        self.exec_run = MagicMock(return_value=(0, b"Mock exec output"))


class MockImage:
    """
    Mock class for a Docker image.
    """
    
    def __init__(self, image_data: Dict[str, Any]):
        """
        Initialize a mock image with sample data.
        
        Args:
            image_data: Dictionary with image attributes
        """
        self.id = image_data.get("Id", "mock_image_id")
        self.tags = image_data.get("RepoTags", ["mock_image:latest"])
        self.attrs = image_data
        
        # Mock methods
        self.tag = MagicMock(return_value=True)
        self.remove = MagicMock(return_value=None)


class MockVolume:
    """
    Mock class for a Docker volume.
    """
    
    def __init__(self, volume_data: Dict[str, Any]):
        """
        Initialize a mock volume with sample data.
        
        Args:
            volume_data: Dictionary with volume attributes
        """
        self.id = volume_data.get("Name", "mock_volume")
        self.name = volume_data.get("Name", "mock_volume")
        self.attrs = volume_data
        
        # Mock methods
        self.remove = MagicMock(return_value=None)


class MockNetwork:
    """
    Mock class for a Docker network.
    """
    
    def __init__(self, network_data: Dict[str, Any]):
        """
        Initialize a mock network with sample data.
        
        Args:
            network_data: Dictionary with network attributes
        """
        self.id = network_data.get("Id", "mock_network_id")
        self.name = network_data.get("Name", "mock_network")
        self.attrs = network_data
        
        # Mock methods
        self.remove = MagicMock(return_value=None)


class MockContainerCollection:
    """
    Mock class for Docker containers collection.
    """
    
    def __init__(self):
        """Initialize with sample container data."""
        self.containers = {
            container["Id"]: MockContainer(container)
            for container in sample_container_list
        }
    
    def get(self, container_id: str, **kwargs) -> MockContainer:
        """
        Get a container by ID.
        
        Args:
            container_id: The ID of the container to get
            
        Returns:
            A mock container
            
        Raises:
            NotFound: If the container ID is not found
        """
        if container_id not in self.containers:
            from docker.errors import NotFound
            raise NotFound(f"Container '{container_id}' not found")
        return self.containers[container_id]
    
    def list(self, all: bool = False, filters: Optional[Dict[str, Any]] = None) -> List[MockContainer]:
        """
        List containers.
        
        Args:
            all: Whether to include stopped containers
            filters: Filters to apply
            
        Returns:
            A list of mock containers
        """
        containers = list(self.containers.values())
        
        if filters:
            if "name" in filters:
                name_filter = filters["name"]
                containers = [c for c in containers if name_filter in c.name]
                
            if "status" in filters:
                status_filter = filters["status"]
                containers = [c for c in containers if c.status == status_filter]
        
        if not all:
            containers = [c for c in containers if c.status == "running"]
            
        return containers
    
    def run(
        self,
        image: str,
        command: Optional[str] = None,
        name: Optional[str] = None,
        environment: Optional[Dict[str, str]] = None,
        detach: bool = False,
        remove: bool = False,
        **kwargs
    ) -> MockContainer:
        """
        Run a container.
        
        Args:
            image: The image to run
            command: The command to run
            name: The name for the container
            environment: Environment variables
            detach: Whether to run in detached mode
            remove: Whether to remove the container when it exits
            **kwargs: Additional arguments
            
        Returns:
            A mock container
        """
        container_id = f"mock_container_{len(self.containers) + 1}"
        container_data = {
            "Id": container_id,
            "Names": [f"/{name or container_id}"],
            "Image": image,
            "State": {"Status": "running"},
            "Labels": {},
            "Command": command or "",
            "Env": [f"{k}={v}" for k, v in (environment or {}).items()]
        }
        
        mock_container = MockContainer(container_data)
        self.containers[container_id] = mock_container
        return mock_container


class MockImageCollection:
    """
    Mock class for Docker images collection.
    """
    
    def __init__(self):
        """Initialize with sample image data."""
        self.images = {
            image["Id"]: MockImage(image)
            for image in sample_image_list
        }
    
    def get(self, image_id: str) -> MockImage:
        """
        Get an image by ID.
        
        Args:
            image_id: The ID of the image to get
            
        Returns:
            A mock image
            
        Raises:
            NotFound: If the image ID is not found
        """
        if image_id not in self.images:
            from docker.errors import NotFound
            raise NotFound(f"Image '{image_id}' not found")
        return self.images[image_id]
    
    def list(self, name: Optional[str] = None, filters: Optional[Dict[str, Any]] = None) -> List[MockImage]:
        """
        List images.
        
        Args:
            name: Filter by name
            filters: Additional filters
            
        Returns:
            A list of mock images
        """
        images = list(self.images.values())
        
        if name:
            images = [i for i in images if any(name in tag for tag in i.tags)]
            
        return images
    
    def pull(self, repository: str, tag: Optional[str] = None, **kwargs) -> MockImage:
        """
        Pull an image.
        
        Args:
            repository: The repository to pull from
            tag: The tag to pull
            **kwargs: Additional arguments
            
        Returns:
            A mock image
        """
        image_tag = tag or "latest"
        image_id = f"sha256:{repository.replace('/', '_')}_{image_tag}"
        
        image_data = {
            "Id": image_id,
            "RepoTags": [f"{repository}:{image_tag}"],
            "RepoDigests": [f"{repository}@{image_id}"],
            "Created": "2023-01-01T00:00:00Z",
            "Size": 100000000,
            "VirtualSize": 100000000,
            "SharedSize": 0,
            "Labels": {},
            "Containers": 0
        }
        
        mock_image = MockImage(image_data)
        self.images[image_id] = mock_image
        return mock_image


class MockVolumeCollection:
    """
    Mock class for Docker volumes collection.
    """
    
    def __init__(self):
        """Initialize with sample volume data."""
        self.volumes = {
            volume["Name"]: MockVolume(volume)
            for volume in sample_volume_list
        }
    
    def get(self, volume_name: str) -> MockVolume:
        """
        Get a volume by name.
        
        Args:
            volume_name: The name of the volume to get
            
        Returns:
            A mock volume
            
        Raises:
            NotFound: If the volume name is not found
        """
        if volume_name not in self.volumes:
            from docker.errors import NotFound
            raise NotFound(f"Volume '{volume_name}' not found")
        return self.volumes[volume_name]
    
    def list(self, filters: Optional[Dict[str, Any]] = None) -> List[MockVolume]:
        """
        List volumes.
        
        Args:
            filters: Filters to apply
            
        Returns:
            A list of mock volumes
        """
        volumes = list(self.volumes.values())
        return volumes
    
    def create(self, name: Optional[str] = None, driver: Optional[str] = None, **kwargs) -> MockVolume:
        """
        Create a volume.
        
        Args:
            name: The name for the volume
            driver: The driver to use
            **kwargs: Additional arguments
            
        Returns:
            A mock volume
        """
        volume_name = name or f"mock_volume_{len(self.volumes) + 1}"
        
        volume_data = {
            "Name": volume_name,
            "Driver": driver or "local",
            "Mountpoint": f"/var/lib/docker/volumes/{volume_name}/_data",
            "Labels": kwargs.get("labels", {}),
            "Scope": "local",
            "Options": kwargs.get("driver_opts", {})
        }
        
        mock_volume = MockVolume(volume_data)
        self.volumes[volume_name] = mock_volume
        return mock_volume


class MockNetworkCollection:
    """
    Mock class for Docker networks collection.
    """
    
    def __init__(self):
        """Initialize with sample network data."""
        self.networks = {
            network["Id"]: MockNetwork(network)
            for network in sample_network_list
        }
    
    def get(self, network_id: str) -> MockNetwork:
        """
        Get a network by ID.
        
        Args:
            network_id: The ID of the network to get
            
        Returns:
            A mock network
            
        Raises:
            NotFound: If the network ID is not found
        """
        if network_id not in self.networks:
            from docker.errors import NotFound
            raise NotFound(f"Network '{network_id}' not found")
        return self.networks[network_id]
    
    def list(self, names: Optional[List[str]] = None, filters: Optional[Dict[str, Any]] = None) -> List[MockNetwork]:
        """
        List networks.
        
        Args:
            names: Filter by names
            filters: Additional filters
            
        Returns:
            A list of mock networks
        """
        networks = list(self.networks.values())
        
        if names:
            networks = [n for n in networks if n.name in names]
            
        return networks
    
    def create(self, name: str, driver: Optional[str] = None, **kwargs) -> MockNetwork:
        """
        Create a network.
        
        Args:
            name: The name for the network
            driver: The driver to use
            **kwargs: Additional arguments
            
        Returns:
            A mock network
        """
        network_id = f"mock_network_{len(self.networks) + 1}"
        
        network_data = {
            "Id": network_id,
            "Name": name,
            "Driver": driver or "bridge",
            "Scope": "local",
            "EnableIPv6": False,
            "IPAM": {
                "Driver": "default",
                "Options": {},
                "Config": []
            },
            "Internal": False,
            "Attachable": False,
            "Ingress": False,
            "ConfigFrom": {
                "Network": ""
            },
            "ConfigOnly": False,
            "Containers": {},
            "Options": {},
            "Labels": {}
        }
        
        mock_network = MockNetwork(network_data)
        self.networks[network_id] = mock_network
        return mock_network


class MockDockerClient:
    """
    Mock Docker client for testing.
    """
    
    def __init__(self):
        """Initialize the mock Docker client with collections."""
        self.containers = MockContainerCollection()
        self.images = MockImageCollection()
        self.volumes = MockVolumeCollection()
        self.networks = MockNetworkCollection()
        self.api = MagicMock()
        
        # Mock methods
        self.ping = MagicMock(return_value=True)
        self.version = MagicMock(return_value={"Version": "20.10.0", "ApiVersion": "1.41"})
        self.info = MagicMock(return_value={"Name": "mock_docker", "NCPU": 4, "MemTotal": 8589934592})
    
    def close(self):
        """Close the client (no-op in mock)."""
        pass


def mock_docker_from_env(**kwargs):
    """
    Mock for docker.from_env() function.

    Returns:
        A MockDockerClient instance
    """
    return MockDockerClient()
