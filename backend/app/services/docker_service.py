"""
Docker Service.

This service provides methods for interacting with the Docker daemon.
It handles containers, images, volumes, and networks.
"""

import logging
from typing import Dict, List, Any, Optional, Union
import docker
from docker.errors import DockerException, APIError, NotFound as DockerNotFound
from docker.models.containers import Container
from docker.models.images import Image
from docker.models.volumes import Volume
from docker.models.networks import Network

from fastapi import HTTPException, status

# Configure logging
logger = logging.getLogger(__name__)

class DockerService:
    """Service for managing Docker resources."""
    
    @staticmethod
    def _get_docker_client():
        """Get a Docker client instance."""
        try:
            return docker.from_env(timeout=10)
        except DockerException as e:
            logger.error(f"Failed to initialize Docker client: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
                detail=f"Docker error: {str(e)}"
            )
    
    # --- Container Methods ---
    
    @staticmethod
    def list_containers() -> List[Dict[str, Any]]:
        """List all Docker containers."""
        try:
            client = DockerService._get_docker_client()
            containers = client.containers.list(all=True)
            
            result = []
            for container in containers:
                # Format container data similar to Docker API response
                container_data = {
                    "Id": container.id,
                    "Names": [f"/{name.lstrip('/')}" for name in container.attrs.get('Names', [container.name])],
                    "Image": container.image.tags[0] if container.image.tags else container.image.id,
                    "ImageID": container.image.id,
                    "Command": container.attrs.get('Command', ''),
                    "Created": container.attrs.get('Created', 0),
                    "State": container.attrs.get('State', {}),
                    "Status": container.status,
                    "Ports": container.attrs.get('Ports', []),
                    "Labels": container.labels,
                    "HostConfig": {
                        "NetworkMode": container.attrs.get('HostConfig', {}).get('NetworkMode', 'default')
                    },
                    "NetworkSettings": container.attrs.get('NetworkSettings', {}),
                    "Mounts": container.attrs.get('Mounts', [])
                }
                result.append(container_data)
            
            return result
        except DockerException as e:
            logger.error(f"Error listing containers: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to list containers: {str(e)}"
            )
    
    @staticmethod
    def get_container(container_id: str) -> Dict[str, Any]:
        """Get details of a specific container."""
        try:
            client = DockerService._get_docker_client()
            try:
                container = client.containers.get(container_id)
                # Include all container attributes
                return container.attrs
            except DockerNotFound:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Container {container_id} not found"
                )
        except DockerException as e:
            if isinstance(e, HTTPException):
                raise e
            logger.error(f"Error getting container {container_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get container details: {str(e)}"
            )
    
    @staticmethod
    def start_container(container_id: str) -> Dict[str, str]:
        """Start a Docker container."""
        try:
            client = DockerService._get_docker_client()
            try:
                container = client.containers.get(container_id)
                container.start()
                logger.info(f"Started container {container_id}")
                return {
                    "status": "success",
                    "message": f"Container {container_id} started successfully"
                }
            except DockerNotFound:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Container {container_id} not found"
                )
        except DockerException as e:
            if isinstance(e, HTTPException):
                raise e
            logger.error(f"Error starting container {container_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to start container: {str(e)}"
            )
    
    @staticmethod
    def stop_container(container_id: str) -> Dict[str, str]:
        """Stop a Docker container."""
        try:
            client = DockerService._get_docker_client()
            try:
                container = client.containers.get(container_id)
                container.stop(timeout=10)
                logger.info(f"Stopped container {container_id}")
                return {
                    "status": "success",
                    "message": f"Container {container_id} stopped successfully"
                }
            except DockerNotFound:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Container {container_id} not found"
                )
        except DockerException as e:
            if isinstance(e, HTTPException):
                raise e
            logger.error(f"Error stopping container {container_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to stop container: {str(e)}"
            )
    
    @staticmethod
    def restart_container(container_id: str) -> Dict[str, str]:
        """Restart a Docker container."""
        try:
            client = DockerService._get_docker_client()
            try:
                container = client.containers.get(container_id)
                container.restart(timeout=10)
                logger.info(f"Restarted container {container_id}")
                return {
                    "status": "success",
                    "message": f"Container {container_id} restarted successfully"
                }
            except DockerNotFound:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Container {container_id} not found"
                )
        except DockerException as e:
            if isinstance(e, HTTPException):
                raise e
            logger.error(f"Error restarting container {container_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to restart container: {str(e)}"
            )
    
    @staticmethod
    def remove_container(container_id: str) -> Dict[str, str]:
        """Remove a Docker container."""
        try:
            client = DockerService._get_docker_client()
            try:
                container = client.containers.get(container_id)
                container.remove(force=True)
                logger.info(f"Removed container {container_id}")
                return {
                    "status": "success",
                    "message": f"Container {container_id} removed successfully"
                }
            except DockerNotFound:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Container {container_id} not found"
                )
        except DockerException as e:
            if isinstance(e, HTTPException):
                raise e
            logger.error(f"Error removing container {container_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to remove container: {str(e)}"
            )
    
    # --- Image Methods ---
    
    @staticmethod
    def list_images() -> List[Dict[str, Any]]:
        """List all Docker images."""
        try:
            client = DockerService._get_docker_client()
            images = client.images.list(all=True)
            
            result = []
            for image in images:
                # Format image data similar to Docker API response
                image_data = {
                    "Id": image.id,
                    "RepoTags": image.tags,
                    "RepoDigests": image.attrs.get('RepoDigests', []),
                    "Created": image.attrs.get('Created', 0),
                    "Size": image.attrs.get('Size', 0),
                    "VirtualSize": image.attrs.get('VirtualSize', 0),
                    "SharedSize": image.attrs.get('SharedSize', 0),
                    "Labels": image.attrs.get('Labels', {}),
                    "Containers": image.attrs.get('Containers', 0)
                }
                result.append(image_data)
            
            return result
        except DockerException as e:
            logger.error(f"Error listing images: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to list images: {str(e)}"
            )
    
    @staticmethod
    def get_image(image_id: str) -> Dict[str, Any]:
        """Get details of a specific image."""
        try:
            client = DockerService._get_docker_client()
            try:
                # Handle both regular ID and sha256: prefix
                image_id_clean = image_id.replace('sha256:', '')
                # First try direct ID lookup
                try:
                    image = client.images.get(image_id)
                except DockerNotFound:
                    # If not found, try with sha256: prefix removed
                    image = client.images.get(image_id_clean)
                
                # Include all image attributes
                return image.attrs
            except DockerNotFound:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Image {image_id} not found"
                )
        except DockerException as e:
            if isinstance(e, HTTPException):
                raise e
            logger.error(f"Error getting image {image_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get image details: {str(e)}"
            )
    
    @staticmethod
    def pull_image(repository: str, tag: str = "latest") -> Dict[str, str]:
        """Pull a Docker image from a registry."""
        try:
            client = DockerService._get_docker_client()
            image_name = f"{repository}:{tag}"
            client.images.pull(repository, tag)
            logger.info(f"Pulled image {image_name}")
            return {
                "status": "success",
                "message": f"Image {image_name} pulled successfully"
            }
        except DockerException as e:
            logger.error(f"Error pulling image {repository}:{tag}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to pull image: {str(e)}"
            )
    
    @staticmethod
    def remove_image(image_id: str) -> Dict[str, str]:
        """Remove a Docker image."""
        try:
            client = DockerService._get_docker_client()
            try:
                client.images.remove(image_id, force=True)
                logger.info(f"Removed image {image_id}")
                return {
                    "status": "success",
                    "message": f"Image {image_id} removed successfully"
                }
            except DockerNotFound:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Image {image_id} not found"
                )
        except DockerException as e:
            if isinstance(e, HTTPException):
                raise e
            logger.error(f"Error removing image {image_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to remove image: {str(e)}"
            )
    
    # --- Volume Methods ---
    
    @staticmethod
    def list_volumes() -> List[Dict[str, Any]]:
        """List all Docker volumes."""
        try:
            client = DockerService._get_docker_client()
            volumes = client.volumes.list()
            
            result = []
            for volume in volumes:
                # Format volume data similar to Docker API response
                volume_data = {
                    "Name": volume.name,
                    "Driver": volume.attrs.get('Driver', 'local'),
                    "Mountpoint": volume.attrs.get('Mountpoint', ''),
                    "CreatedAt": volume.attrs.get('CreatedAt', ''),
                    "Labels": volume.attrs.get('Labels', {}),
                    "Options": volume.attrs.get('Options', {}),
                    "Scope": volume.attrs.get('Scope', 'local')
                }
                result.append(volume_data)
            
            return result
        except DockerException as e:
            logger.error(f"Error listing volumes: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to list volumes: {str(e)}"
            )
    
    @staticmethod
    def get_volume(volume_name: str) -> Dict[str, Any]:
        """Get details of a specific volume."""
        try:
            client = DockerService._get_docker_client()
            try:
                volume = client.volumes.get(volume_name)
                # Include all volume attributes
                return volume.attrs
            except DockerNotFound:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Volume {volume_name} not found"
                )
        except DockerException as e:
            if isinstance(e, HTTPException):
                raise e
            logger.error(f"Error getting volume {volume_name}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get volume details: {str(e)}"
            )
    
    @staticmethod
    def create_volume(name: str, driver: str = "local") -> Dict[str, Any]:
        """Create a Docker volume."""
        try:
            client = DockerService._get_docker_client()
            volume = client.volumes.create(name=name, driver=driver)
            logger.info(f"Created volume {name}")
            return volume.attrs
        except DockerException as e:
            logger.error(f"Error creating volume {name}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to create volume: {str(e)}"
            )
    
    @staticmethod
    def remove_volume(volume_name: str) -> Dict[str, str]:
        """Remove a Docker volume."""
        try:
            client = DockerService._get_docker_client()
            try:
                volume = client.volumes.get(volume_name)
                volume.remove(force=True)
                logger.info(f"Removed volume {volume_name}")
                return {
                    "status": "success",
                    "message": f"Volume {volume_name} removed successfully"
                }
            except DockerNotFound:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Volume {volume_name} not found"
                )
        except DockerException as e:
            if isinstance(e, HTTPException):
                raise e
            logger.error(f"Error removing volume {volume_name}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to remove volume: {str(e)}"
            )
    
    # --- Network Methods ---
    
    @staticmethod
    def list_networks() -> List[Dict[str, Any]]:
        """List all Docker networks."""
        try:
            client = DockerService._get_docker_client()
            networks = client.networks.list()
            
            result = []
            for network in networks:
                # Format network data similar to Docker API response
                network_data = {
                    "Id": network.id,
                    "Name": network.name,
                    "Created": network.attrs.get('Created', ''),
                    "Scope": network.attrs.get('Scope', 'local'),
                    "Driver": network.attrs.get('Driver', 'bridge'),
                    "EnableIPv6": network.attrs.get('EnableIPv6', False),
                    "IPAM": network.attrs.get('IPAM', {}),
                    "Internal": network.attrs.get('Internal', False),
                    "Attachable": network.attrs.get('Attachable', False),
                    "Ingress": network.attrs.get('Ingress', False),
                    "ConfigFrom": network.attrs.get('ConfigFrom', {'Network': ''}),
                    "ConfigOnly": network.attrs.get('ConfigOnly', False),
                    "Containers": network.attrs.get('Containers', {}),
                    "Options": network.attrs.get('Options', {}),
                    "Labels": network.attrs.get('Labels', {})
                }
                result.append(network_data)
            
            return result
        except DockerException as e:
            logger.error(f"Error listing networks: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to list networks: {str(e)}"
            )
    
    @staticmethod
    def get_network(network_id: str) -> Dict[str, Any]:
        """Get details of a specific network."""
        try:
            client = DockerService._get_docker_client()
            try:
                network = client.networks.get(network_id)
                # Include all network attributes
                return network.attrs
            except DockerNotFound:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Network {network_id} not found"
                )
        except DockerException as e:
            if isinstance(e, HTTPException):
                raise e
            logger.error(f"Error getting network {network_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get network details: {str(e)}"
            )
    
    @staticmethod
    def create_network(name: str, driver: str = "bridge") -> Dict[str, Any]:
        """Create a Docker network."""
        try:
            client = DockerService._get_docker_client()
            network = client.networks.create(name=name, driver=driver)
            logger.info(f"Created network {name}")
            return network.attrs
        except DockerException as e:
            logger.error(f"Error creating network {name}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to create network: {str(e)}"
            )
    
    @staticmethod
    def remove_network(network_id: str) -> Dict[str, str]:
        """Remove a Docker network."""
        try:
            client = DockerService._get_docker_client()
            try:
                network = client.networks.get(network_id)
                network.remove()
                logger.info(f"Removed network {network_id}")
                return {
                    "status": "success",
                    "message": f"Network {network_id} removed successfully"
                }
            except DockerNotFound:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Network {network_id} not found"
                )
        except DockerException as e:
            if isinstance(e, HTTPException):
                raise e
            logger.error(f"Error removing network {network_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to remove network: {str(e)}"
            )
