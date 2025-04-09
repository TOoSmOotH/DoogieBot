"""
Docker API routes.

These routes allow interaction with the Docker daemon
for managing containers, images, etc.
Requires admin privileges.
"""
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

# Import DockerService to handle the logic
from app.services.docker_service import DockerService
from app.utils.deps import get_current_admin_user
from app.models.user import User # Import User model

router = APIRouter()

# --- Placeholder Schemas (adjust as needed) ---
class DockerPullImageRequest(BaseModel):
    repository: str
    tag: Optional[str] = "latest"

class DockerCreateVolumeRequest(BaseModel):
    name: str
    driver: Optional[str] = "local"

class DockerCreateNetworkRequest(BaseModel):
    name: str
    driver: Optional[str] = "bridge"

class DockerActionResponse(BaseModel):
    status: str
    message: Optional[str] = None

# --- Container Routes ---

@router.get("/containers", response_model=List[Dict[str, Any]], tags=["docker"])
async def list_containers(current_user: User = Depends(get_current_admin_user)):
    """List Docker containers."""
    return DockerService.list_containers()

@router.get("/containers/{container_id}", response_model=Dict[str, Any], tags=["docker"])
async def get_container(container_id: str, current_user: User = Depends(get_current_admin_user)):
    """Get details for a specific container."""
    return DockerService.get_container(container_id)


@router.post("/containers/{container_id}/start", response_model=DockerActionResponse, tags=["docker"])
async def start_container(container_id: str, current_user: User = Depends(get_current_admin_user)):
    """Start a specific container."""
    return DockerService.start_container(container_id)

@router.post("/containers/{container_id}/stop", response_model=DockerActionResponse, tags=["docker"])
async def stop_container(container_id: str, current_user: User = Depends(get_current_admin_user)):
    """Stop a specific container."""
    return DockerService.stop_container(container_id)

@router.post("/containers/{container_id}/restart", response_model=DockerActionResponse, tags=["docker"])
async def restart_container(container_id: str, current_user: User = Depends(get_current_admin_user)):
    """Restart a specific container."""
    return DockerService.restart_container(container_id)

@router.delete("/containers/{container_id}", response_model=DockerActionResponse, tags=["docker"])
async def remove_container(container_id: str, current_user: User = Depends(get_current_admin_user)):
    """Remove a specific container."""
    return DockerService.remove_container(container_id)

# --- Image Routes ---

@router.get("/images", response_model=List[Dict[str, Any]], tags=["docker"])
async def list_images(current_user: User = Depends(get_current_admin_user)):
    """List Docker images."""
    return DockerService.list_images()

@router.get("/images/{image_id}", response_model=Dict[str, Any], tags=["docker"])
async def get_image(image_id: str, current_user: User = Depends(get_current_admin_user)):
    """Get details for a specific image."""
    return DockerService.get_image(image_id)

@router.post("/images/pull", response_model=DockerActionResponse, tags=["docker"])
async def pull_image(request: DockerPullImageRequest, current_user: User = Depends(get_current_admin_user)):
    """Pull a Docker image from a registry."""
    return DockerService.pull_image(request.repository, request.tag)

@router.delete("/images/{image_id}", response_model=DockerActionResponse, tags=["docker"])
async def remove_image(image_id: str, current_user: User = Depends(get_current_admin_user)):
    """Remove a specific image."""
    return DockerService.remove_image(image_id)

# --- Volume Routes ---

@router.get("/volumes", response_model=List[Dict[str, Any]], tags=["docker"])
async def list_volumes(current_user: User = Depends(get_current_admin_user)):
    """List Docker volumes."""
    return DockerService.list_volumes()

@router.get("/volumes/{volume_name}", response_model=Dict[str, Any], tags=["docker"])
async def get_volume(volume_name: str, current_user: User = Depends(get_current_admin_user)):
    """Get details for a specific volume."""
    return DockerService.get_volume(volume_name)

@router.post("/volumes", response_model=Dict[str, Any], tags=["docker"])
async def create_volume(request: DockerCreateVolumeRequest, current_user: User = Depends(get_current_admin_user)):
    """Create a Docker volume."""
    return DockerService.create_volume(request.name, request.driver)

@router.delete("/volumes/{volume_name}", response_model=DockerActionResponse, tags=["docker"])
async def remove_volume(volume_name: str, current_user: User = Depends(get_current_admin_user)):
    """Remove a specific volume."""
    return DockerService.remove_volume(volume_name)

# --- Network Routes ---

@router.get("/networks", response_model=List[Dict[str, Any]], tags=["docker"])
async def list_networks(current_user: User = Depends(get_current_admin_user)):
    """List Docker networks."""
    return DockerService.list_networks()

@router.get("/networks/{network_id}", response_model=Dict[str, Any], tags=["docker"])
async def get_network(network_id: str, current_user: User = Depends(get_current_admin_user)):
    """Get details for a specific network."""
    return DockerService.get_network(network_id)

@router.post("/networks", response_model=Dict[str, Any], tags=["docker"])
async def create_network(request: DockerCreateNetworkRequest, current_user: User = Depends(get_current_admin_user)):
    """Create a Docker network."""
    return DockerService.create_network(request.name, request.driver)

@router.delete("/networks/{network_id}", response_model=DockerActionResponse, tags=["docker"])
async def remove_network(network_id: str, current_user: User = Depends(get_current_admin_user)):
    """Remove a specific network."""
    return DockerService.remove_network(network_id)