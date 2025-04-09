#!/bin/bash
# Make this script executable with: chmod +x scripts/docker_setup.sh
set -e

# Docker setup script for Doogie Chat Bot MCP implementation

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print colored messages
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if script is run as root
if [ "$EUID" -ne 0 ]; then
    print_error "This script must be run as root or with sudo privileges"
    exit 1
fi

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker before running this script."
    exit 1
fi

print_info "Docker is installed."

# Create MCP data directory
MCP_DATA_DIR="/var/lib/doogie-chat/mcp"
if [ ! -d "$MCP_DATA_DIR" ]; then
    print_info "Creating MCP data directory at $MCP_DATA_DIR"
    mkdir -p "$MCP_DATA_DIR"
else
    print_info "MCP data directory already exists at $MCP_DATA_DIR"
fi

# Set appropriate permissions
print_info "Setting appropriate permissions for MCP data directory"
chmod 755 "$MCP_DATA_DIR"

# Create Docker network for MCP servers if it doesn't exist
NETWORK_NAME="mcp-network"
if ! docker network inspect "$NETWORK_NAME" &> /dev/null; then
    print_info "Creating Docker network: $NETWORK_NAME"
    docker network create "$NETWORK_NAME"
else
    print_info "Docker network '$NETWORK_NAME' already exists"
fi

# Pull MCP server images
print_info "Pulling MCP server images"
MCP_IMAGES=(
    "mcp/filesystem:latest"
    "mcp/git:latest"
    "mcp/github:latest"
    "mcp/postgres:latest"
)

for image in "${MCP_IMAGES[@]}"; do
    print_info "Pulling image: $image"
    if ! docker pull "$image" &> /dev/null; then
        print_warning "Failed to pull image: $image"
        print_warning "This is expected for custom or non-public images"
        print_warning "You may need to build these images locally"
    fi
done

# Configure Docker socket permissions
DOCKER_SOCKET="/var/run/docker.sock"
if [ -e "$DOCKER_SOCKET" ]; then
    print_info "Setting permissions for Docker socket"
    
    # Get group ID for the Docker group
    DOCKER_GID=$(stat -c '%g' "$DOCKER_SOCKET")
    
    # Set user and group for Docker socket
    chmod 660 "$DOCKER_SOCKET"
    
    print_info "Docker socket owner GID: $DOCKER_GID"
    print_info "Make sure the application runs with this group ID or has access to the Docker socket"
else
    print_error "Docker socket not found at $DOCKER_SOCKET"
    exit 1
fi

# Additional setup for Docker-in-Docker
print_info "Setting up Docker-in-Docker configuration"

# Create Docker configuration directory if it doesn't exist
DOCKER_CONFIG_DIR="/etc/docker"
if [ ! -d "$DOCKER_CONFIG_DIR" ]; then
    print_info "Creating Docker configuration directory at $DOCKER_CONFIG_DIR"
    mkdir -p "$DOCKER_CONFIG_DIR"
fi

# Configure Docker daemon to allow insecure registries if needed
# (for development environments only)
if [ "$1" == "--dev" ]; then
    print_warning "Configuring Docker for development environment"
    cat > "$DOCKER_CONFIG_DIR/daemon.json" <<EOF
{
    "insecure-registries": ["127.0.0.1:5000"],
    "log-driver": "json-file",
    "log-opts": {
        "max-size": "10m",
        "max-file": "3"
    }
}
EOF
fi

print_info "Docker setup completed successfully!"
print_info "You can now run MCP servers using Docker"

# Display additional information
print_info "Available MCP server images:"
docker images | grep "mcp/"

print_info "Docker network information:"
docker network inspect "$NETWORK_NAME"

exit 0