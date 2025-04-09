from sqlalchemy import Column, String, DateTime, JSON, ForeignKey, Boolean, Integer
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid
import enum

from app.db.base import Base

class MCPServerStatus(str, enum.Enum):
    """Status of an MCP server"""
    CREATING = "creating"
    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"
    DELETED = "deleted"

class MCPServerType(str, enum.Enum):
    """Type of MCP server"""
    OLLAMA = "ollama"
    LMSTUDIO = "lmstudio"
    CUSTOM = "custom"

class MCPServerConfig(Base):
    """
    Model for storing MCP (Model Control Panel) server configurations.
    """
    __tablename__ = "mcp_server_configs"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False)
    description = Column(String, nullable=True)
    server_type = Column(String, nullable=False)  # enum: ollama, lmstudio, custom
    base_url = Column(String, nullable=True)
    api_key = Column(String, nullable=True)
    models = Column(JSON, nullable=True)  # List of models available on this server
    status = Column(String, default=MCPServerStatus.STOPPED)
    port = Column(Integer, nullable=True)
    container_id = Column(String, nullable=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    is_active = Column(Boolean, default=True)
    
    # Additional configuration stored as JSON
    config = Column(JSON, nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="mcp_configs")

    def __repr__(self):
        return f"<MCPServerConfig id={self.id}, name={self.name}, type={self.server_type}>"
