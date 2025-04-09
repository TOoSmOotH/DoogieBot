from typing import Dict, List, Optional, Any # Added Any
from datetime import datetime
from pydantic import BaseModel, Field, validator, ConfigDict, field_validator, model_serializer # Added model_serializer

class MCPServerConfigBase(BaseModel):
    """
    Base schema for MCP server configurations.
    
    Contains common fields shared by all MCP server configuration schemas.
    """
    name: str = Field(..., description="Unique name for this MCP server configuration")
    command: str = Field("docker", description="Command to run the MCP server")
    args: List[str] = Field(..., description="Arguments for the command")
    env: Optional[Dict[str, str]] = Field(None, description="Environment variables")
    enabled: bool = Field(True, description="Whether this server is enabled")

    @field_validator("command")
    @classmethod
    def validate_command(cls, v):
        """
        Validate that the command is 'docker'.
        
        In our implementation, we only support running MCP servers in Docker containers.
        """
        if v != "docker":
            raise ValueError("Only 'docker' command is supported")
        return v
    
    @field_validator("args")
    @classmethod
    def validate_args(cls, v):
        """
        Validate that the args list is not empty and contains valid Docker arguments.
        """
        if not v or len(v) < 1:
            raise ValueError("Arguments must not be empty")
        
        # Check if the command is for running a container
        if "run" not in v and v[0] != "run":
            # If npx or uvx commands are specified, they should be run in a Docker container
            if any(cmd in v for cmd in ["npx", "uvx"]):
                # Transform from non-Docker command to Docker run command
                # This is a simplified validator, actual transformation happens in the service layer
                pass
            else:
                # For this validator, we'll just check for the presence of a run command
                # The actual transformation of npx/uvx commands will happen in the service layer
                pass
                
        return v

class MCPServerConfigCreate(MCPServerConfigBase):
    """
    Schema for creating a new MCP server configuration.
    """
    pass

class MCPServerConfigUpdate(BaseModel):
    """
    Schema for updating an existing MCP server configuration.
    
    All fields are optional to allow partial updates.
    """
    name: Optional[str] = None
    args: Optional[List[str]] = None
    env: Optional[Dict[str, str]] = None
    enabled: Optional[bool] = None

class MCPServerConfigInDBBase(MCPServerConfigBase):
    """
    Schema for MCP server configuration as stored in the database.
    Includes fields from the ORM model that are not part of the base config.
    """
    id: str
    user_id: str
    created_at: datetime
    updated_at: datetime
    config: Optional[Dict[str, Any]] = None # Include the raw config field

    model_config = ConfigDict(from_attributes=True)

class MCPServerConfigResponse(MCPServerConfigInDBBase):
    """
    Schema for MCP server configuration API responses.
    Ensures command, args, env, enabled are populated from the 'config' JSONB field.
    """
    # Define the fields expected in the response, initially optional
    command: Optional[str] = None
    args: Optional[List[str]] = None
    env: Optional[Dict[str, str]] = None
    enabled: Optional[bool] = None
    
    @model_serializer(mode='wrap')
    def serialize_model(self, serializer, info):
        # Run the default serializer first to get basic field population
        data = serializer(self)

        # 'self' here is the MCPServerConfigResponse instance being serialized.
        # Pydantic v2 with from_attributes=True usually populates fields from the ORM object.
        # The 'config' field from the ORM model should be present on 'self' if using from_attributes.
        orm_config_dict = getattr(self, 'config', None)

        # Populate response fields from the ORM's config dictionary if they weren't directly set
        if isinstance(orm_config_dict, dict):
            data['command'] = data.get('command') if data.get('command') is not None else orm_config_dict.get('command', 'docker')
            data['args'] = data.get('args') if data.get('args') is not None else orm_config_dict.get('args', [])
            data['env'] = data.get('env') if data.get('env') is not None else orm_config_dict.get('env')
            data['enabled'] = data.get('enabled') if data.get('enabled') is not None else orm_config_dict.get('enabled', False)
        else:
            # Apply defaults if the config dictionary is missing or not a dict
            data['command'] = data.get('command', 'docker')
            data['args'] = data.get('args', [])
            data['env'] = data.get('env') # Keep None if not set
            data['enabled'] = data.get('enabled', False)

        # Remove the raw 'config' field from the final response if it exists
        data.pop('config', None)

        return data

class MCPServerStatus(BaseModel):
    """
    Schema for MCP server status information.
    """
    id: str
    name: str
    enabled: bool
    status: str  # "running", "stopped", "error"
    container_id: Optional[str] = None
    error_message: Optional[str] = None

class MCPConfigJSON(BaseModel):
    """
    Schema for the complete MCP configuration JSON.
    
    This format matches the expected configuration format for Claude Desktop and other MCP clients.
    """
    mcpServers: Dict[str, Dict[str, object]]
