from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime

class EmbeddingConfigBase(BaseModel):
    """Base schema for embedding configuration."""
    provider: str
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    config: Optional[Dict[str, Any]] = None

class EmbeddingConfigCreate(EmbeddingConfigBase):
    """Schema for creating a new embedding configuration."""
    pass

class EmbeddingConfigUpdate(BaseModel):
    """Schema for updating an embedding configuration."""
    provider: Optional[str] = None
    model: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    is_active: Optional[bool] = None
    config: Optional[Dict[str, Any]] = None

class EmbeddingConfigInDB(EmbeddingConfigBase):
    """Schema for embedding configuration in the database."""
    id: str
    is_active: bool
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)

class EmbeddingConfigResponse(EmbeddingConfigInDB):
    """Schema for embedding configuration response."""
    pass