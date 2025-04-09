from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime

class RerankingConfigBase(BaseModel):
    """Base schema for reranking configuration."""
    provider: str
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    config: Optional[Dict[str, Any]] = None

class RerankingConfigCreate(RerankingConfigBase):
    """Schema for creating a new reranking configuration."""
    pass

class RerankingConfigUpdate(BaseModel):
    """Schema for updating a reranking configuration."""
    provider: Optional[str] = None
    model: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    is_active: Optional[bool] = None
    config: Optional[Dict[str, Any]] = None

class RerankingConfigInDB(RerankingConfigBase):
    """Schema for reranking configuration in the database."""
    id: str
    is_active: bool
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)

class RerankingConfigResponse(RerankingConfigInDB):
    """Schema for reranking configuration response."""
    pass