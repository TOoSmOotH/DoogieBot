from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class LLMConfigBase(BaseModel):
    """Base schema for LLM configuration."""

    provider: str
    model: str
    embedding_model: str
    system_prompt: str = Field(
        ..., description="Global system prompt used for all LLM providers"
    )
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    config: Optional[Dict[str, Any]] = None


class LLMConfigCreate(LLMConfigBase):
    """Schema for creating a new LLM configuration."""

    pass


class LLMConfigUpdate(BaseModel):
    """Schema for updating an LLM configuration."""

    provider: Optional[str] = None
    model: Optional[str] = None
    embedding_model: Optional[str] = None
    system_prompt: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    is_active: Optional[bool] = None
    config: Optional[Dict[str, Any]] = None


class LLMConfigInDB(LLMConfigBase):
    """Schema for LLM configuration in the database."""

    id: str
    is_active: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class LLMConfigResponse(LLMConfigInDB):
    """Schema for LLM configuration response."""

    pass


class LLMProviderInfo(BaseModel):
    """Schema for LLM provider information."""

    id: str
    name: str
    available: bool
    requires_api_key: bool
    requires_base_url: bool
    default_model: str
    models: List[str] = []


class LLMProviderResponse(BaseModel):
    """Schema for LLM provider response."""

    providers: List[LLMProviderInfo]


class ModelsResponse(BaseModel):
    """Schema for available models response."""

    chat_models: List[str]
    embedding_models: List[str]
