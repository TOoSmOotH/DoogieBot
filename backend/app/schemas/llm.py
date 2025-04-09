from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime

class LLMConfigBase(BaseModel):
    """Base schema for LLM configuration."""
    chat_provider: str
    embedding_provider: str
    model: str
    embedding_model: str
    system_prompt: str = Field(..., description="Global system prompt used for all LLM providers")
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: Optional[float] = Field(0.7, description="LLM temperature setting (0.0-2.0). Default is 0.7")
    config: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Generic config dictionary for provider-specific settings like rag_top_k, use_reranking")
    reranked_top_n: Optional[int] = Field(None, description="Number of documents to send to LLM after reranking. Defaults to rag_top_k if not set.")

class LLMConfigCreate(LLMConfigBase):
    """Schema for creating a new LLM configuration."""
    pass

class LLMConfigUpdate(BaseModel):
    """Schema for updating an LLM configuration."""
    chat_provider: Optional[str] = None
    embedding_provider: Optional[str] = None
    model: Optional[str] = None
    embedding_model: Optional[str] = None
    system_prompt: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    is_active: Optional[bool] = None
    temperature: Optional[float] = None
    config: Optional[Dict[str, Any]] = None
    reranked_top_n: Optional[int] = None

class LLMConfigInDB(LLMConfigBase):
    """Schema for LLM configuration in the database."""
    id: str
    is_active: bool
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)

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