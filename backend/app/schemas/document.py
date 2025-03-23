from typing import Any, Dict, Generic, List, Optional, TypeVar, Union

T = TypeVar("T")
from datetime import datetime

from pydantic import BaseModel, Field, validator


# Document schemas
class DocumentBase(BaseModel):
    title: str


class DocumentCreate(DocumentBase):
    pass


class ManualDocumentCreate(DocumentBase):
    content: str
    meta_data: Optional[Dict[str, Any]] = None


class DocumentUpdate(BaseModel):
    title: Optional[str] = None
    meta_data: Optional[Dict[str, Any]] = None


class DocumentResponse(DocumentBase):
    id: str
    filename: Optional[str] = None
    type: str
    uploaded_by: str
    created_at: datetime
    updated_at: datetime
    meta_data: Optional[Dict[str, Any]] = None

    class Config:
        from_attributes = True


class DocumentDetailResponse(DocumentResponse):
    content: Optional[str] = None
    chunks: Optional[List["DocumentChunkResponse"]] = None

    class Config:
        from_attributes = True


# Document Chunk schemas
class DocumentChunkBase(BaseModel):
    content: str
    chunk_index: int
    meta_data: Optional[Dict[str, Any]] = None


class DocumentChunkCreate(DocumentChunkBase):
    document_id: str
    embedding: Optional[List[float]] = None


class DocumentChunkResponse(DocumentChunkBase):
    id: str
    document_id: str
    created_at: datetime
    embedding: Optional[List[float]] = None

    class Config:
        from_attributes = True

    @validator("embedding", pre=True)
    def parse_embedding(cls, v):
        if isinstance(v, str):
            try:
                import json

                return json.loads(v)
            except:
                return None
        return v


# Processing schemas
class ProcessingStatus(BaseModel):
    status: str
    message: str
    document_id: Optional[str] = None
    total_chunks: Optional[int] = None


class ChunkingConfig(BaseModel):
    chunk_size: int = Field(1000, description="Size of each chunk in characters")
    chunk_overlap: int = Field(200, description="Overlap between chunks in characters")


class EmbeddingConfig(BaseModel):
    model: str = Field(..., description="Embedding model to use")
    provider: str = Field(..., description="LLM provider for embeddings")


class ProcessingConfig(BaseModel):
    chunking: ChunkingConfig
    embedding: Optional[EmbeddingConfig] = None
    rebuild_index: bool = Field(
        True, description="Whether to rebuild the index after processing"
    )


# Pagination schemas
class PaginatedResponse(BaseModel, Generic[T]):
    items: List[T]
    total: int
    page: int
    size: int
    pages: int


# Update forward references
DocumentDetailResponse.update_forward_refs()
