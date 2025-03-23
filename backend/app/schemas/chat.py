from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


# Message schemas
class MessageBase(BaseModel):
    role: str
    content: str


class MessageCreate(MessageBase):
    pass


class MessageUpdate(BaseModel):
    feedback: Optional[str] = None
    feedback_text: Optional[str] = None
    reviewed: Optional[bool] = None


class MessageResponse(MessageBase):
    id: str
    chat_id: str
    created_at: datetime
    tokens: Optional[int] = None
    tokens_per_second: Optional[float] = None
    model: Optional[str] = None
    provider: Optional[str] = None
    feedback: Optional[str] = None
    feedback_text: Optional[str] = None
    reviewed: Optional[bool] = False
    context_documents: Optional[List[str]] = None

    class Config:
        from_attributes = True


# Chat schemas
class ChatBase(BaseModel):
    title: Optional[str] = "New Chat"


class ChatCreate(ChatBase):
    pass


class ChatUpdate(BaseModel):
    title: Optional[str] = None


class ChatResponse(ChatBase):
    id: str
    user_id: str
    created_at: datetime
    updated_at: datetime
    messages: Optional[List[MessageResponse]] = None

    class Config:
        from_attributes = True


class ChatListResponse(ChatBase):
    id: str
    user_id: str
    created_at: datetime
    updated_at: datetime
    messages: Optional[List[MessageResponse]] = None

    class Config:
        from_attributes = True


# Paginated response schemas
class PaginatedChatListResponse(BaseModel):
    items: List[ChatListResponse]
    total: int
    page: int
    size: int
    pages: int


# Feedback schemas
class FeedbackCreate(BaseModel):
    feedback: str = Field(..., description="Feedback type: 'positive' or 'negative'")
    feedback_text: Optional[str] = Field(None, description="Additional feedback text")


# LLM response schemas
class LLMResponseMetadata(BaseModel):
    tokens: int
    tokens_per_second: float
    model: str
    provider: str


class StreamingResponse(BaseModel):
    content: str
    metadata: Optional[LLMResponseMetadata] = None
    done: bool = False
