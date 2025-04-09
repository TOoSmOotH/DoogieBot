from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, field_validator, ConfigDict
from datetime import datetime
import json # For validating tool_calls

# Import the enum from the model to ensure consistency
from app.models.chat import MessageRole, FeedbackType

# Message schemas
class MessageBase(BaseModel):
    role: str # Should use MessageRole enum values
    content: Optional[str] = None # Make content optional for tool calls/results

    @field_validator('role')
    @classmethod
    def validate_role(cls, value):
        if value not in MessageRole.__members__.values():
            raise ValueError(f"Invalid role: {value}. Must be one of {list(MessageRole.__members__.values())}")
        return value

class MessageCreate(MessageBase):
    # Fields specific to creating messages, potentially including tool info
    tool_calls: Optional[List[Dict[str, Any]]] = None # For assistant message requesting calls
    tool_call_id: Optional[str] = None # For tool message providing result
    name: Optional[str] = None # For tool message, function name

    # Ensure either content or tool_calls is present for assistant messages
    # Ensure content is present for user/system messages
    # Ensure content and tool_call_id/name are present for tool messages
    # (Validation can be added here or in the service layer)

class MessageUpdate(BaseModel):
    feedback: Optional[str] = None
    feedback_text: Optional[str] = None
    reviewed: Optional[bool] = None

    @field_validator('feedback')
    @classmethod
    def validate_feedback(cls, value):
        if value is not None and value not in FeedbackType.__members__.values():
            raise ValueError(f"Invalid feedback type: {value}. Must be one of {list(FeedbackType.__members__.values())}")
        return value


class MessageResponse(MessageBase):
    id: str
    chat_id: str
    created_at: datetime

    # LLM Metadata
    tokens: Optional[int] = None # Total tokens for the LLM call generating this message
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    tokens_per_second: Optional[float] = None
    model: Optional[str] = None
    provider: Optional[str] = None
    finish_reason: Optional[str] = None # Reason LLM stopped generating

    # Tool Call / Result Data
    tool_calls: Optional[List[Dict[str, Any]]] = None # Assistant's request to call tools
    tool_call_id: Optional[str] = None # ID linking a tool result back to a tool call
    name: Optional[str] = None # Name of the function called (for tool role messages)

    # Feedback
    feedback: Optional[str] = None # FeedbackType enum values
    feedback_text: Optional[str] = None
    reviewed: Optional[bool] = False

    # RAG metadata
    context_documents: Optional[List[str]] = None
    related_question_content: Optional[str] = None # Added field for related question content

    @field_validator('context_documents', mode='before')
    @classmethod
    def validate_context_documents(cls, v: Any) -> Optional[List[str]]:
        """Ensure context_documents is a list of strings or None."""
        if v is None: return None
        if isinstance(v, list): return [str(item) for item in v if item is not None]
        return [] # Fallback

    @field_validator('tool_calls', mode='before')
    @classmethod
    def validate_tool_calls(cls, v: Any) -> Optional[List[Dict[str, Any]]]:
        """Ensure tool_calls is a list of dicts or None."""
        if v is None: return None
        if isinstance(v, str): # Handle case where it might be stored as JSON string
            try: v = json.loads(v)
            except json.JSONDecodeError: return [] # Invalid JSON
        if isinstance(v, list) and all(isinstance(item, dict) for item in v):
            return v
        logger.warning(f"Invalid tool_calls format received: {type(v)}. Returning empty list.")
        return [] # Fallback for invalid format

    model_config = ConfigDict(from_attributes=True)

# Chat schemas
class ChatBase(BaseModel):
    title: Optional[str] = "New Chat"

class ChatCreate(ChatBase):
    pass

class ChatUpdate(BaseModel):
    title: Optional[str] = None
    tags: Optional[List[str]] = None

class ChatResponse(ChatBase):
    id: str
    user_id: str
    created_at: datetime
    updated_at: datetime
    messages: Optional[List[MessageResponse]] = None # Ensure this uses the updated MessageResponse

    model_config = ConfigDict(from_attributes=True)

class ChatListResponse(ChatBase):
    id: str
    user_id: str
    created_at: datetime
    updated_at: datetime
    # Optionally include last message preview or message count
    # messages: Optional[List[MessageResponse]] = None # Might be too heavy for list view

    model_config = ConfigDict(from_attributes=True)

# Paginated response schemas
class PaginatedChatListResponse(BaseModel):
    items: List[ChatListResponse]
    total: int
    page: int
    size: int
    pages: int

class PaginatedMessageResponse(BaseModel):
    items: List[MessageResponse] # Ensure this uses the updated MessageResponse
    total: int
    page: int
    size: int
    pages: int

# Feedback schemas
class FeedbackCreate(BaseModel):
    feedback: str = Field(..., description="Feedback type: 'positive' or 'negative'")
    feedback_text: Optional[str] = Field(None, description="Additional feedback text")

    @field_validator('feedback')
    @classmethod
    def validate_feedback_type(cls, value):
        if value not in FeedbackType.__members__.values():
            raise ValueError(f"Invalid feedback type: {value}. Must be one of {list(FeedbackType.__members__.values())}")
        return value

# LLM response schemas (Used internally by services, maybe not exposed via API directly)
class LLMResponseMetadata(BaseModel):
    tokens: Optional[int] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    tokens_per_second: Optional[float] = None
    model: Optional[str] = None
    provider: Optional[str] = None
    finish_reason: Optional[str] = None

# Streaming response schema (for SSE events) - Needs careful design
class StreamingResponseChunk(BaseModel):
    type: str # e.g., 'start', 'delta', 'final', 'error', 'tool_start', 'tool_delta', 'tool_stop'
    content: Optional[str] = None # For text deltas
    tool_calls_delta: Optional[List[Dict[str, Any]]] = None # For tool call deltas
    tool_calls: Optional[List[Dict[str, Any]]] = None # For final tool calls
    usage: Optional[Dict[str, int]] = None # For start/final chunks
    tokens_per_second: Optional[float] = None # For final chunk
    finish_reason: Optional[str] = None # For final chunk
    error: Optional[str] = None # For error chunk
    model: Optional[str] = None # For start/final chunks
    provider: Optional[str] = None # For start/final chunks
    done: bool = False # True only for final/error chunks

# This schema might not be directly used as an API response_model
# but serves as a reference for the structure yielded by stream_llm_response
# and potentially consumed by the frontend.

# Tool retry request schema
class ToolRetryRequest(BaseModel):
    tool_call_id: str = Field(..., description="ID of the tool call to retry")
    function_name: str = Field(..., description="Name of the function to call")
    arguments: str = Field(..., description="Arguments for the function call as a JSON string")
