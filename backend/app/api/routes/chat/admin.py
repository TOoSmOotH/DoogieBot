import logging
from typing import Any
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.base import get_db
from app.models.user import User
from app.utils.deps import get_current_admin_user
from app.schemas.chat import (
    PaginatedChatListResponse,
    MessageResponse,
    MessageUpdate,
    PaginatedMessageResponse,
)
from app.services.chat import ChatService

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/admin/chats/flagged", response_model=PaginatedChatListResponse)
async def get_flagged_chats(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_admin_user),
) -> Any:
    """
    Get paginated chats with negative feedback. Admin only.
    """
    chats, total = ChatService.get_flagged_chats(db, skip=skip, limit=limit)

    # Serialize chats to ensure all required fields are included
    return {
        "items": [
            {
                "id": chat.id,
                "user_id": chat.user_id,
                "title": chat.title,
                "created_at": chat.created_at,
                "updated_at": chat.updated_at,
                "messages": [
                    {
                        "id": msg.id,
                        "chat_id": msg.chat_id,
                        "role": msg.role,
                        "content": msg.content,
                        "created_at": msg.created_at,
                        "tokens": msg.tokens,
                        "tokens_per_second": msg.tokens_per_second,
                        "model": msg.model,
                        "provider": msg.provider,
                        "feedback": msg.feedback,
                        "feedback_text": msg.feedback_text,
                        "reviewed": msg.reviewed,
                        "context_documents": msg.context_documents if msg.context_documents is None else [str(doc_id) for doc_id in msg.context_documents]
                    }
                    for msg in chat.messages
                ] if chat.messages else None
            }
            for chat in chats
        ],
        "total": total,
        "page": skip // limit + 1 if limit > 0 else 1,
        "size": limit,
        "pages": (total + limit - 1) // limit if limit > 0 else 1
    }

@router.get("/admin/feedback", response_model=PaginatedMessageResponse) # Update response model
def read_feedback_messages(
    db: Session = Depends(get_db),
    feedback_type: str = None,
    reviewed: bool = None,
    skip: int = 0, # Add skip parameter
    limit: int = 100, # Add limit parameter
    current_user: User = Depends(get_current_admin_user),
) -> Any:
    """
    Get paginated messages with feedback. Admin only.
    """
    messages, total = ChatService.get_feedback_messages(
        db, feedback_type, reviewed, skip=skip, limit=limit # Pass skip and limit
    )

    # Construct paginated response
    page = skip // limit + 1 if limit > 0 else 1
    pages = (total + limit - 1) // limit if limit > 0 else 1

    # Let FastAPI handle serialization using the response_model and from_attributes=True
    return {
        "items": messages, # Return the list of SQLAlchemy Message objects directly
        "total": total,
        "page": page,
        "size": limit,
        "pages": pages
    }

@router.put("/admin/messages/{message_id}", response_model=MessageResponse)
def update_message(
    message_id: str,
    message_in: MessageUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_admin_user),
) -> Any:
    """
    Update a message. Admin only.
    """
    if message_in.reviewed is not None:
        message = ChatService.mark_as_reviewed(db, message_id)
        if not message:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Message not found",
            )
        return message

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="No valid update fields provided",
    )