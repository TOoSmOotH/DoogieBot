from typing import Any, List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.base import get_db
from app.models.user import User
from app.utils.deps import get_current_user
from app.schemas.chat import (
    MessageCreate,
    MessageResponse,
    FeedbackCreate,
)
from app.services.chat import ChatService

router = APIRouter()

@router.post("/{chat_id}/messages", response_model=MessageResponse)
def add_message(
    chat_id: str,
    message_in: MessageCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Add a message to a chat.
    """
    chat = ChatService.get_chat(db, chat_id)
    if not chat:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat not found",
        )

    # Check if user owns the chat
    if chat.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions",
        )

    message = ChatService.add_message(
        db,
        chat_id,
        message_in.role,
        message_in.content
    )
    return message

@router.get("/{chat_id}/messages", response_model=List[MessageResponse])
def read_messages(
    chat_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Get all messages for a chat.
    """
    chat = ChatService.get_chat(db, chat_id)
    if not chat:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat not found",
        )

    # Check if user owns the chat
    if chat.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions",
        )

    messages = ChatService.get_messages(db, chat_id)
    return messages

@router.post("/{chat_id}/messages/{message_id}/feedback", response_model=MessageResponse)
def add_feedback(
    chat_id: str,
    message_id: str,
    feedback_in: FeedbackCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Add feedback to a message.
    """
    chat = ChatService.get_chat(db, chat_id)
    if not chat:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat not found",
        )

    # Check if user owns the chat
    if chat.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions",
        )

    message = ChatService.add_feedback(
        db,
        message_id,
        feedback_in.feedback,
        feedback_in.feedback_text
    )

    if not message:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Message not found",
        )

    return message