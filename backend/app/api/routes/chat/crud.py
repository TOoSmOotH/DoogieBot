from typing import Any, List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.base import get_db
from app.models.user import User
from app.utils.deps import get_current_user
from app.schemas.chat import (
    ChatCreate,
    ChatUpdate,
    ChatResponse,
    ChatListResponse,
)
from app.services.chat import ChatService

router = APIRouter()

# Function definition without decorator
def create_chat_func(
    chat_in: ChatCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Create a new chat. (Function implementation)
    """
    chat = ChatService.create_chat(db, current_user.id, chat_in.title)
    return chat

# Function definition without decorator
def read_chats_func(
    db: Session = Depends(get_db),
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Retrieve user's chats. (Function implementation)
    """
    chats = ChatService.get_user_chats(db, current_user.id, skip=skip, limit=limit)

    # Ensure context_documents is properly formatted for each message
    for chat in chats:
        if chat.messages:
            for message in chat.messages:
                if message.context_documents is not None and not isinstance(message.context_documents, list):
                    # If it's a dict with a 'documents' key containing a list of objects with 'id' fields
                    if isinstance(message.context_documents, dict) and 'documents' in message.context_documents:
                        docs = message.context_documents['documents']
                        if isinstance(docs, list) and all(isinstance(doc, dict) and 'id' in doc for doc in docs):
                            message.context_documents = [doc['id'] for doc in docs]
                        else:
                            message.context_documents = []
                    else:
                        # For other non-list formats, convert to empty list
                        message.context_documents = []

    return chats

# Routes with non-empty paths remain decorated
@router.get("/{chat_id}", response_model=ChatResponse)
def read_chat(
    chat_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Get a specific chat by id.
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

    # Get messages for the chat
    chat.messages = ChatService.get_messages(db, chat_id)
    for message in chat.messages:
        if message.context_documents is not None and not isinstance(message.context_documents, list):
            message.context_documents = [str(doc_id) for doc_id in message.context_documents]
    return chat

@router.put("/{chat_id}", response_model=ChatResponse)
def update_chat(
    chat_id: str,
    chat_in: ChatUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Update a chat.
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

    # If tags are provided, update them separately
    if chat_in.tags is not None:
        from app.services.tag import update_chat_tags # Local import
        success = update_chat_tags(db, chat_id, chat_in.tags)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update chat tags",
            )

    # Update chat title
    if chat_in.title is not None:
        chat = ChatService.update_chat(db, chat_id, chat_in.title)

    # Get the updated chat with associated messages
    chat.messages = ChatService.get_messages(db, chat_id)
    return chat

@router.delete("/{chat_id}", response_model=bool)
def delete_chat(
    chat_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Delete a chat.
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

    result = ChatService.delete_chat(db, chat_id)
    return result