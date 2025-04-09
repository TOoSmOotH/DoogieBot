from typing import List, Any
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

# Import routers from the new modules within the 'chat' subdirectory
from .chat import admin, crud, llm, messages, stream
# Import specific functions that were previously at the root path
from .chat.crud import create_chat_func, read_chats_func

# Import necessary dependencies and schemas for the root routes
from app.db.base import get_db
from app.models.user import User
from app.utils.deps import get_current_user
from app.schemas.chat import ChatCreate, ChatResponse, ChatListResponse

router = APIRouter()

# Define the root routes directly on this router
@router.post("", response_model=ChatResponse, tags=["chat_crud"])
def create_chat_endpoint(
    chat_in: ChatCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Any:
    """ Create a new chat. """
    return create_chat_func(chat_in=chat_in, db=db, current_user=current_user)

@router.get("", response_model=List[ChatListResponse], tags=["chat_crud"])
def read_chats_endpoint(
    db: Session = Depends(get_db),
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_user),
) -> Any:
    """ Retrieve user's chats. """
    return read_chats_func(db=db, skip=skip, limit=limit, current_user=current_user)


# Include the routers from the submodules
# Note: crud.router now only contains routes with non-empty paths like /{chat_id}
router.include_router(admin.router, tags=["chat_admin"])
router.include_router(crud.router, tags=["chat_crud"]) # Includes /{chat_id}, PUT, DELETE etc.
router.include_router(messages.router, tags=["chat_messages"])
router.include_router(llm.router, tags=["chat_llm"])
router.include_router(stream.router, tags=["chat_stream"])