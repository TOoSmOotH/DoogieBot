from typing import Any, List

from app.db.base import get_db
from app.models.user import User
from app.schemas.chat import (ChatCreate, ChatListResponse, ChatResponse,
                              ChatUpdate, FeedbackCreate, MessageCreate,
                              MessageResponse, MessageUpdate,
                              PaginatedChatListResponse)
from app.schemas.chat import StreamingResponse as StreamingResponseSchema
from app.services.chat import ChatService
from app.services.llm import LLMService
from app.utils.deps import (get_current_admin_user, get_current_user,
                            get_current_user_stream)
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

router = APIRouter()


# Admin endpoints first (more specific routes)
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
                "messages": (
                    [
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
                            "context_documents": (
                                msg.context_documents
                                if msg.context_documents is None
                                else [str(doc_id) for doc_id in msg.context_documents]
                            ),
                        }
                        for msg in chat.messages
                    ]
                    if chat.messages
                    else None
                ),
            }
            for chat in chats
        ],
        "total": total,
        "page": skip // limit + 1 if limit > 0 else 1,
        "size": limit,
        "pages": (total + limit - 1) // limit if limit > 0 else 1,
    }


@router.get("/admin/feedback", response_model=List[MessageResponse])
def read_feedback_messages(
    db: Session = Depends(get_db),
    feedback_type: str = None,
    reviewed: bool = None,
    current_user: User = Depends(get_current_admin_user),
) -> Any:
    """
    Get messages with feedback. Admin only.
    """
    messages = ChatService.get_feedback_messages(db, feedback_type, reviewed)
    return messages


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


# Regular chat endpoints
@router.post("", response_model=ChatResponse)
def create_chat(
    chat_in: ChatCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Create a new chat.
    """
    chat = ChatService.create_chat(db, current_user.id, chat_in.title)
    return chat


@router.get("", response_model=List[ChatListResponse])
def read_chats(
    db: Session = Depends(get_db),
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Retrieve user's chats.
    """
    chats = ChatService.get_user_chats(db, current_user.id, skip=skip, limit=limit)

    # Ensure context_documents is properly formatted for each message
    for chat in chats:
        if chat.messages:
            for message in chat.messages:
                if message.context_documents is not None and not isinstance(
                    message.context_documents, list
                ):
                    # If it's a dict with a 'documents' key containing a list of objects with 'id' fields
                    if (
                        isinstance(message.context_documents, dict)
                        and "documents" in message.context_documents
                    ):
                        docs = message.context_documents["documents"]
                        if isinstance(docs, list) and all(
                            isinstance(doc, dict) and "id" in doc for doc in docs
                        ):
                            message.context_documents = [doc["id"] for doc in docs]
                        else:
                            message.context_documents = []
                    else:
                        # For other non-list formats, convert to empty list
                        message.context_documents = []

    return chats


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
        if message.context_documents is not None and not isinstance(
            message.context_documents, list
        ):
            message.context_documents = [
                str(doc_id) for doc_id in message.context_documents
            ]
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

    chat = ChatService.update_chat(db, chat_id, chat_in.title)
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

    message = ChatService.add_message(db, chat_id, message_in.role, message_in.content)
    return message


@router.post("/{chat_id}/llm", response_model=MessageResponse)
async def send_to_llm(
    chat_id: str,
    message_in: MessageCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Send a message to the LLM and get a response.
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

    # Add user message to the chat
    ChatService.add_message(db, chat_id, "user", message_in.content)

    # Initialize LLM service
    llm_service = LLMService(db)

    # Send message to LLM and get response
    response = await llm_service.chat(
        chat_id=chat_id, user_message=message_in.content, use_rag=True, stream=False
    )

    # Get the last message (assistant's response)
    messages = ChatService.get_messages(db, chat_id)
    if not messages:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get assistant response",
        )

    # Return the assistant's message
    for message in reversed(messages):
        if message.role == "assistant":
            return message

    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Failed to get assistant response",
    )


@router.post("/{chat_id}/stream")
async def stream_from_llm(
    chat_id: str,
    message_in: MessageCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> StreamingResponse:
    """
    Stream a response from the LLM.
    """
    import logging

    from app.core.config import settings

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    logger.debug(f"POST Stream request received for chat {chat_id}")

    chat = ChatService.get_chat(db, chat_id)
    if not chat:
        logger.error(f"Chat {chat_id} not found")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat not found",
        )

    # Check if user owns the chat
    if chat.user_id != current_user.id:
        logger.error(f"User {current_user.id} does not own chat {chat_id}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions",
        )

    # Add user message to the chat
    ChatService.add_message(db, chat_id, "user", message_in.content)

    logger.debug(f"Initializing LLM service for streaming")
    # Initialize LLM service
    llm_service = LLMService(db)

    # Create async generator for streaming response with improved error handling
    async def response_generator():
        import asyncio
        import json
        import logging
        import time
        import traceback

        logger = logging.getLogger(__name__)

        # Send a keep-alive message to prevent connection timeouts
        last_sent_time = time.time()
        keep_alive_interval = 15  # seconds

        try:
            logger.debug(f"Starting chat stream for chat {chat_id}")

            # Send an initial message to establish the connection
            initial_chunk = {"content": "", "status": "processing", "done": False}
            yield f"data: {json.dumps(initial_chunk)}\n\n"

            # Get the chat stream from the LLM service with a timeout wrapper
            try:
                # Use a timeout for the entire streaming operation
                chat_stream = await asyncio.wait_for(
                    llm_service.chat(
                        chat_id=chat_id,
                        user_message=message_in.content,
                        use_rag=True,
                        stream=True,
                    ),
                    timeout=30,  # 30 second timeout for getting the initial stream
                )

                # Stream chunks to the client with error handling
                chunk_count = 0
                try:
                    async for chunk in chat_stream:
                        chunk_count += 1
                        current_time = time.time()

                        # Log only occasionally to reduce overhead
                        if chunk_count % 10 == 0 or chunk.get("done", False):
                            logger.debug(
                                f"Streaming chunk {chunk_count}: {chunk.get('content', '')[:30]}... (done: {chunk.get('done', False)})"
                            )

                        # Convert chunk to JSON string and format as SSE
                        try:
                            yield f"data: {json.dumps(chunk)}\n\n"
                            last_sent_time = current_time
                        except Exception as json_error:
                            logger.error(
                                f"Error serializing chunk {chunk_count}: {str(json_error)}"
                            )
                            # Continue with next chunk instead of failing
                            continue

                        # Use a minimal delay only when necessary
                        if chunk_count % 20 == 0 and not chunk.get("done", False):
                            # Very minimal delay every 20 chunks
                            await asyncio.sleep(0.005)  # 5ms delay
                        else:
                            # Just yield to event loop without actual delay
                            await asyncio.sleep(0)

                        # Send keep-alive messages if needed
                        if (
                            current_time - last_sent_time > keep_alive_interval
                            and not chunk.get("done", False)
                        ):
                            keep_alive_chunk = {
                                "content": chunk.get("content", ""),
                                "status": "processing",
                                "done": False,
                            }
                            yield f"data: {json.dumps(keep_alive_chunk)}\n\n"
                            last_sent_time = current_time

                    logger.debug(
                        f"Finished streaming {chunk_count} chunks for chat {chat_id}"
                    )

                    # Send a final done message if we didn't get one from the stream
                    if chunk_count == 0 or not chunk.get("done", False):
                        final_chunk = {"content": "Response complete.", "done": True}
                        yield f"data: {json.dumps(final_chunk)}\n\n"
                except Exception as stream_loop_error:
                    # Handle errors in the streaming loop
                    logger.exception(
                        f"Error in streaming loop for chat {chat_id}: {str(stream_loop_error)}"
                    )
                    error_chunk = {
                        "content": f"An error occurred during streaming: {str(stream_loop_error)}",
                        "error": True,
                        "done": True,
                    }
                    yield f"data: {json.dumps(error_chunk)}\n\n"

                    # Add error message to chat
                    ChatService.add_message(
                        db,
                        chat_id,
                        "assistant",
                        f"An error occurred during streaming: {str(stream_loop_error)}",
                        context_documents={"error": str(stream_loop_error)},
                    )
            except asyncio.TimeoutError:
                logger.error(f"Timeout getting initial stream for chat {chat_id}")
                error_chunk = {
                    "content": "The response took too long to start. This might be due to high server load or complexity of the query with RAG processing.",
                    "error": True,
                    "done": True,
                }
                yield f"data: {json.dumps(error_chunk)}\n\n"

                # Add error message to chat
                ChatService.add_message(
                    db,
                    chat_id,
                    "assistant",
                    "The response took too long to start. This might be due to high server load or complexity of the query with RAG processing.",
                    context_documents={"error": "timeout_getting_stream"},
                )
            except Exception as chat_error:
                logger.exception(
                    f"Error getting chat stream for chat {chat_id}: {str(chat_error)}"
                )
                error_chunk = {
                    "content": f"An error occurred while preparing the response: {str(chat_error)}",
                    "error": True,
                    "done": True,
                }
                yield f"data: {json.dumps(error_chunk)}\n\n"

                # Add error message to chat
                ChatService.add_message(
                    db,
                    chat_id,
                    "assistant",
                    f"An error occurred while preparing the response: {str(chat_error)}",
                    context_documents={"error": str(chat_error)},
                )

        except asyncio.TimeoutError:
            logger.error(f"Timeout in streaming response for chat {chat_id}")
            error_chunk = {
                "content": "The response took too long to generate. This might be due to high server load or complexity of the query with RAG processing.",
                "error": True,
                "done": True,
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"

        except Exception as e:
            logger.exception(
                f"Error in streaming response: {str(e)}\n{traceback.format_exc()}"
            )
            # Send error message to client
            error_chunk = {
                "content": f"An error occurred: {str(e)}",
                "error": True,
                "done": True,
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"

    logger.debug(f"Returning StreamingResponse for chat {chat_id}")
    return StreamingResponse(
        response_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
            "X-Accel-Buffering": "no",  # Important for nginx proxying
            "Transfer-Encoding": "chunked",
        },
    )


@router.get("/{chat_id}/stream")
async def stream_from_llm_get(
    request: Request,
    chat_id: str,
    content: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user_stream),
) -> StreamingResponse:
    """
    Stream a response from the LLM using GET (for EventSource).
    """
    import logging

    from app.core.config import settings

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    logger.debug(
        f"Stream request received for chat {chat_id} with content: {content[:50]}..."
    )

    chat = ChatService.get_chat(db, chat_id)
    if not chat:
        logger.error(f"Chat {chat_id} not found")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat not found",
        )

    # Check if user owns the chat
    if chat.user_id != current_user.id:
        logger.error(f"User {current_user.id} does not own chat {chat_id}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions",
        )

    # Add user message to the chat
    ChatService.add_message(db, chat_id, "user", content)

    logger.debug(f"Initializing LLM service for streaming")
    # Initialize LLM service
    llm_service = LLMService(db)

    # Create async generator for streaming response with improved error handling
    async def response_generator():
        import asyncio
        import json
        import logging
        import time
        import traceback

        logger = logging.getLogger(__name__)

        # Send a keep-alive message to prevent connection timeouts
        last_sent_time = time.time()
        keep_alive_interval = 15  # seconds

        try:
            logger.debug(f"Starting chat stream for chat {chat_id}")

            # Send an initial message to establish the connection
            initial_chunk = {"content": "", "status": "processing", "done": False}
            yield f"data: {json.dumps(initial_chunk)}\n\n"

            # Get the chat stream from the LLM service with a timeout wrapper
            try:
                # Use a timeout for the entire streaming operation
                chat_stream = await asyncio.wait_for(
                    llm_service.chat(
                        chat_id=chat_id, user_message=content, use_rag=True, stream=True
                    ),
                    timeout=30,  # 30 second timeout for getting the initial stream
                )

                # Stream chunks to the client with error handling
                chunk_count = 0
                try:
                    async for chunk in chat_stream:
                        chunk_count += 1
                        current_time = time.time()

                        # Log only occasionally to reduce overhead
                        if chunk_count % 10 == 0 or chunk.get("done", False):
                            logger.debug(
                                f"Streaming chunk {chunk_count}: {chunk.get('content', '')[:30]}... (done: {chunk.get('done', False)})"
                            )

                        # Convert chunk to JSON string and format as SSE
                        try:
                            yield f"data: {json.dumps(chunk)}\n\n"
                            last_sent_time = current_time
                        except Exception as json_error:
                            logger.error(
                                f"Error serializing chunk {chunk_count}: {str(json_error)}"
                            )
                            # Continue with next chunk instead of failing
                            continue

                        # Use a minimal delay only when necessary
                        # This helps ensure the client can process chunks properly
                        # without overwhelming it or causing browser buffering issues
                        if chunk_count % 20 == 0 and not chunk.get("done", False):
                            # Very minimal delay every 20 chunks
                            await asyncio.sleep(0.005)  # 5ms delay
                        else:
                            # Just yield to event loop without actual delay
                            await asyncio.sleep(0)

                        # Send keep-alive messages if needed
                        if (
                            current_time - last_sent_time > keep_alive_interval
                            and not chunk.get("done", False)
                        ):
                            keep_alive_chunk = {
                                "content": chunk.get("content", ""),
                                "status": "processing",
                                "done": False,
                            }
                            yield f"data: {json.dumps(keep_alive_chunk)}\n\n"
                            last_sent_time = current_time

                    logger.debug(
                        f"Finished streaming {chunk_count} chunks for chat {chat_id}"
                    )

                    # Send a final done message if we didn't get one from the stream
                    if chunk_count == 0 or not chunk.get("done", False):
                        final_chunk = {"content": "Response complete.", "done": True}
                        yield f"data: {json.dumps(final_chunk)}\n\n"
                except Exception as stream_loop_error:
                    # Handle errors in the streaming loop
                    logger.exception(
                        f"Error in streaming loop for chat {chat_id}: {str(stream_loop_error)}"
                    )
                    error_chunk = {
                        "content": f"An error occurred during streaming: {str(stream_loop_error)}",
                        "error": True,
                        "done": True,
                    }
                    yield f"data: {json.dumps(error_chunk)}\n\n"

                    # Add error message to chat
                    ChatService.add_message(
                        db,
                        chat_id,
                        "assistant",
                        f"An error occurred during streaming: {str(stream_loop_error)}",
                        context_documents={"error": str(stream_loop_error)},
                    )
            except asyncio.TimeoutError:
                logger.error(f"Timeout getting initial stream for chat {chat_id}")
                error_chunk = {
                    "content": "The response took too long to start. This might be due to high server load or complexity of the query with RAG processing.",
                    "error": True,
                    "done": True,
                }
                yield f"data: {json.dumps(error_chunk)}\n\n"

                # Add error message to chat
                ChatService.add_message(
                    db,
                    chat_id,
                    "assistant",
                    "The response took too long to start. This might be due to high server load or complexity of the query with RAG processing.",
                    context_documents={"error": "timeout_getting_stream"},
                )
            except Exception as chat_error:
                logger.exception(
                    f"Error getting chat stream for chat {chat_id}: {str(chat_error)}"
                )
                error_chunk = {
                    "content": f"An error occurred while preparing the response: {str(chat_error)}",
                    "error": True,
                    "done": True,
                }
                yield f"data: {json.dumps(error_chunk)}\n\n"

                # Add error message to chat
                ChatService.add_message(
                    db,
                    chat_id,
                    "assistant",
                    f"An error occurred while preparing the response: {str(chat_error)}",
                    context_documents={"error": str(chat_error)},
                )

        except asyncio.TimeoutError:
            logger.error(f"Timeout in streaming response for chat {chat_id}")
            error_chunk = {
                "content": "The response took too long to generate. This might be due to high server load or complexity of the query with RAG processing.",
                "error": True,
                "done": True,
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"

        except Exception as e:
            logger.exception(
                f"Error in streaming response: {str(e)}\n{traceback.format_exc()}"
            )
            # Send error message to client
            error_chunk = {
                "content": f"An error occurred: {str(e)}",
                "error": True,
                "done": True,
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"

    logger.debug(f"Returning StreamingResponse for chat {chat_id}")
    return StreamingResponse(
        response_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
            "X-Accel-Buffering": "no",  # Important for nginx proxying
            "Transfer-Encoding": "chunked",
        },
    )


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


@router.post(
    "/{chat_id}/messages/{message_id}/feedback", response_model=MessageResponse
)
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
        db, message_id, feedback_in.feedback, feedback_in.feedback_text
    )

    if not message:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Message not found",
        )

    return message
