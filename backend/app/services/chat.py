import uuid
from typing import List, Optional

from app.models.chat import Chat, FeedbackType, Message, MessageRole
from app.models.user import User
from sqlalchemy.orm import Session, joinedload


class ChatService:
    @staticmethod
    def create_chat(db: Session, user_id: str, title: Optional[str] = None) -> Chat:
        """
        Create a new chat for a user.
        """
        chat_id = str(uuid.uuid4())
        chat = Chat(id=chat_id, user_id=user_id, title=title or "New Chat")
        db.add(chat)
        db.commit()
        db.refresh(chat)
        return chat

    @staticmethod
    def get_chat(db: Session, chat_id: str) -> Optional[Chat]:
        """
        Get a chat by ID.
        """
        return db.query(Chat).filter(Chat.id == chat_id).first()

    @staticmethod
    def get_user_chats(
        db: Session, user_id: str, skip: int = 0, limit: int = 100
    ) -> List[Chat]:
        """
        Get all chats for a user.
        """
        return (
            db.query(Chat)
            .filter(Chat.user_id == user_id)
            .order_by(Chat.updated_at.desc())
            .offset(skip)
            .limit(limit)
            .all()
        )

    @staticmethod
    def update_chat(db: Session, chat_id: str, title: str) -> Optional[Chat]:
        """
        Update a chat's title.
        """
        chat = ChatService.get_chat(db, chat_id)
        if not chat:
            return None

        chat.title = title
        db.commit()
        db.refresh(chat)
        return chat

    @staticmethod
    def delete_chat(db: Session, chat_id: str) -> bool:
        """
        Delete a chat and all its messages.
        """
        chat = ChatService.get_chat(db, chat_id)
        if not chat:
            return False

        db.delete(chat)
        db.commit()
        return True

    @staticmethod
    def add_message(
        db: Session,
        chat_id: str,
        role: str,
        content: str,
        tokens: Optional[int] = None,
        tokens_per_second: Optional[float] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        context_documents: Optional[dict] = None,
    ) -> Message:
        """
        Add a message to a chat.
        """
        from datetime import datetime

        # Get current timestamp
        current_time = datetime.utcnow()

        message_id = str(uuid.uuid4())
        message = Message(
            id=message_id,
            chat_id=chat_id,
            role=role,
            content=content,
            tokens=tokens,
            tokens_per_second=tokens_per_second,
            model=model,
            provider=provider,
            context_documents=context_documents,
            created_at=current_time,
        )
        db.add(message)

        # Update chat's updated_at timestamp
        chat = ChatService.get_chat(db, chat_id)
        if chat:
            chat.updated_at = current_time

        db.commit()
        db.refresh(message)
        return message

    @staticmethod
    def get_messages(db: Session, chat_id: str) -> List[Message]:
        """
        Get all messages for a chat.
        """
        return (
            db.query(Message)
            .filter(Message.chat_id == chat_id)
            .order_by(Message.created_at)
            .all()
        )

    @staticmethod
    def add_feedback(
        db: Session, message_id: str, feedback: str, feedback_text: Optional[str] = None
    ) -> Optional[Message]:
        """
        Add feedback to a message.
        """
        message = db.query(Message).filter(Message.id == message_id).first()
        if not message:
            return None

        message.feedback = feedback
        message.feedback_text = feedback_text
        db.commit()
        db.refresh(message)
        return message

    @staticmethod
    def mark_as_reviewed(db: Session, message_id: str) -> Optional[Message]:
        """
        Mark a message as reviewed.
        """
        message = db.query(Message).filter(Message.id == message_id).first()
        if not message:
            return None

        message.reviewed = True
        db.commit()
        db.refresh(message)
        return message

    @staticmethod
    def get_feedback_messages(
        db: Session,
        feedback_type: Optional[str] = None,
        reviewed: Optional[bool] = None,
    ) -> List[Message]:
        """
        Get messages with feedback, optionally filtered by feedback type and review status.
        """
        query = db.query(Message).filter(Message.feedback.isnot(None))

        if feedback_type:
            query = query.filter(Message.feedback == feedback_type)

        if reviewed is not None:
            query = query.filter(Message.reviewed == reviewed)

        return query.order_by(Message.created_at.desc()).all()

    @staticmethod
    def get_flagged_chats(
        db: Session, skip: int = 0, limit: int = 10
    ) -> tuple[List[Chat], int]:
        """
        Get paginated chats that have messages with negative feedback.
        Returns a tuple of (chats, total_count).
        """
        # Subquery to get chat IDs that have messages with negative feedback
        from sqlalchemy import distinct

        flagged_chat_ids = (
            db.query(distinct(Message.chat_id))
            .filter(Message.feedback == "negative")
            .subquery()
        )

        # Query to get the chats with these IDs and load messages
        query = (
            db.query(Chat)
            .filter(Chat.id.in_(flagged_chat_ids))
            .options(joinedload(Chat.messages))
        )

        # Get total count before pagination
        total = query.count()

        # Apply pagination
        chats = query.order_by(Chat.updated_at.desc()).offset(skip).limit(limit).all()

        return chats, total
