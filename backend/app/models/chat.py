from sqlalchemy import Boolean, Column, String, DateTime, ForeignKey, Text, Integer, Float, JSON
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
import enum
from app.db.base import Base

class MessageRole(str, enum.Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"

class FeedbackType(str, enum.Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"

class Chat(Base):
    __tablename__ = "chats"

    id = Column(String, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    title = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    
    # Relationships
    messages = relationship("Message", back_populates="chat", cascade="all, delete-orphan")
    chat_tags = relationship("ChatTag", back_populates="chat", cascade="all, delete-orphan")
    tags = relationship("Tag", secondary="chat_tags", back_populates="chats", overlaps="chat_tags")
    
    def __repr__(self):
        return f"<Chat {self.id}>"

class Message(Base):
    __tablename__ = "messages"

    id = Column(String, primary_key=True, index=True)
    chat_id = Column(String, ForeignKey("chats.id"), nullable=False)
    role = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # LLM metadata
    tokens = Column(Integer, nullable=True)
    tokens_per_second = Column(Float, nullable=True)
    model = Column(String, nullable=True)
    provider = Column(String, nullable=True)
    
    # Feedback
    feedback = Column(String, nullable=True)
    feedback_text = Column(Text, nullable=True)
    reviewed = Column(Boolean, default=False)
    
    # RAG metadata
    context_documents = Column(JSON, nullable=True)
    
    # Link to the user question for feedback context
    related_question_id = Column(String, ForeignKey("messages.id"), nullable=True)
    related_question = relationship("Message", remote_side=[id]) # Self-referential relationship
    
    # Relationships
    chat = relationship("Chat", back_populates="messages")
    
    def __repr__(self):
        return f"<Message {self.id}>"
