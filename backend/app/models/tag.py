from sqlalchemy import Column, String, ForeignKey, Table, DateTime
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid

from app.db.base import Base

class Tag(Base):
    """
    Model for storing user-defined tags for chats.
    """
    __tablename__ = "tags"

    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False)
    color = Column(String, nullable=False, default="#3498db")  # Default color
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="tags")
    chats = relationship("Chat", secondary="chat_tags", back_populates="tags", overlaps="chat_tags")
    chat_tags = relationship("ChatTag", back_populates="tag")
    
    def __repr__(self):
        return f"<Tag id={self.id}, name={self.name}>"

class ChatTag(Base):
    """
    Association table for Chat-Tag many-to-many relationship.
    """
    __tablename__ = "chat_tags"

    chat_id = Column(String, ForeignKey("chats.id", ondelete="CASCADE"), primary_key=True)
    tag_id = Column(String, ForeignKey("tags.id", ondelete="CASCADE"), primary_key=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Relationships
    chat = relationship("Chat", back_populates="chat_tags")
    tag = relationship("Tag", back_populates="chat_tags")
    
    def __repr__(self):
        return f"<ChatTag chat_id={self.chat_id}, tag_id={self.tag_id}>"
