from sqlalchemy import Column, String, DateTime, JSON, ForeignKey, Boolean, Integer
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid

from app.db.base import Base

class IndexMeta(Base):
    """
    Model for storing metadata about RAG indexes.
    """
    __tablename__ = "index_meta"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False)
    type = Column(String, nullable=False)  # bm25, faiss, graph
    document_count = Column(Integer, default=0)
    chunk_count = Column(Integer, default=0)
    last_updated = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now())
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    is_active = Column(Boolean, default=True)
    
    # Additional metadata
    config = Column(JSON, nullable=True)
    stats = Column(JSON, nullable=True)
    
    # Relationships
    operations = relationship("IndexOperation", back_populates="index", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<IndexMeta id={self.id}, name={self.name}, type={self.type}>"

class IndexOperation(Base):
    """
    Model for tracking index operations like builds and updates.
    """
    __tablename__ = "index_operations"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    index_id = Column(String, ForeignKey("index_meta.id"), nullable=False)
    operation_type = Column(String, nullable=False)  # build, update, delete
    status = Column(String, nullable=False)  # pending, running, completed, failed
    started_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    documents_processed = Column(Integer, default=0)
    errors = Column(JSON, nullable=True)
    
    # Relationship to index
    index = relationship("IndexMeta", back_populates="operations")

    def __repr__(self):
        return f"<IndexOperation id={self.id}, index_id={self.index_id}, type={self.operation_type}, status={self.status}>"
