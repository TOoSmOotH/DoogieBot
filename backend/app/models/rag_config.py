from sqlalchemy import Column, String, Boolean, DateTime, JSON
from sqlalchemy.sql import func
import uuid

from app.db.base import Base

class RAGConfig(Base):
    """
    Model for storing RAG configuration settings.
    Controls which RAG components are enabled and their settings.
    """
    __tablename__ = "rag_config"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    bm25_enabled = Column(Boolean, default=True)
    faiss_enabled = Column(Boolean, default=True)
    graph_enabled = Column(Boolean, default=True)
    graph_implementation = Column(String, default="networkx")
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Additional configuration stored as JSON
    config = Column(JSON, nullable=True)
    
    def __repr__(self):
        return f"<RAGConfig id={self.id}>"
