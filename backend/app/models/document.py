from sqlalchemy import Column, String, Text, DateTime, ForeignKey, JSON, Integer
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import enum
import uuid

from app.db.base import Base

class DocumentType(str, enum.Enum):
    """Enum for document types."""
    PDF = "pdf"
    DOCX = "docx"
    MARKDOWN = "markdown"
    RST = "rst"
    TEXT = "txt"
    JSON = "json"
    JSONL = "jsonl"
    YAML = "yaml"
    YML = "yml"
    MANUAL = "manual"
    TXT = "txt"
    HTML = "html"
    PPTX = "pptx"
    CSV = "csv"
    XLSX = "xlsx"
    XML = "xml"
    PLAINTEXT = "plaintext"
    OTHER = "other"

class Document(Base):
    """
    Model for storing document metadata and original content.
    """
    __tablename__ = "documents"

    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    filename = Column(String, nullable=True)
    title = Column(String, nullable=True)
    type = Column(String, nullable=False)  # e.g., pdf, txt, etc.
    content = Column(Text, nullable=True)  # Original document content
    meta_data = Column(JSON, nullable=True)  # Additional metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    uploaded_by = Column(String, ForeignKey("users.id"), nullable=False)
    
    # Relationships
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")
    uploader = relationship("User", back_populates="documents")
    
    def __repr__(self):
        return f"<Document id={self.id}, title={self.title}>"

class DocumentChunk(Base):
    """
    Model for storing document chunks for RAG.
    Each document is split into multiple chunks for efficient embedding and retrieval.
    """
    __tablename__ = "document_chunks"

    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    document_id = Column(String, ForeignKey("documents.id"), nullable=False)
    content = Column(Text, nullable=False)  # Chunk text content
    meta_data = Column(JSON, nullable=True)  # Additional metadata
    chunk_index = Column(Integer, nullable=False)  # Position in the document
    embedding = Column(JSON, nullable=True)  # Vector embedding
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Relationships
    document = relationship("Document", back_populates="chunks")
    graph_nodes = relationship("GraphNode", back_populates="document_chunk", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<DocumentChunk id={self.id}, document_id={self.document_id}, index={self.chunk_index}>"
