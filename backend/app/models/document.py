import enum

from app.db.base import Base
from sqlalchemy import (JSON, Column, DateTime, ForeignKey, Integer, String,
                        Text)
from sqlalchemy.sql import func


class DocumentType(str, enum.Enum):
    PDF = "pdf"
    DOCX = "docx"
    MARKDOWN = "md"
    RST = "rst"
    TEXT = "txt"
    JSON = "json"
    JSONL = "jsonl"
    YAML = "yaml"
    YML = "yml"
    MANUAL = "manual"


class Document(Base):
    __tablename__ = "documents"

    id = Column(String, primary_key=True, index=True)
    filename = Column(String, nullable=True)
    title = Column(String, nullable=True)
    type = Column(String, nullable=False)
    content = Column(Text, nullable=True)  # Original content or path to file
    meta_data = Column(JSON, nullable=True)
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
    uploaded_by = Column(String, ForeignKey("users.id"), nullable=False)

    def __repr__(self):
        return f"<Document {self.id}>"


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(String, primary_key=True, index=True)
    document_id = Column(String, ForeignKey("documents.id"), nullable=False)
    content = Column(Text, nullable=False)
    meta_data = Column(JSON, nullable=True)
    chunk_index = Column(Integer, nullable=False)
    embedding = Column(JSON, nullable=True)  # Store as JSON for SQLite compatibility
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    def __repr__(self):
        return f"<DocumentChunk {self.id}>"


class GraphNode(Base):
    __tablename__ = "graph_nodes"

    id = Column(String, primary_key=True, index=True)
    chunk_id = Column(String, ForeignKey("document_chunks.id"), nullable=False)
    node_type = Column(String, nullable=False)  # entity, concept, etc.
    content = Column(Text, nullable=False)
    meta_data = Column(JSON, nullable=True)
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    def __repr__(self):
        return f"<GraphNode {self.id}>"


class GraphEdge(Base):
    __tablename__ = "graph_edges"

    id = Column(String, primary_key=True, index=True)
    source_id = Column(String, ForeignKey("graph_nodes.id"), nullable=False)
    target_id = Column(String, ForeignKey("graph_nodes.id"), nullable=False)
    relation_type = Column(String, nullable=False)
    weight = Column(Integer, nullable=True)
    meta_data = Column(JSON, nullable=True)
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    def __repr__(self):
        return f"<GraphEdge {self.id}>"
