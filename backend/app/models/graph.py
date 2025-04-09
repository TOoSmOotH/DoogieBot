from sqlalchemy import Column, String, Integer, ForeignKey, DateTime, Text, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid

from app.db.base import Base

class GraphNode(Base):
    """
    Model for storing graph nodes that represent entities extracted from documents.
    Each node is linked to a document chunk.
    """
    __tablename__ = "graph_nodes"

    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    chunk_id = Column(String, ForeignKey("document_chunks.id"), nullable=False)
    node_type = Column(String, nullable=False)  # e.g., person, place, concept, etc.
    content = Column(Text, nullable=False)
    meta_data = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Relationships
    # Define relationships to GraphEdge for both source and target nodes
    outgoing_edges = relationship("GraphEdge", 
                                  foreign_keys="GraphEdge.source_id",
                                  back_populates="source_node", 
                                  cascade="all, delete-orphan")
    incoming_edges = relationship("GraphEdge", 
                                 foreign_keys="GraphEdge.target_id",
                                 back_populates="target_node", 
                                 cascade="all, delete-orphan")
    document_chunk = relationship("DocumentChunk", back_populates="graph_nodes")
    
    def __repr__(self):
        return f"<GraphNode id={self.id}, type={self.node_type}>"

class GraphEdge(Base):
    """
    Model for storing relationships between graph nodes.
    Each edge connects two nodes with a specific relationship type.
    """
    __tablename__ = "graph_edges"

    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    source_id = Column(String, ForeignKey("graph_nodes.id"), nullable=False)
    target_id = Column(String, ForeignKey("graph_nodes.id"), nullable=False)
    relation_type = Column(String, nullable=False)  # e.g., "works_for", "located_in", etc.
    weight = Column(Integer, nullable=True)  # For weighted graphs
    meta_data = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Relationships
    source_node = relationship("GraphNode", 
                              foreign_keys=[source_id],
                              back_populates="outgoing_edges")
    target_node = relationship("GraphNode", 
                              foreign_keys=[target_id],
                              back_populates="incoming_edges")
    
    def __repr__(self):
        return f"<GraphEdge id={self.id}, relation={self.relation_type}>"
