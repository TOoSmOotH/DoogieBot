from sqlalchemy import Column, String, Boolean, DateTime, Float, Integer, func
from sqlalchemy.dialects.sqlite import JSON
import uuid

from app.db.base import Base

class LLMConfig(Base):
    """
    Model for storing LLM configuration.
    Only one configuration should be active at a time.
    The system_prompt is global and used for all LLM providers.
    """
    __tablename__ = "llm_config"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    provider = Column(String, nullable=False)  # Legacy field for backward compatibility
    chat_provider = Column(String, nullable=False)
    embedding_provider = Column(String, nullable=False)
    model = Column(String, nullable=False)
    embedding_model = Column(String, nullable=False)
    system_prompt = Column(String, nullable=False)
    api_key = Column(String, nullable=True)
    base_url = Column(String, nullable=True)
    temperature = Column(Float, nullable=True, default=0.7)  # Added temperature field
    is_active = Column(Boolean, default=False)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    reranked_top_n = Column(Integer, nullable=True) # Number of docs to send to LLM after reranking

    # Additional configuration stored as JSON
    # Can include:
    # - rag_top_k: Number of RAG results to retrieve initially
    # - use_reranking: Boolean flag to enable/disable reranking
    # - (reranked_top_n is now a top-level field)
    # - (temperature is now a top-level field)
    config = Column(JSON, nullable=True)