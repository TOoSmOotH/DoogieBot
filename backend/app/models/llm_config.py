from sqlalchemy import Column, String, Boolean, DateTime, Float, Integer, func, JSON
import uuid

from app.db.base import Base

class LLMConfig(Base):
    """
    Model for storing LLM configuration.
    Each configuration can be independently activated.
    """
    __tablename__ = "llm_config"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    provider = Column(String, nullable=False)  # Legacy field - kept for backward compatibility
    chat_provider = Column(String, nullable=True)  # New field for chat provider
    embedding_provider = Column(String, nullable=True)  # New field for embedding provider
    model = Column(String, nullable=False)
    embedding_model = Column(String, nullable=False)
    system_prompt = Column(String, nullable=False)
    api_key = Column(String, nullable=True)
    base_url = Column(String, nullable=True)
    temperature = Column(Float, nullable=True, default=0.7)  # Added temperature field
    is_active = Column(Boolean, default=False, index=True)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    reranked_top_n = Column(Integer, nullable=True) # Number of docs to send to LLM after reranking

    # Additional configuration stored as JSON
    config = Column(JSON, nullable=True)
    
    def __repr__(self):
        return f"<LLMConfig id={self.id}, provider={self.provider}>"
