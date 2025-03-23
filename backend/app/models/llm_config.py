import uuid

from app.db.base import Base
from sqlalchemy import Boolean, Column, DateTime, String, func
from sqlalchemy.dialects.sqlite import JSON


class LLMConfig(Base):
    """
    Model for storing LLM configuration.
    Only one configuration should be active at a time.
    The system_prompt is global and used for all LLM providers.
    """

    __tablename__ = "llm_config"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    provider = Column(String, nullable=False)
    model = Column(String, nullable=False)
    embedding_model = Column(String, nullable=False)
    system_prompt = Column(String, nullable=False)
    api_key = Column(String, nullable=True)
    base_url = Column(String, nullable=True)
    is_active = Column(Boolean, default=False)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # Additional configuration stored as JSON
    config = Column(JSON, nullable=True)
