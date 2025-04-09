from sqlalchemy import Column, String, Boolean, DateTime, func
from sqlalchemy.dialects.sqlite import JSON
import uuid

from app.db.base import Base

class RerankingConfig(Base):
    """
    Model for storing reranking configuration.
    Each configuration can be independently activated.
    """
    __tablename__ = "reranking_config"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    provider = Column(String, nullable=False)
    model = Column(String, nullable=False)
    api_key = Column(String, nullable=True)
    base_url = Column(String, nullable=True)
    is_active = Column(Boolean, default=False)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Additional configuration stored as JSON
    config = Column(JSON, nullable=True)

    def __repr__(self):
        return f"<RerankingConfig id={self.id}, provider={self.provider}, model={self.model}>"
