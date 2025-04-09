"""
This module ensures all SQLAlchemy models are imported
before any database operations to prevent missing table errors.
"""

import logging
from app.db.base import Base

logger = logging.getLogger(__name__)

# Import all models here using try-except to handle missing modules
try:
    from app.models.user import User, UserRole, UserStatus
    logger.debug("Imported User models")
except ImportError as e:
    logger.warning(f"Failed to import User models: {e}")

try:
    from app.models.chat import Chat, Message, MessageRole, FeedbackType
    logger.debug("Imported Chat models")
except ImportError as e:
    logger.warning(f"Failed to import Chat models: {e}")

try:
    from app.models.document import Document, DocumentChunk
    logger.debug("Imported Document models")
except ImportError as e:
    logger.warning(f"Failed to import Document models: {e}")

try:
    from app.models.graph import GraphNode, GraphEdge
    logger.debug("Imported Graph models")
except ImportError as e:
    logger.warning(f"Failed to import Graph models: {e}")

try:
    from app.models.llm_config import LLMConfig
    logger.debug("Imported LLMConfig models")
except ImportError as e:
    logger.warning(f"Failed to import LLMConfig models: {e}")

try:
    from app.models.rag_config import RAGConfig
    logger.debug("Imported RAGConfig models")
except ImportError as e:
    logger.warning(f"Failed to import RAGConfig models: {e}")

try:
    from app.models.tag import Tag, ChatTag
    logger.debug("Imported Tag models")
except ImportError as e:
    logger.warning(f"Failed to import Tag models: {e}")

try:
    from app.models.embedding_config import EmbeddingConfig
    logger.debug("Imported EmbeddingConfig models")
except ImportError as e:
    logger.warning(f"Failed to import EmbeddingConfig models: {e}")

# This ensures all models are known to SQLAlchemy 
# before any operations involving Base.metadata
