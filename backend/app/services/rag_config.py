from typing import Optional

from app.models.rag_config import RAGConfig
from sqlalchemy.orm import Session


class RAGConfigService:
    @staticmethod
    def get_config(db: Session) -> Optional[RAGConfig]:
        """
        Get the RAG configuration. If no configuration exists, create a default one.

        Args:
            db: Database session

        Returns:
            RAG configuration
        """
        # Get the first config (there should only be one)
        config = db.query(RAGConfig).first()

        # If no config exists, create a default one
        if not config:
            config = RAGConfig(
                bm25_enabled=True, faiss_enabled=True, graph_enabled=True
            )
            db.add(config)
            db.commit()
            db.refresh(config)

        return config

    @staticmethod
    def update_component_status(
        db: Session, component: str, enabled: bool
    ) -> Optional[RAGConfig]:
        """
        Update the enabled status of a RAG component.

        Args:
            db: Database session
            component: Component name ('bm25', 'faiss', or 'graph')
            enabled: Whether the component should be enabled

        Returns:
            Updated RAG configuration
        """
        # Get the config
        config = RAGConfigService.get_config(db)

        # Update the component status
        if component == "bm25":
            config.bm25_enabled = enabled
        elif component == "faiss":
            config.faiss_enabled = enabled
        elif component == "graph":
            config.graph_enabled = enabled
        else:
            return None

        # Save changes
        db.commit()
        db.refresh(config)

        return config
