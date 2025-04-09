from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import update
import logging
from datetime import datetime, UTC
from app.models.embedding_config import EmbeddingConfig
from app.schemas.embedding import EmbeddingConfigCreate, EmbeddingConfigUpdate
from app.core.config import settings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingConfigService:
    """
    Service for managing embedding configurations.
    """
    
    @staticmethod
    def create_config(db: Session, config: EmbeddingConfigCreate) -> EmbeddingConfig:
        """
        Create a new embedding configuration.
        
        Args:
            db: Database session
            config: Embedding configuration data
            
        Returns:
            Created embedding configuration
        """
        # Import datetime for explicit datetime fields
        from datetime import datetime, UTC
        
        # Create config with explicit datetime fields
        db_config = EmbeddingConfig(
            provider=config.provider,
            model=config.model,
            api_key=config.api_key,
            base_url=config.base_url,
            config=config.config,
            is_active=False,  # New configs are not active by default
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC)
        )
        
        db.add(db_config)
        db.commit()
        db.refresh(db_config)
        
        return db_config
    @staticmethod
    def get_config(db: Session, config_id: str) -> Optional[EmbeddingConfig]:
        """
        Get an embedding configuration by ID.
        
        Args:
            db: Database session
            config_id: Embedding configuration ID
            
        Returns:
            Embedding configuration or None if not found
        """
        # Import datetime for explicit datetime fields
        from datetime import datetime
        
        # Get config by ID
        config = db.query(EmbeddingConfig).filter(EmbeddingConfig.id == config_id).first()
        
        # If config exists but has None for datetime fields, set them
        if config:
            if config.created_at is None:
                config.created_at = datetime.now(UTC)
                
            if config.updated_at is None:
                config.updated_at = datetime.now(UTC)
                
            # Commit the changes to ensure the fields are saved
            db.commit()
            
        return config
    @staticmethod
    def get_active_config(db: Session) -> Optional[EmbeddingConfig]:
        """
        Get the active embedding configuration.
        
        Args:
            db: Database session
            
        Returns:
            Active embedding configuration or None if not found
        """
        # Import datetime for explicit datetime fields
        from datetime import datetime
        
        # Get active config
        config = db.query(EmbeddingConfig).filter(EmbeddingConfig.is_active == True).first()
        
        # If config exists but has None for datetime fields, set them
        if config:
            if config.created_at is None:
                config.created_at = datetime.now(UTC)
                
            if config.updated_at is None:
                config.updated_at = datetime.now(UTC)
                
            # Commit the changes to ensure the fields are saved
            db.commit()
            
        return config
    
    @staticmethod
    def get_all_configs(db: Session) -> List[EmbeddingConfig]:
        """
        Get all embedding configurations.
        
        Args:
            db: Database session
            
        Returns:
            List of embedding configurations
        """
        # Import datetime for explicit datetime fields
        from datetime import datetime
        
        # Get all configs
        configs = db.query(EmbeddingConfig).all()
        
        # Check each config for None datetime fields
        for config in configs:
            if config.created_at is None:
                config.created_at = datetime.now(UTC)
                
            if config.updated_at is None:
                config.updated_at = datetime.now(UTC)
        
        # If any configs were updated, commit the changes
        if configs and any(config.created_at is None or config.updated_at is None for config in configs):
            db.commit()
            
        return configs
    
    @staticmethod
    def update_config(db: Session, config_id: str, config_update: EmbeddingConfigUpdate) -> Optional[EmbeddingConfig]:
        """
        Update an embedding configuration.
        
        Args:
            db: Database session
            config_id: Embedding configuration ID
            config_update: Embedding configuration update data
            
        Returns:
            Updated embedding configuration or None if not found
        """
        # Import datetime for explicit datetime fields
        from datetime import datetime
        
        # Get existing config
        db_config = EmbeddingConfigService.get_config(db, config_id)
        if not db_config:
            return None
        
        # Update fields
        update_data = config_update.model_dump(exclude_unset=True)
        
        # If setting this config as active, deactivate all other configs
        if update_data.get("is_active", False):
            db.execute(
                update(EmbeddingConfig)
                .where(EmbeddingConfig.id != config_id)
                .values(is_active=False)
            )
        
        # Update config
        for key, value in update_data.items():
            setattr(db_config, key, value)
        
        # Explicitly set updated_at to ensure it's not None
        db_config.updated_at = datetime.now(UTC)
        
        # Ensure created_at is set if it's None
        if db_config.created_at is None:
            db_config.created_at = datetime.now(UTC)
        
        db.commit()
        db.refresh(db_config)
        
        return db_config
    
    @staticmethod
    def delete_config(db: Session, config_id: str) -> bool:
        """
        Delete an embedding configuration.
        
        Args:
            db: Database session
            config_id: Embedding configuration ID
            
        Returns:
            True if deleted, False if not found
        """
        # Get existing config
        db_config = EmbeddingConfigService.get_config(db, config_id)
        if not db_config:
            return False
        
        # Cannot delete active config
        if db_config.is_active:
            raise ValueError("Cannot delete active configuration")
        
        # Delete config
        db.delete(db_config)
        db.commit()
        
        return True
    
    @staticmethod
    def set_active_config(db: Session, config_id: str) -> Optional[EmbeddingConfig]:
        """
        Set an embedding configuration as active.
        
        Args:
            db: Database session
            config_id: Embedding configuration ID
            
        Returns:
            Activated embedding configuration or None if not found
        """
        # Import datetime for explicit datetime fields
        from datetime import datetime
        
        # Deactivate all configs
        db.execute(
            update(EmbeddingConfig)
            .values(is_active=False)
        )
        
        # Get config to activate
        db_config = EmbeddingConfigService.get_config(db, config_id)
        if not db_config:
            return None
        
        # Activate config
        db_config.is_active = True
        
        # Explicitly set updated_at to ensure it's not None
        db_config.updated_at = datetime.now(UTC)
        
        # Ensure created_at is set if it's None
        if db_config.created_at is None:
            db_config.created_at = datetime.now(UTC)
            
        db.commit()
        db.refresh(db_config)
        
        return db_config
    
    @staticmethod
    def create_default_config_if_needed(db: Session) -> Optional[EmbeddingConfig]:
        """
        Create a default embedding configuration if none exists.
        
        Args:
            db: Database session
            
        Returns:
            Created embedding configuration or None if already exists
        """
        # Check if any config exists
        existing_configs = db.query(EmbeddingConfig).count()
        if existing_configs > 0:
            return None
        
        # Create default config
        default_config = EmbeddingConfigCreate(
            provider=settings.DEFAULT_LLM_PROVIDER,
            model=settings.DEFAULT_EMBEDDING_MODEL
        )
        
        # Create and activate config
        db_config = EmbeddingConfigService.create_config(db, default_config)
        EmbeddingConfigService.set_active_config(db, db_config.id)
        
        logger.info(f"Created default embedding configuration with provider {db_config.provider} and model {db_config.model}")
        
        return db_config