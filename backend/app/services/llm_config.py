import logging
from typing import Any, Dict, List, Optional

from app.core.config import settings
from app.llm.factory import LLMFactory
from app.models.llm_config import LLMConfig
from app.schemas.llm import LLMConfigCreate, LLMConfigUpdate
from sqlalchemy import update
from sqlalchemy.orm import Session

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMConfigService:
    """
    Service for managing LLM configurations.
    """

    @staticmethod
    def create_config(db: Session, config: LLMConfigCreate) -> LLMConfig:
        """
        Create a new LLM configuration.

        Args:
            db: Database session
            config: LLM configuration data

        Returns:
            Created LLM configuration
        """
        # Create new config
        db_config = LLMConfig(
            provider=config.provider,
            model=config.model,
            embedding_model=config.embedding_model,
            system_prompt=config.system_prompt,
            api_key=config.api_key,
            base_url=config.base_url,
            config=config.config,
            is_active=False,  # New configs are not active by default
        )

        db.add(db_config)
        db.commit()
        db.refresh(db_config)

        return db_config

    @staticmethod
    def get_config(db: Session, config_id: str) -> Optional[LLMConfig]:
        """
        Get an LLM configuration by ID.

        Args:
            db: Database session
            config_id: LLM configuration ID

        Returns:
            LLM configuration or None if not found
        """
        return db.query(LLMConfig).filter(LLMConfig.id == config_id).first()

    @staticmethod
    def get_active_config(db: Session) -> Optional[LLMConfig]:
        """
        Get the active LLM configuration.

        Args:
            db: Database session

        Returns:
            Active LLM configuration or None if not found
        """
        return db.query(LLMConfig).filter(LLMConfig.is_active == True).first()

    @staticmethod
    def get_all_configs(db: Session) -> List[LLMConfig]:
        """
        Get all LLM configurations.

        Args:
            db: Database session

        Returns:
            List of LLM configurations
        """
        return db.query(LLMConfig).all()

    @staticmethod
    def update_config(
        db: Session, config_id: str, config_update: LLMConfigUpdate
    ) -> Optional[LLMConfig]:
        """
        Update an LLM configuration.

        Args:
            db: Database session
            config_id: LLM configuration ID
            config_update: LLM configuration update data

        Returns:
            Updated LLM configuration or None if not found
        """
        # Get existing config
        db_config = LLMConfigService.get_config(db, config_id)
        if not db_config:
            return None

        # Update fields
        update_data = config_update.model_dump(exclude_unset=True)

        # If setting this config as active, deactivate all other configs
        if update_data.get("is_active", False):
            db.execute(
                update(LLMConfig)
                .where(LLMConfig.id != config_id)
                .values(is_active=False)
            )

        # Update config
        for key, value in update_data.items():
            setattr(db_config, key, value)

        db.commit()
        db.refresh(db_config)

        return db_config

    @staticmethod
    def delete_config(db: Session, config_id: str) -> bool:
        """
        Delete an LLM configuration.

        Args:
            db: Database session
            config_id: LLM configuration ID

        Returns:
            True if deleted, False if not found
        """
        # Get existing config
        db_config = LLMConfigService.get_config(db, config_id)
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
    def set_active_config(db: Session, config_id: str) -> Optional[LLMConfig]:
        """
        Set an LLM configuration as active.

        Args:
            db: Database session
            config_id: LLM configuration ID

        Returns:
            Activated LLM configuration or None if not found
        """
        # Deactivate all configs
        db.execute(update(LLMConfig).values(is_active=False))

        # Get config to activate
        db_config = LLMConfigService.get_config(db, config_id)
        if not db_config:
            return None

        # Activate config
        db_config.is_active = True
        db.commit()
        db.refresh(db_config)

        return db_config

    @staticmethod
    def get_available_providers(db: Session) -> Dict[str, Any]:
        """
        Get available LLM providers.

        Args:
            db: Database session

        Returns:
            Dictionary of available providers
        """
        # Get providers from factory
        providers = LLMFactory.get_available_providers()

        # Get active config
        active_config = LLMConfigService.get_active_config(db)

        # Add active status to providers
        if active_config:
            for provider_id, provider_info in providers.items():
                provider_info["active"] = provider_id == active_config.provider

        return providers

    @staticmethod
    def create_default_config_if_needed(db: Session) -> Optional[LLMConfig]:
        """
        Create a default LLM configuration if none exists.

        Args:
            db: Database session

        Returns:
            Created LLM configuration or None if already exists
        """
        # Check if any config exists
        existing_configs = db.query(LLMConfig).count()
        if existing_configs > 0:
            return None

        # Create default config with global system prompt
        default_config = LLMConfigCreate(
            provider=settings.DEFAULT_LLM_PROVIDER,
            model=settings.DEFAULT_CHAT_MODEL,
            embedding_model=settings.DEFAULT_EMBEDDING_MODEL,
            system_prompt=settings.DEFAULT_SYSTEM_PROMPT,  # Global system prompt for all providers
        )

        # Create and activate config
        db_config = LLMConfigService.create_config(db, default_config)
        LLMConfigService.set_active_config(db, db_config.id)

        logger.info(
            f"Created default LLM configuration with provider {db_config.provider}"
        )

        return db_config
