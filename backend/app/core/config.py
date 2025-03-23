import logging
import os
from typing import Any, Dict, List, Optional, Union

from pydantic import AnyHttpUrl, EmailStr, field_validator
from pydantic_settings import BaseSettings

# Configure logging
logging.basicConfig(
    level=(
        logging.DEBUG if os.getenv("DEBUG", "true").lower() == "true" else logging.INFO
    ),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Configure SQLAlchemy logging separately
sqlalchemy_logger = logging.getLogger("sqlalchemy.engine")
sqlalchemy_logger.setLevel(
    logging.INFO if os.getenv("DISABLE_SQL_LOGS", "false").lower() == "true" else None
)


class Settings(BaseSettings):
    # API settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Doogie Chat Bot"

    # Debug mode
    DEBUG: bool = os.getenv("DEBUG", "true").lower() == "true"
    DISABLE_SQL_LOGS: bool = os.getenv("DISABLE_SQL_LOGS", "false").lower() == "true"
    LLM_DEBUG_LOGGING: bool = os.getenv("LLM_DEBUG_LOGGING", "false").lower() == "true"

    # CORS settings
    BACKEND_CORS_ORIGINS: List[AnyHttpUrl] = []

    @field_validator("BACKEND_CORS_ORIGINS", mode="before")
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> Union[List[str], str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)

    # Security settings
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
    ALGORITHM: str = os.getenv("ALGORITHM", "HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(
        os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30")
    )
    REFRESH_TOKEN_EXPIRE_DAYS: int = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))

    # Admin user settings
    FIRST_ADMIN_EMAIL: EmailStr = os.getenv("FIRST_ADMIN_EMAIL", "admin@example.com")
    FIRST_ADMIN_PASSWORD: str = os.getenv(
        "FIRST_ADMIN_PASSWORD", "change-this-password"
    )

    # Database settings
    SQLITE_DATABASE_URL: str = "sqlite:///./doogie.db"

    # LLM service settings
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    OPENROUTER_API_KEY: Optional[str] = os.getenv("OPENROUTER_API_KEY")
    DEEPSEEK_API_KEY: Optional[str] = os.getenv("DEEPSEEK_API_KEY")
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    LM_STUDIO_BASE_URL: str = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:8000")

    # GitHub API settings
    GITHUB_API_TOKEN: Optional[str] = os.getenv("GITHUB_API_TOKEN")

    # Default LLM settings
    DEFAULT_LLM_PROVIDER: str = (
        "openai"  # Options: openai, anthropic, openrouter, deepseek, ollama, lmstudio
    )
    DEFAULT_CHAT_MODEL: str = "gpt-3.5-turbo"
    DEFAULT_EMBEDDING_MODEL: str = "text-embedding-ada-002"
    DEFAULT_SYSTEM_PROMPT: str = (
        "You are Doogie, a helpful AI assistant."  # Global system prompt for all LLM providers
    )

    # RAG settings
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    RAG_INDEX_BUILD_TIMEOUT: int = int(
        os.getenv("RAG_INDEX_BUILD_TIMEOUT", "3600")
    )  # 1 hour default timeout

    # Upload settings
    UPLOAD_DIR: str = "../uploads"
    MAX_UPLOAD_SIZE: int = 50 * 1024 * 1024  # 50 MB

    model_config = {"case_sensitive": True, "env_file": ".env"}


settings = Settings()

# Set debug mode for loggers if DEBUG is True
if settings.DEBUG:
    for logger_name in logging.root.manager.loggerDict:
        logger = logging.getLogger(logger_name)
        # Skip SQLAlchemy loggers if SQL logs are disabled
        if settings.DISABLE_SQL_LOGS and logger_name.startswith("sqlalchemy"):
            continue
        logger.setLevel(logging.DEBUG)
    logging.debug("Debug logging enabled")
