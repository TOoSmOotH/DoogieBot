from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
import os
import logging

from app.core.config import settings

# Set up logging
logger = logging.getLogger(__name__)

# Create SQLite engine
engine = create_engine(
    settings.SQLITE_DATABASE_URL, 
    connect_args={"check_same_thread": False}  # Needed for SQLite
)

# Create SessionLocal class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create Base class
Base = declarative_base()

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Function to initialize database
def init_db():
    # Import models to register them with SQLAlchemy
    logger.info("Initializing database - importing models...")
    import importlib
    import pkgutil
    import app.models
    
    # Dynamically import all modules in the models package
    for _, name, _ in pkgutil.iter_modules(app.models.__path__, app.models.__name__ + "."):
        try:
            importlib.import_module(name)
            logger.info(f"Imported model module: {name}")
        except ImportError as e:
            logger.warning(f"Failed to import {name}: {e}")
    
    # Now import our init_models module to ensure all models are registered
    try:
        from app.db.init_models import Base as InitModelsBase
        logger.info("Imported models from init_models")
    except ImportError as e:
        logger.warning(f"Failed to import from init_models: {e}")
    
    # Check if database file exists
    db_path = settings.SQLITE_DATABASE_URL.replace('sqlite:///', '')
    logger.info(f"Database path: {db_path}")
    
    # Create tables
    logger.info("Creating database tables...")
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully!")
    except Exception as e:
        logger.error(f"Error creating database tables: {e}")
        raise