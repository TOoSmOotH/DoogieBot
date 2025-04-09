from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os
import logging
from pathlib import Path
from sqlalchemy.orm import Session
from app.db.base import get_db, init_db
from app.core.config import settings
from app.services.user import UserService
from app.services.llm_config import LLMConfigService
from app.rag.singleton import rag_singleton
from app.utils.middleware import TrailingSlashMiddleware
from contextlib import asynccontextmanager

# Import the session manager instance
from app.services.mcp_config_service.manager import mcp_session_manager

# Setup logger
logger = logging.getLogger(__name__)

# Create the app directory if it doesn't exist
app_dir = Path(__file__).parent / "app"
app_dir.mkdir(exist_ok=True)

# Create the uploads directory if it doesn't exist
uploads_dir = Path(__file__).parent.parent / "uploads"
uploads_dir.mkdir(exist_ok=True)

# Global flag to track if RAG singleton has been initialized
_rag_initialized = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic (before yield)
    global _rag_initialized
    
    # Ensure database tables exist
    try:
        # Initialize the database (create tables if they don't exist)
        # init_db() # Commented out: Alembic handles schema creation via migrations
        logger.info("Database tables initialized")
    except Exception as e:
        logger.error(f"Error initializing database tables: {str(e)}")
        logger.warning("This may be due to database connection issues. Will continue and try again later.")
    
    # Get DB session
    db = next(get_db())
    
    try:
        # Create first admin user if credentials are provided
        if settings.FIRST_ADMIN_EMAIL and settings.FIRST_ADMIN_PASSWORD:
            admin_email = settings.FIRST_ADMIN_EMAIL
            admin_password = settings.FIRST_ADMIN_PASSWORD
            
            # Try to create tables directly if they don't exist
            try:
                # Create users table directly with SQL if it doesn't exist
                import sqlite3
                conn = sqlite3.connect('./db/doogie.db')
                cursor = conn.cursor()
                
                # Check if users table exists
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users';")
                if not cursor.fetchone():
                    print("Creating users table manually...")
                    cursor.execute("""
                    CREATE TABLE users (
                        id VARCHAR PRIMARY KEY,
                        email VARCHAR NOT NULL UNIQUE,
                        hashed_password VARCHAR NOT NULL,
                        role VARCHAR(5) NOT NULL,
                        status VARCHAR(8) NOT NULL,
                        theme_preference VARCHAR NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
                        last_login TIMESTAMP
                    );
                    """)
                    conn.commit()
                    logger.info("Users table created manually.")
                
                conn.close()
            except Exception as e:
                logger.error(f"Error creating users table manually: {str(e)}")
            
            try:
                admin_user = UserService.create_first_admin(db, admin_email, admin_password)
                if admin_user:
                    logger.info(f"Created first admin user: {admin_email}")
                else:
                    logger.info("Admin user already exists, skipping creation")
            except Exception as e:
                logger.error(f"Error creating admin user: {str(e)}")
                logger.warning("Will create admin user on first successful database connection.")
        else:
            logger.info("Admin credentials not provided, skipping admin user creation")
        
        # Create default LLM configuration if needed
        try:
            default_config = LLMConfigService.create_default_config_if_needed(db)
            if default_config:
                logger.info(f"Created default LLM configuration with provider: {default_config.provider}")
            else:
                logger.info("LLM configuration already exists, skipping creation")
        except Exception as e:
            logger.error(f"Error creating default LLM configuration: {str(e)}")
            logger.warning("Will create default LLM configuration on first successful database connection.")
    except Exception as e:
        # If there's an error (like missing tables), log it but continue
        logger.error(f"Error during startup initialization: {str(e)}")
        logger.warning("This may be due to database not being fully initialized yet. The application will retry on first request.")
    
    # Initialize RAG singleton only if not already initialized
    if not _rag_initialized:
        logger.info("Initializing RAG components (first worker)...")
        # Set the flag before initializing to prevent other workers from initializing
        _rag_initialized = True
        # Initialize with minimal loading - we'll load components on demand
        try:
            # Don't actually load the components yet, just set up the singleton
            # The actual loading will happen when the components are first accessed
            logger.info("RAG singleton initialized, components will be loaded on demand")
        except Exception as e:
            logger.error(f"Error initializing RAG singleton: {str(e)}")
    else:
        logger.info("RAG singleton already initialized by another worker")

    # --- Start enabled MCP servers ---
    try:
        # Import functions directly from the new package
        from app.services.mcp_config_service import get_all_enabled_configs, start_server
        logger.info("Checking for enabled MCP servers to start...") # Use logger
        # Use the new method to get only enabled configs
        enabled_mcp_configs = get_all_enabled_configs(db) # Call function directly
        logger.info(f"Found {len(enabled_mcp_configs)} enabled MCP configurations.") # Use logger
        started_count = 0
        # Iterate through only the enabled configs
        for config in enabled_mcp_configs:
            # No need to check 'enabled' again here
            logger.info(f"Attempting to start enabled MCP server: {config.name} (ID: {config.id})") # Use logger
            try:
                status = start_server(db, config.id) # Call function directly
                if status and status.status == "running":
                    logger.info(f"Successfully started/verified MCP server: {config.name}") # Use logger
                    started_count += 1
                else:
                    error_msg = status.error_message if status else 'N/A'
                    status_msg = status.status if status else 'Unknown'
                    logger.error(f"Failed to start MCP server {config.name}. Status: {status_msg}, Error: {error_msg}") # Use logger
            except Exception as mcp_start_err:
                logger.exception(f"Error attempting to start MCP server {config.name}: {mcp_start_err}") # Use logger with exception info
        logger.info(f"Finished MCP server startup check. Started/Verified {started_count} servers.") # Use logger
    except Exception as e:
        logger.exception(f"Error during MCP server startup check: {e}") # Use logger with exception info
    finally:
        db.close() # Ensure the session used for startup is closed
    # --- End MCP server startup ---

    yield  # This is where the app runs

    # Shutdown logic (after yield)
    logger.info("Starting application shutdown sequence...")
    # Close all managed MCP sessions
    try:
        logger.info("Closing MCP sessions...")
        await mcp_session_manager.close_all_sessions()
        logger.info("MCP sessions closed.")
    except Exception as e:
        logger.exception(f"Error during MCP session shutdown: {e}")
    # Add any other cleanup code here if needed
    logger.info("Application shutdown sequence complete.")

# Create the FastAPI app
app = FastAPI(
    title="Doogie Chat Bot API",
    description="API for Doogie Chat Bot with Hybrid RAG system",
    version="0.1.0",
    # Disable automatic redirects for trailing slashes
    # This ensures URLs work consistently with or without trailing slashes
    redirect_slashes=False,
    # Add the lifespan context manager
    lifespan=lifespan,
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["Authorization", "Content-Type", "Accept"],
)

# Add trailing slash middleware
app.add_middleware(TrailingSlashMiddleware)

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to Doogie Chat Bot API"}

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Include API router
from app.api.api import api_router
app.include_router(api_router, prefix=settings.API_V1_STR)

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    logging.error(f"HTTP error: {exc.status_code} - {exc.detail} - Path: {request.url.path}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logging.error(f"Unhandled exception: {str(exc)} - Path: {request.url.path}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"},
    )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)