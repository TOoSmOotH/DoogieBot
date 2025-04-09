import pytest
import asyncio
from typing import AsyncGenerator, Generator, Any
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from httpx import AsyncClient
from fastapi.testclient import TestClient  # Although we use AsyncClient, TestClient might be needed for setup

# Import the main FastAPI app and settings
from backend.main import app
from app.core.config import settings
from app.db.base import Base, get_db
from app.models.user import User, UserRole, UserStatus  # Import User model
from app.services.user import UserService  # Import UserService
from app.utils.security import create_access_token # Import token creation utility

# Use an in-memory SQLite database for testing
SQLALCHEMY_DATABASE_URL = "sqlite+aiosqlite:///:memory:" # Use aiosqlite for async

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False}, # Required for SQLite
    pool_pre_ping=True,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Apply migrations or create tables for the test database
# Since we use in-memory, create tables directly
Base.metadata.create_all(bind=engine)

@pytest.fixture(scope="session")
def event_loop(request: Any) -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an instance of the default event loop for each test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="function")
def db() -> Generator[Session, None, None]:
    """Fixture to provide a test database session."""
    connection = engine.connect()
    # Begin a non-ORM transaction
    transaction = connection.begin()
    # Bind an individual Session to the connection
    db_session = TestingSessionLocal(bind=connection)
    yield db_session
    # Rollback the transaction after the test is done
    db_session.close()
    transaction.rollback()
    connection.close()

@pytest.fixture(scope="function")
async def override_get_db(db: Session) -> AsyncGenerator[Session, None]:
    """Fixture to override the get_db dependency in routes."""
    yield db

@pytest.fixture(scope="function")
async def client(override_get_db: Session) -> AsyncGenerator[AsyncClient, None]:
    """Fixture to provide an httpx.AsyncClient for integration tests."""
    # Override the dependency
    app.dependency_overrides[get_db] = lambda: override_get_db
    async with AsyncClient(app=app, base_url="http://test") as async_client:
        yield async_client
    # Clean up dependency overrides
    app.dependency_overrides.clear()


# --- User and Auth Fixtures ---

TEST_USER_EMAIL = "test@example.com"
TEST_USER_PASSWORD = "testpassword"
ADMIN_USER_EMAIL = "admin@example.com"
ADMIN_USER_PASSWORD = "adminpassword"

@pytest.fixture(scope="function")
def test_user(db: Session) -> User:
    """Fixture to create a standard test user."""
    user = UserService.get_by_email(db, email=TEST_USER_EMAIL)
    if not user:
        user_in = {"email": TEST_USER_EMAIL, "password": TEST_USER_PASSWORD}
        user = UserService.create_user(db, user_in=user_in, status=UserStatus.ACTIVE) # Create active user for tests
    # Ensure user is active for tests that require login
    if user.status != UserStatus.ACTIVE:
        user = UserService.activate_user(db, user)
    return user

@pytest.fixture(scope="function")
def admin_user(db: Session) -> User:
    """Fixture to create an admin test user."""
    user = UserService.get_by_email(db, email=ADMIN_USER_EMAIL)
    if not user:
        user_in = {"email": ADMIN_USER_EMAIL, "password": ADMIN_USER_PASSWORD}
        user = UserService.create_user(db, user_in=user_in, role=UserRole.ADMIN, status=UserStatus.ACTIVE)
    # Ensure user is active and admin
    if user.status != UserStatus.ACTIVE:
        user = UserService.activate_user(db, user)
    if user.role != UserRole.ADMIN:
        user.role = UserRole.ADMIN
        db.commit()
        db.refresh(user)
    return user

@pytest.fixture(scope="function")
def test_user_token_headers(test_user: User) -> dict[str, str]:
    """Fixture to generate auth headers for the standard test user."""
    access_token = create_access_token(subject=test_user.id)
    return {"Authorization": f"Bearer {access_token}"}

@pytest.fixture(scope="function")
def admin_user_token_headers(admin_user: User) -> dict[str, str]:
    """Fixture to generate auth headers for the admin test user."""
    access_token = create_access_token(subject=admin_user.id)
    return {"Authorization": f"Bearer {access_token}"}