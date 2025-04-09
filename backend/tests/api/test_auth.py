import pytest
from typing import AsyncGenerator, Dict, Any
from httpx import AsyncClient
from sqlalchemy.orm import Session
from fastapi import status
import time

from app.core.config import settings
from app.models.user import User, UserStatus
from app.services.user import UserService
from app.utils.security import decode_token # For checking token contents if needed
from .utils import random_email, random_lower_string

# Use the constants defined in conftest if available, otherwise define here
TEST_USER_EMAIL = "test@example.com"
TEST_USER_PASSWORD = "testpassword"
ADMIN_USER_EMAIL = "admin@example.com"
ADMIN_USER_PASSWORD = "adminpassword"

pytestmark = pytest.mark.asyncio

# --- Test /register ---

async def test_register_success(client: AsyncClient, db: Session) -> None:
    """Test successful user registration."""
    email = random_email()
    password = random_lower_string()
    data = {"email": email, "password": password}
    response = await client.post(f"{settings.API_V1_STR}/auth/register", json=data)
    
    assert response.status_code == status.HTTP_200_OK
    content = response.json()
    assert content["email"] == email
    assert "id" in content
    assert content["status"] == UserStatus.PENDING.value # Default status is PENDING
    assert content["role"] == "user" # Default role
    
    # Verify user exists in DB
    user = UserService.get_by_email(db, email=email)
    assert user is not None
    assert user.email == email
    assert user.status == UserStatus.PENDING

async def test_register_existing_email(client: AsyncClient, test_user: User) -> None:
    """Test registration with an email that already exists."""
    password = random_lower_string()
    data = {"email": TEST_USER_EMAIL, "password": password} # Use existing test_user email
    response = await client.post(f"{settings.API_V1_STR}/auth/register", json=data)
    
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    content = response.json()
    assert "detail" in content
    assert "already exists" in content["detail"]

# --- Test /login ---

async def test_login_success(client: AsyncClient, test_user: User) -> None:
    """Test successful login for an active user."""
    # Ensure user is active (fixture should handle this, but double-check)
    assert test_user.status == UserStatus.ACTIVE
    
    login_data = {"username": TEST_USER_EMAIL, "password": TEST_USER_PASSWORD}
    response = await client.post(f"{settings.API_V1_STR}/auth/login", data=login_data)
    
    assert response.status_code == status.HTTP_200_OK
    content = response.json()
    assert "access_token" in content
    assert "refresh_token" in content
    assert content["token_type"] == "bearer"
    
    # Optional: Decode token to verify subject (user ID)
    access_token_payload = decode_token(content["access_token"])
    assert access_token_payload["sub"] == test_user.id
    assert not access_token_payload.get("refresh", False) # Should be an access token

    refresh_token_payload = decode_token(content["refresh_token"])
    assert refresh_token_payload["sub"] == test_user.id
    assert refresh_token_payload.get("refresh", False) # Should be a refresh token

async def test_login_incorrect_password(client: AsyncClient, test_user: User) -> None:
    """Test login with incorrect password."""
    login_data = {"username": TEST_USER_EMAIL, "password": "wrongpassword"}
    response = await client.post(f"{settings.API_V1_STR}/auth/login", data=login_data)
    
    assert response.status_code == status.HTTP_401_UNAUTHORIZED
    content = response.json()
    assert "detail" in content
    assert "Incorrect email or password" in content["detail"]

async def test_login_nonexistent_user(client: AsyncClient) -> None:
    """Test login with an email that does not exist."""
    login_data = {"username": "nonexistent@example.com", "password": "password"}
    response = await client.post(f"{settings.API_V1_STR}/auth/login", data=login_data)
    
    assert response.status_code == status.HTTP_401_UNAUTHORIZED # Or 404 depending on implementation detail
    content = response.json()
    assert "detail" in content
    # Detail might vary, check for common phrases
    assert "Incorrect email or password" in content["detail"] or "User not found" in content["detail"]

async def test_login_pending_user(client: AsyncClient, db: Session) -> None:
    """Test login attempt by a user whose account is pending."""
    email = random_email()
    password = random_lower_string()
    # Create a pending user directly
    user_in = {"email": email, "password": password}
    UserService.create_user(db, user_in=user_in, status=UserStatus.PENDING)
    
    login_data = {"username": email, "password": password}
    response = await client.post(f"{settings.API_V1_STR}/auth/login", data=login_data)
    
    assert response.status_code == status.HTTP_403_FORBIDDEN
    content = response.json()
    assert "detail" in content
    assert "pending approval" in content["detail"]

async def test_login_inactive_user(client: AsyncClient, db: Session) -> None:
    """Test login attempt by a user whose account is inactive."""
    email = random_email()
    password = random_lower_string()
    # Create an inactive user directly
    user_in = {"email": email, "password": password}
    UserService.create_user(db, user_in=user_in, status=UserStatus.INACTIVE)
    
    login_data = {"username": email, "password": password}
    response = await client.post(f"{settings.API_V1_STR}/auth/login", data=login_data)
    
    assert response.status_code == status.HTTP_403_FORBIDDEN
    content = response.json()
    assert "detail" in content
    assert "deactivated" in content["detail"]

async def test_login_remember_me(client: AsyncClient, test_user: User) -> None:
    """Test login with remember_me=True potentially affects token expiry."""
    # Note: Precisely testing expiry requires mocking time or checking token 'exp'
    # Here, we just verify tokens are generated for both cases.
    
    # Case 1: remember_me = False (default)
    login_data_no_remember = {"username": TEST_USER_EMAIL, "password": TEST_USER_PASSWORD}
    response_no_remember = await client.post(f"{settings.API_V1_STR}/auth/login", data=login_data_no_remember)
    assert response_no_remember.status_code == status.HTTP_200_OK
    content_no_remember = response_no_remember.json()
    assert "access_token" in content_no_remember
    assert "refresh_token" in content_no_remember
    
    # Case 2: remember_me = True
    login_data_remember = {"username": TEST_USER_EMAIL, "password": TEST_USER_PASSWORD}
    # Pass remember_me as a query parameter
    response_remember = await client.post(
        f"{settings.API_V1_STR}/auth/login?remember_me=true",
        data=login_data_remember
    )
    assert response_remember.status_code == status.HTTP_200_OK
    content_remember = response_remember.json()
    assert "access_token" in content_remember
    assert "refresh_token" in content_remember
    
    # Basic check: Ensure tokens are different (highly likely due to timestamp)
    assert content_no_remember["access_token"] != content_remember["access_token"]
    assert content_no_remember["refresh_token"] != content_remember["refresh_token"]

# --- Test /refresh ---

async def test_refresh_token_success(client: AsyncClient, test_user: User) -> None:
    """Test refreshing tokens with a valid refresh token."""
    # 1. Login to get initial tokens
    login_data = {"username": TEST_USER_EMAIL, "password": TEST_USER_PASSWORD}
    login_response = await client.post(f"{settings.API_V1_STR}/auth/login", data=login_data)
    assert login_response.status_code == status.HTTP_200_OK
    initial_tokens = login_response.json()
    initial_refresh_token = initial_tokens["refresh_token"]
    initial_access_token = initial_tokens["access_token"]

    # 2. Use the refresh token to get new tokens
    refresh_data = {"refresh_token": initial_refresh_token}
    refresh_response = await client.post(f"{settings.API_V1_STR}/auth/refresh", json=refresh_data)
    
    assert refresh_response.status_code == status.HTTP_200_OK
    new_tokens = refresh_response.json()
    assert "access_token" in new_tokens
    assert "refresh_token" in new_tokens
    assert new_tokens["token_type"] == "bearer"
    
    # Ensure new tokens are different from the initial ones
    assert new_tokens["access_token"] != initial_access_token
    # The refresh token might or might not be rotated depending on strategy,
    # but the access token MUST be new.
    
    # Verify the new access token is valid for the user
    new_access_token_payload = decode_token(new_tokens["access_token"])
    assert new_access_token_payload["sub"] == test_user.id
    assert not new_access_token_payload.get("refresh", False)

async def test_refresh_token_invalid(client: AsyncClient) -> None:
    """Test refreshing with an invalid or malformed token."""
    refresh_data = {"refresh_token": "this.is.invalid"}
    response = await client.post(f"{settings.API_V1_STR}/auth/refresh", json=refresh_data)
    
    assert response.status_code == status.HTTP_401_UNAUTHORIZED
    content = response.json()
    assert "detail" in content
    assert "Invalid refresh token" in content["detail"]

async def test_refresh_token_expired(client: AsyncClient, test_user: User) -> None:
    """Test refreshing with an expired refresh token (requires mocking or specific token creation)."""
    # This is hard to test reliably without time mocking.
    # We can simulate by creating a token known to be expired if the utility allows.
    # For now, we'll skip the precise expiry test and rely on the 'invalid' test case.
    # If a time-mocking library (like freezegun) is added, this test can be implemented.
    pass

async def test_refresh_token_inactive_user(client: AsyncClient, db: Session, test_user: User) -> None:
    """Test refreshing token when the associated user is inactive."""
     # 1. Login to get initial tokens while user is active
    login_data = {"username": TEST_USER_EMAIL, "password": TEST_USER_PASSWORD}
    login_response = await client.post(f"{settings.API_V1_STR}/auth/login", data=login_data)
    assert login_response.status_code == status.HTTP_200_OK
    initial_tokens = login_response.json()
    initial_refresh_token = initial_tokens["refresh_token"]

    # 2. Deactivate the user
    UserService.deactivate_user(db, test_user)
    assert test_user.status == UserStatus.INACTIVE

    # 3. Attempt to refresh the token
    refresh_data = {"refresh_token": initial_refresh_token}
    refresh_response = await client.post(f"{settings.API_V1_STR}/auth/refresh", json=refresh_data)

    assert refresh_response.status_code == status.HTTP_403_FORBIDDEN
    content = refresh_response.json()
    assert "detail" in content
    assert "Inactive user account" in content["detail"]

async def test_refresh_using_access_token(client: AsyncClient, test_user_token_headers: Dict[str, str]) -> None:
    """Test attempting to refresh using an access token instead of a refresh token."""
    access_token = test_user_token_headers["Authorization"].split(" ")[1]
    refresh_data = {"refresh_token": access_token}
    response = await client.post(f"{settings.API_V1_STR}/auth/refresh", json=refresh_data)

    # Expecting unauthorized because the token payload lacks the 'refresh: true' claim
    assert response.status_code == status.HTTP_401_UNAUTHORIZED
    content = response.json()
    assert "detail" in content
    assert "Invalid refresh token" in content["detail"]


# --- Test /check ---

async def test_check_auth_success(client: AsyncClient, test_user_token_headers: Dict[str, str], test_user: User) -> None:
    """Test checking authentication status when logged in."""
    response = await client.get(f"{settings.API_V1_STR}/auth/check", headers=test_user_token_headers)
    
    assert response.status_code == status.HTTP_200_OK
    content = response.json()
    assert content["authenticated"] is True
    assert content["user_id"] == test_user.id
    assert content["email"] == test_user.email

async def test_check_auth_no_token(client: AsyncClient) -> None:
    """Test checking authentication status without providing a token."""
    response = await client.get(f"{settings.API_V1_STR}/auth/check")
    
    assert response.status_code == status.HTTP_401_UNAUTHORIZED
    content = response.json()
    assert "detail" in content
    assert "Not authenticated" in content["detail"] # Or similar message from get_current_user dependency

async def test_check_auth_invalid_token(client: AsyncClient) -> None:
    """Test checking authentication status with an invalid token."""
    headers = {"Authorization": "Bearer invalidtoken"}
    response = await client.get(f"{settings.API_V1_STR}/auth/check", headers=headers)
    
    assert response.status_code == status.HTTP_401_UNAUTHORIZED
    content = response.json()
    assert "detail" in content
    # Detail might vary based on JWTError or token validation logic
    assert "Invalid token" in content["detail"] or "Could not validate credentials" in content["detail"]