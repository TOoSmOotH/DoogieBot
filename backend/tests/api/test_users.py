import pytest
from typing import AsyncGenerator, Dict, Any, List
from httpx import AsyncClient
from sqlalchemy.orm import Session
from fastapi import status
import uuid

from app.core.config import settings
from app.models.user import User, UserRole, UserStatus
from app.schemas.user import UserCreate, UserUpdate, UserResponse
from app.services.user import UserService
from .utils import random_email, random_lower_string

# Constants from conftest or define if needed
TEST_USER_EMAIL = "test@example.com"
TEST_USER_PASSWORD = "testpassword"
ADMIN_USER_EMAIL = "admin@example.com"
ADMIN_USER_PASSWORD = "adminpassword"

pytestmark = pytest.mark.asyncio

# --- Test /me ---

async def test_read_current_user_me(
    client: AsyncClient, test_user_token_headers: Dict[str, str], test_user: User
) -> None:
    """Test getting the current user's details."""
    response = await client.get(f"{settings.API_V1_STR}/users/me", headers=test_user_token_headers)
    
    assert response.status_code == status.HTTP_200_OK
    content = response.json()
    assert content["email"] == test_user.email
    assert content["id"] == test_user.id
    assert content["status"] == test_user.status.value
    assert content["role"] == test_user.role.value

async def test_read_current_user_me_unauthenticated(client: AsyncClient) -> None:
    """Test getting /me without authentication."""
    response = await client.get(f"{settings.API_V1_STR}/users/me")
    assert response.status_code == status.HTTP_401_UNAUTHORIZED

async def test_update_current_user_me_success(
    client: AsyncClient, test_user_token_headers: Dict[str, str], test_user: User, db: Session
) -> None:
    """Test successfully updating allowed fields for the current user."""
    new_password = "newpassword123"
    update_data = {"password": new_password} # Only password update is shown here, add others if allowed
    
    response = await client.put(f"{settings.API_V1_STR}/users/me", headers=test_user_token_headers, json=update_data)
    
    assert response.status_code == status.HTTP_200_OK
    content = response.json()
    assert content["email"] == test_user.email # Email shouldn't change unless explicitly updated and allowed
    
    # Verify password was updated in the database
    db.refresh(test_user) # Refresh user object from DB
    assert UserService.verify_password(new_password, test_user.hashed_password)

async def test_update_current_user_me_forbidden_fields(
    client: AsyncClient, test_user_token_headers: Dict[str, str], test_user: User
) -> None:
    """Test attempting to update forbidden fields (role, status) via /me."""
    update_data_role = {"role": UserRole.ADMIN.value}
    response_role = await client.put(f"{settings.API_V1_STR}/users/me", headers=test_user_token_headers, json=update_data_role)
    assert response_role.status_code == status.HTTP_403_FORBIDDEN
    assert "Not allowed to update role or status" in response_role.json()["detail"]

    update_data_status = {"status": UserStatus.INACTIVE.value}
    response_status = await client.put(f"{settings.API_V1_STR}/users/me", headers=test_user_token_headers, json=update_data_status)
    assert response_status.status_code == status.HTTP_403_FORBIDDEN
    assert "Not allowed to update role or status" in response_status.json()["detail"]

async def test_update_current_user_me_unauthenticated(client: AsyncClient) -> None:
    """Test updating /me without authentication."""
    update_data = {"password": "newpassword"}
    response = await client.put(f"{settings.API_V1_STR}/users/me", json=update_data)
    assert response.status_code == status.HTTP_401_UNAUTHORIZED

# --- Test Admin User Routes ---

# Helper to create multiple users for pagination tests
def create_multiple_users(db: Session, count: int, status: UserStatus = UserStatus.ACTIVE) -> List[User]:
    users = []
    for i in range(count):
        email = f"testuser{i+100}@example.com" # Avoid collision with standard fixtures
        password = "password"
        user_in = {"email": email, "password": password}
        user = UserService.create_user(db, user_in=user_in, status=status)
        users.append(user)
    return users

# --- Test GET /users (Admin) ---

async def test_read_users_admin_success(
    client: AsyncClient, admin_user_token_headers: Dict[str, str], db: Session, admin_user: User, test_user: User
) -> None:
    """Test retrieving users list as admin."""
    # Ensure at least admin and test_user exist
    response = await client.get(f"{settings.API_V1_STR}/users", headers=admin_user_token_headers)
    
    assert response.status_code == status.HTTP_200_OK
    content = response.json()
    assert "items" in content
    assert "total" in content
    assert "page" in content
    assert "size" in content
    assert "pages" in content
    
    assert content["total"] >= 2 # Should include at least admin and test_user
    assert len(content["items"]) > 0
    # Check structure of one item
    user_item = content["items"][0]
    assert "id" in user_item
    assert "email" in user_item
    assert "role" in user_item
    assert "status" in user_item
    assert "is_active" in user_item # Frontend compatibility field
    assert "is_admin" in user_item # Frontend compatibility field
    assert "hashed_password" not in user_item # Ensure sensitive data isn't returned

async def test_read_users_admin_pagination(
    client: AsyncClient, admin_user_token_headers: Dict[str, str], db: Session
) -> None:
    """Test pagination for retrieving users list."""
    # Create more users to test pagination
    create_multiple_users(db, 15)
    total_users = UserService.count_users(db) # Get total count including fixtures

    # Request page 1, size 5
    response_page1 = await client.get(f"{settings.API_V1_STR}/users?page=1&size=5", headers=admin_user_token_headers)
    assert response_page1.status_code == status.HTTP_200_OK
    content_page1 = response_page1.json()
    assert len(content_page1["items"]) == 5
    assert content_page1["page"] == 1
    assert content_page1["size"] == 5
    assert content_page1["total"] == total_users
    
    # Request page 2, size 5
    response_page2 = await client.get(f"{settings.API_V1_STR}/users?page=2&size=5", headers=admin_user_token_headers)
    assert response_page2.status_code == status.HTTP_200_OK
    content_page2 = response_page2.json()
    assert len(content_page2["items"]) > 0 # Should have items on page 2
    assert content_page2["page"] == 2
    
    # Ensure items are different between pages
    ids_page1 = {item["id"] for item in content_page1["items"]}
    ids_page2 = {item["id"] for item in content_page2["items"]}
    assert not ids_page1.intersection(ids_page2)

async def test_read_users_admin_filter_status(
    client: AsyncClient, admin_user_token_headers: Dict[str, str], db: Session
) -> None:
    """Test filtering users list by status."""
    # Create some pending users
    create_multiple_users(db, 3, status=UserStatus.PENDING)
    active_count = UserService.count_users(db, status=UserStatus.ACTIVE)
    pending_count = UserService.count_users(db, status=UserStatus.PENDING)
    
    # Filter by ACTIVE
    response_active = await client.get(f"{settings.API_V1_STR}/users?status=active", headers=admin_user_token_headers)
    assert response_active.status_code == status.HTTP_200_OK
    content_active = response_active.json()
    assert content_active["total"] == active_count
    assert all(item["status"] == UserStatus.ACTIVE.value for item in content_active["items"])
    
    # Filter by PENDING
    response_pending = await client.get(f"{settings.API_V1_STR}/users?status=pending", headers=admin_user_token_headers)
    assert response_pending.status_code == status.HTTP_200_OK
    content_pending = response_pending.json()
    assert content_pending["total"] == pending_count
    assert all(item["status"] == UserStatus.PENDING.value for item in content_pending["items"])

async def test_read_users_non_admin(
    client: AsyncClient, test_user_token_headers: Dict[str, str]
) -> None:
    """Test retrieving users list as a non-admin user."""
    response = await client.get(f"{settings.API_V1_STR}/users", headers=test_user_token_headers)
    assert response.status_code == status.HTTP_403_FORBIDDEN

async def test_read_users_unauthenticated(client: AsyncClient) -> None:
    """Test retrieving users list without authentication."""
    response = await client.get(f"{settings.API_V1_STR}/users")
    assert response.status_code == status.HTTP_401_UNAUTHORIZED

# --- Test POST /users (Admin) ---

async def test_create_user_admin_success(
    client: AsyncClient, admin_user_token_headers: Dict[str, str], db: Session
) -> None:
    """Test creating a new user as admin."""
    email = random_email()
    password = random_lower_string()
    data = {"email": email, "password": password}
    
    response = await client.post(f"{settings.API_V1_STR}/users", headers=admin_user_token_headers, json=data)
    
    assert response.status_code == status.HTTP_200_OK
    content = response.json()
    assert content["email"] == email
    assert "id" in content
    assert content["status"] == UserStatus.ACTIVE.value # Admin creates active users by default
    assert content["role"] == UserRole.USER.value # Default role
    
    # Verify user exists in DB
    user = UserService.get_by_email(db, email=email)
    assert user is not None
    assert user.email == email
    assert user.status == UserStatus.ACTIVE

async def test_create_user_admin_existing_email(
    client: AsyncClient, admin_user_token_headers: Dict[str, str], test_user: User
) -> None:
    """Test creating a user with an existing email as admin."""
    password = random_lower_string()
    data = {"email": test_user.email, "password": password}
    
    response = await client.post(f"{settings.API_V1_STR}/users", headers=admin_user_token_headers, json=data)
    
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    content = response.json()
    assert "detail" in content
    assert "already exists" in content["detail"]

async def test_create_user_non_admin(
    client: AsyncClient, test_user_token_headers: Dict[str, str]
) -> None:
    """Test creating a user as a non-admin."""
    email = random_email()
    password = random_lower_string()
    data = {"email": email, "password": password}
    response = await client.post(f"{settings.API_V1_STR}/users", headers=test_user_token_headers, json=data)
    assert response.status_code == status.HTTP_403_FORBIDDEN

async def test_create_user_unauthenticated(client: AsyncClient) -> None:
    """Test creating a user without authentication."""
    email = random_email()
    password = random_lower_string()
    data = {"email": email, "password": password}
    response = await client.post(f"{settings.API_V1_STR}/users", json=data)
    assert response.status_code == status.HTTP_401_UNAUTHORIZED

# --- Test GET /users/pending (Admin) ---

async def test_read_pending_users_admin_success(
    client: AsyncClient, admin_user_token_headers: Dict[str, str], db: Session
) -> None:
    """Test retrieving pending users list as admin."""
    # Create some pending users
    pending_users = create_multiple_users(db, 3, status=UserStatus.PENDING)
    pending_count = len(pending_users)
    
    response = await client.get(f"{settings.API_V1_STR}/users/pending", headers=admin_user_token_headers)
    
    assert response.status_code == status.HTTP_200_OK
    content = response.json()
    assert "items" in content
    assert content["total"] == pending_count
    assert len(content["items"]) == pending_count # Assuming default size >= 3
    assert all(item["status"] == UserStatus.PENDING.value for item in content["items"])

async def test_read_pending_users_non_admin(
    client: AsyncClient, test_user_token_headers: Dict[str, str]
) -> None:
    """Test retrieving pending users as non-admin."""
    response = await client.get(f"{settings.API_V1_STR}/users/pending", headers=test_user_token_headers)
    assert response.status_code == status.HTTP_403_FORBIDDEN

async def test_read_pending_users_unauthenticated(client: AsyncClient) -> None:
    """Test retrieving pending users without authentication."""
    response = await client.get(f"{settings.API_V1_STR}/users/pending")
    assert response.status_code == status.HTTP_401_UNAUTHORIZED

# --- Test GET /users/{user_id} (Admin) ---

async def test_read_user_admin_success(
    client: AsyncClient, admin_user_token_headers: Dict[str, str], test_user: User
) -> None:
    """Test retrieving a specific user by ID as admin."""
    response = await client.get(f"{settings.API_V1_STR}/users/{test_user.id}", headers=admin_user_token_headers)
    
    assert response.status_code == status.HTTP_200_OK
    content = response.json()
    assert content["id"] == test_user.id
    assert content["email"] == test_user.email

async def test_read_user_admin_not_found(
    client: AsyncClient, admin_user_token_headers: Dict[str, str]
) -> None:
    """Test retrieving a non-existent user by ID as admin."""
    non_existent_id = str(uuid.uuid4())
    response = await client.get(f"{settings.API_V1_STR}/users/{non_existent_id}", headers=admin_user_token_headers)
    assert response.status_code == status.HTTP_404_NOT_FOUND

async def test_read_user_non_admin(
    client: AsyncClient, test_user_token_headers: Dict[str, str], admin_user: User # Need another user's ID
) -> None:
    """Test retrieving a specific user by ID as non-admin."""
    response = await client.get(f"{settings.API_V1_STR}/users/{admin_user.id}", headers=test_user_token_headers)
    assert response.status_code == status.HTTP_403_FORBIDDEN

async def test_read_user_unauthenticated(client: AsyncClient, test_user: User) -> None:
    """Test retrieving a specific user by ID without authentication."""
    response = await client.get(f"{settings.API_V1_STR}/users/{test_user.id}")
    assert response.status_code == status.HTTP_401_UNAUTHORIZED

# --- Test PUT /users/{user_id} (Admin) ---

async def test_update_user_admin_success(
    client: AsyncClient, admin_user_token_headers: Dict[str, str], test_user: User, db: Session
) -> None:
    """Test updating a user's details as admin."""
    new_email = random_email()
    new_role = UserRole.ADMIN.value
    new_status = UserStatus.INACTIVE.value
    
    update_data = {
        "email": new_email,
        "role": new_role,
        "status": new_status,
        # "password": "newpassword" # Optional password update
    }
    
    response = await client.put(f"{settings.API_V1_STR}/users/{test_user.id}", headers=admin_user_token_headers, json=update_data)
    
    assert response.status_code == status.HTTP_200_OK
    content = response.json()
    assert content["id"] == test_user.id
    assert content["email"] == new_email
    assert content["role"] == new_role
    assert content["status"] == new_status
    assert content["is_active"] is False # Check frontend compatibility field
    assert content["is_admin"] is True  # Check frontend compatibility field
    
    # Verify changes in DB
    db.refresh(test_user)
    assert test_user.email == new_email
    assert test_user.role == UserRole.ADMIN
    assert test_user.status == UserStatus.INACTIVE

async def test_update_user_admin_not_found(
    client: AsyncClient, admin_user_token_headers: Dict[str, str]
) -> None:
    """Test updating a non-existent user as admin."""
    non_existent_id = str(uuid.uuid4())
    update_data = {"email": random_email()}
    response = await client.put(f"{settings.API_V1_STR}/users/{non_existent_id}", headers=admin_user_token_headers, json=update_data)
    assert response.status_code == status.HTTP_404_NOT_FOUND

async def test_update_user_non_admin(
    client: AsyncClient, test_user_token_headers: Dict[str, str], admin_user: User
) -> None:
    """Test updating a user as non-admin."""
    update_data = {"email": random_email()}
    response = await client.put(f"{settings.API_V1_STR}/users/{admin_user.id}", headers=test_user_token_headers, json=update_data)
    assert response.status_code == status.HTTP_403_FORBIDDEN

async def test_update_user_unauthenticated(client: AsyncClient, test_user: User) -> None:
    """Test updating a user without authentication."""
    update_data = {"email": random_email()}
    response = await client.put(f"{settings.API_V1_STR}/users/{test_user.id}", json=update_data)
    assert response.status_code == status.HTTP_401_UNAUTHORIZED

# --- Test DELETE /users/{user_id} (Admin) ---

async def test_delete_user_admin_success(
    client: AsyncClient, admin_user_token_headers: Dict[str, str], test_user: User, db: Session
) -> None:
    """Test deleting a user as admin."""
    user_id_to_delete = test_user.id
    response = await client.delete(f"{settings.API_V1_STR}/users/{user_id_to_delete}", headers=admin_user_token_headers)
    
    assert response.status_code == status.HTTP_200_OK
    assert response.json() is True # Endpoint returns boolean
    
    # Verify user is deleted from DB
    deleted_user = UserService.get_by_id(db, user_id=user_id_to_delete)
    assert deleted_user is None

async def test_delete_user_admin_self(
    client: AsyncClient, admin_user_token_headers: Dict[str, str], admin_user: User
) -> None:
    """Test admin attempting to delete their own account."""
    response = await client.delete(f"{settings.API_V1_STR}/users/{admin_user.id}", headers=admin_user_token_headers)
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "Cannot delete your own user account" in response.json()["detail"]

async def test_delete_user_admin_not_found(
    client: AsyncClient, admin_user_token_headers: Dict[str, str]
) -> None:
    """Test deleting a non-existent user as admin."""
    non_existent_id = str(uuid.uuid4())
    response = await client.delete(f"{settings.API_V1_STR}/users/{non_existent_id}", headers=admin_user_token_headers)
    assert response.status_code == status.HTTP_404_NOT_FOUND

async def test_delete_user_non_admin(
    client: AsyncClient, test_user_token_headers: Dict[str, str], admin_user: User
) -> None:
    """Test deleting a user as non-admin."""
    response = await client.delete(f"{settings.API_V1_STR}/users/{admin_user.id}", headers=test_user_token_headers)
    assert response.status_code == status.HTTP_403_FORBIDDEN

async def test_delete_user_unauthenticated(client: AsyncClient, test_user: User) -> None:
    """Test deleting a user without authentication."""
    response = await client.delete(f"{settings.API_V1_STR}/users/{test_user.id}")
    assert response.status_code == status.HTTP_401_UNAUTHORIZED

# --- Test POST /users/{user_id}/activate (Admin) ---

async def test_activate_user_admin_success(
    client: AsyncClient, admin_user_token_headers: Dict[str, str], db: Session
) -> None:
    """Test activating a pending user as admin."""
    # Create a pending user first
    email = random_email()
    password = random_lower_string()
    user_in = {"email": email, "password": password}
    pending_user = UserService.create_user(db, user_in=user_in, status=UserStatus.PENDING)
    assert pending_user.status == UserStatus.PENDING
    
    response = await client.post(f"{settings.API_V1_STR}/users/{pending_user.id}/activate", headers=admin_user_token_headers)
    
    assert response.status_code == status.HTTP_200_OK
    content = response.json()
    assert content["id"] == pending_user.id
    assert content["status"] == UserStatus.ACTIVE.value
    
    # Verify status in DB
    db.refresh(pending_user)
    assert pending_user.status == UserStatus.ACTIVE

async def test_activate_user_admin_already_active(
    client: AsyncClient, admin_user_token_headers: Dict[str, str], test_user: User
) -> None:
    """Test activating an already active user as admin (should likely succeed idempotently)."""
    assert test_user.status == UserStatus.ACTIVE
    response = await client.post(f"{settings.API_V1_STR}/users/{test_user.id}/activate", headers=admin_user_token_headers)
    assert response.status_code == status.HTTP_200_OK # Or potentially 400 if designed to error
    content = response.json()
    assert content["status"] == UserStatus.ACTIVE.value

async def test_activate_user_admin_not_found(
    client: AsyncClient, admin_user_token_headers: Dict[str, str]
) -> None:
    """Test activating a non-existent user as admin."""
    non_existent_id = str(uuid.uuid4())
    response = await client.post(f"{settings.API_V1_STR}/users/{non_existent_id}/activate", headers=admin_user_token_headers)
    assert response.status_code == status.HTTP_404_NOT_FOUND

async def test_activate_user_non_admin(
    client: AsyncClient, test_user_token_headers: Dict[str, str], admin_user: User # Need another user ID
) -> None:
    """Test activating a user as non-admin."""
    response = await client.post(f"{settings.API_V1_STR}/users/{admin_user.id}/activate", headers=test_user_token_headers)
    assert response.status_code == status.HTTP_403_FORBIDDEN

async def test_activate_user_unauthenticated(client: AsyncClient, test_user: User) -> None:
    """Test activating a user without authentication."""
    response = await client.post(f"{settings.API_V1_STR}/users/{test_user.id}/activate")
    assert response.status_code == status.HTTP_401_UNAUTHORIZED

# --- Test POST /users/{user_id}/deactivate (Admin) ---

async def test_deactivate_user_admin_success(
    client: AsyncClient, admin_user_token_headers: Dict[str, str], test_user: User, db: Session
) -> None:
    """Test deactivating an active user as admin."""
    assert test_user.status == UserStatus.ACTIVE
    response = await client.post(f"{settings.API_V1_STR}/users/{test_user.id}/deactivate", headers=admin_user_token_headers)
    
    assert response.status_code == status.HTTP_200_OK
    content = response.json()
    assert content["id"] == test_user.id
    assert content["status"] == UserStatus.INACTIVE.value
    
    # Verify status in DB
    db.refresh(test_user)
    assert test_user.status == UserStatus.INACTIVE

async def test_deactivate_user_admin_self(
    client: AsyncClient, admin_user_token_headers: Dict[str, str], admin_user: User
) -> None:
    """Test admin attempting to deactivate their own account."""
    response = await client.post(f"{settings.API_V1_STR}/users/{admin_user.id}/deactivate", headers=admin_user_token_headers)
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "Cannot deactivate your own user account" in response.json()["detail"]

async def test_deactivate_user_admin_already_inactive(
    client: AsyncClient, admin_user_token_headers: Dict[str, str], db: Session
) -> None:
    """Test deactivating an already inactive user (should likely succeed idempotently)."""
    # Create an inactive user
    email = random_email()
    password = random_lower_string()
    user_in = {"email": email, "password": password}
    inactive_user = UserService.create_user(db, user_in=user_in, status=UserStatus.INACTIVE)
    
    response = await client.post(f"{settings.API_V1_STR}/users/{inactive_user.id}/deactivate", headers=admin_user_token_headers)
    assert response.status_code == status.HTTP_200_OK # Or 400 if designed to error
    content = response.json()
    assert content["status"] == UserStatus.INACTIVE.value

async def test_deactivate_user_admin_not_found(
    client: AsyncClient, admin_user_token_headers: Dict[str, str]
) -> None:
    """Test deactivating a non-existent user as admin."""
    non_existent_id = str(uuid.uuid4())
    response = await client.post(f"{settings.API_V1_STR}/users/{non_existent_id}/deactivate", headers=admin_user_token_headers)
    assert response.status_code == status.HTTP_404_NOT_FOUND

async def test_deactivate_user_non_admin(
    client: AsyncClient, test_user_token_headers: Dict[str, str], admin_user: User
) -> None:
    """Test deactivating a user as non-admin."""
    response = await client.post(f"{settings.API_V1_STR}/users/{admin_user.id}/deactivate", headers=test_user_token_headers)
    assert response.status_code == status.HTTP_403_FORBIDDEN

async def test_deactivate_user_unauthenticated(client: AsyncClient, test_user: User) -> None:
    """Test deactivating a user without authentication."""
    response = await client.post(f"{settings.API_V1_STR}/users/{test_user.id}/deactivate")
    assert response.status_code == status.HTTP_401_UNAUTHORIZED

# --- Helper Utilities (if not already in a shared conftest or utils file) ---
# Note: Moved random_email and random_lower_string to backend/tests/api/utils.py
# Ensure backend/tests/api/utils.py exists and contains these functions.
# If not, create it or include them here.

# Example backend/tests/api/utils.py:
# import random
# import string
#
# def random_lower_string(length: int = 32) -> str:
#     return "".join(random.choices(string.ascii_lowercase, k=length))
#
# def random_email() -> str:
#     return f"{random_lower_string()}@{random_lower_string(length=8)}.com"

# Need to create backend/tests/api/utils.py