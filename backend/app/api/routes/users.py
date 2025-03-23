from typing import Any, List

from app.db.base import get_db
from app.models.user import User, UserRole, UserStatus
from app.schemas.user import UserCreate, UserResponse, UserUpdate
from app.services.user import UserService
from app.utils.deps import get_current_admin_user, get_current_user
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

router = APIRouter()


@router.get("/me", response_model=UserResponse)
def read_current_user(
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Get current user.
    """
    return current_user


@router.put("/me", response_model=UserResponse)
def update_current_user(
    user_in: UserUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Update current user.
    """
    # Only allow updating certain fields for self
    if user_in.role is not None or user_in.status is not None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not allowed to update role or status",
        )

    user = UserService.update_user(db, current_user, user_in)
    return user


@router.get("")
def read_users(
    db: Session = Depends(get_db),
    page: int = 1,
    size: int = 10,
    status: UserStatus = None,
    current_user: User = Depends(get_current_admin_user),
) -> Any:
    """
    Retrieve users. Admin only.
    """
    skip = (page - 1) * size
    users = UserService.get_users(db, skip=skip, limit=size, status=status)
    total = UserService.count_users(db, status=status)

    # Convert User objects to dictionaries with frontend-expected properties
    user_responses = []
    for user in users:
        user_dict = UserResponse.model_validate(user).model_dump()
        # Add is_active and is_admin properties for frontend compatibility
        user_dict["is_active"] = user.status == UserStatus.ACTIVE
        user_dict["is_admin"] = user.role == UserRole.ADMIN
        user_responses.append(user_dict)

    return {
        "items": user_responses,
        "total": total,
        "page": page,
        "size": size,
        "pages": (total + size - 1) // size,  # Ceiling division
    }


@router.post("", response_model=UserResponse)
def create_user(
    user_in: UserCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_admin_user),
) -> Any:
    """
    Create new user. Admin only.
    """
    user = UserService.get_by_email(db, email=user_in.email)
    if user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A user with this email already exists",
        )

    # Admins can create active users directly
    user = UserService.create_user(db, user_in, status=UserStatus.ACTIVE)
    return user


@router.get("/pending")
def read_pending_users(
    db: Session = Depends(get_db),
    page: int = 1,
    size: int = 10,
    current_user: User = Depends(get_current_admin_user),
) -> Any:
    """
    Retrieve pending users. Admin only.
    """
    skip = (page - 1) * size
    users = UserService.get_users(db, skip=skip, limit=size, status=UserStatus.PENDING)
    total = UserService.count_users(db, status=UserStatus.PENDING)

    # Convert User objects to dictionaries with frontend-expected properties
    user_responses = []
    for user in users:
        user_dict = UserResponse.model_validate(user).model_dump()
        # Add is_active and is_admin properties for frontend compatibility
        user_dict["is_active"] = user.status == UserStatus.ACTIVE
        user_dict["is_admin"] = user.role == UserRole.ADMIN
        user_responses.append(user_dict)

    return {
        "items": user_responses,
        "total": total,
        "page": page,
        "size": size,
        "pages": (total + size - 1) // size,  # Ceiling division
    }


@router.get("/{user_id}", response_model=UserResponse)
def read_user(
    user_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_admin_user),
) -> Any:
    """
    Get a specific user by id. Admin only.
    """
    user = UserService.get_by_id(db, user_id=user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    return user


@router.put("/{user_id}", response_model=UserResponse)
def update_user(
    user_id: str,
    user_update_data: dict,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_admin_user),
) -> Any:
    """
    Update a user. Admin only.
    """
    user = UserService.get_by_id(db, user_id=user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    # Convert frontend model to backend model
    user_in = UserUpdate()

    # Handle status update
    if "status" in user_update_data:
        user_in.status = UserStatus(user_update_data["status"])

    # Handle role update
    if "role" in user_update_data:
        user_in.role = UserRole(user_update_data["role"])

    # Handle other fields
    if "email" in user_update_data:
        user_in.email = user_update_data["email"]

    if "password" in user_update_data and user_update_data["password"]:
        user_in.password = user_update_data["password"]

    user = UserService.update_user(db, user, user_in)

    # Add frontend-expected properties
    response = UserResponse.model_validate(user).model_dump()
    response["is_active"] = user.status == UserStatus.ACTIVE
    response["is_admin"] = user.role == UserRole.ADMIN

    return response


@router.delete("/{user_id}", response_model=bool)
def delete_user(
    user_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_admin_user),
) -> Any:
    """
    Delete a user. Admin only.
    """
    # Prevent deleting self
    if user_id == current_user.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete your own user account",
        )

    result = UserService.delete_user(db, user_id=user_id)
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    return result


@router.post("/{user_id}/activate", response_model=UserResponse)
def activate_user(
    user_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_admin_user),
) -> Any:
    """
    Activate a user. Admin only.
    """
    user = UserService.get_by_id(db, user_id=user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    user = UserService.activate_user(db, user)
    return user


@router.post("/{user_id}/deactivate", response_model=UserResponse)
def deactivate_user(
    user_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_admin_user),
) -> Any:
    """
    Deactivate a user. Admin only.
    """
    # Prevent deactivating self
    if user_id == current_user.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot deactivate your own user account",
        )

    user = UserService.get_by_id(db, user_id=user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    user = UserService.deactivate_user(db, user)
    return user
