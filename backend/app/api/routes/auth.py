from datetime import timedelta
from typing import Any

from app.core.config import settings
from app.db.base import get_db
from app.models.user import UserStatus
from app.schemas.token import RefreshToken, Token
from app.schemas.user import UserCreate, UserResponse
from app.services.user import UserService
from app.utils.security import (create_access_token, create_refresh_token,
                                create_token_pair, decode_token)
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from jose import JWTError
from sqlalchemy.orm import Session

router = APIRouter()


@router.post("/register", response_model=UserResponse)
def register(user_in: UserCreate, db: Session = Depends(get_db)) -> Any:
    """
    Register a new user.
    """
    # Check if user already exists
    user = UserService.get_by_email(db, email=user_in.email)
    if user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A user with this email already exists",
        )

    # Create new user (pending approval)
    user = UserService.create_user(db, user_in)
    return user


@router.post("/login", response_model=Token)
def login(
    db: Session = Depends(get_db), form_data: OAuth2PasswordRequestForm = Depends()
) -> Any:
    """
    OAuth2 compatible token login, get an access token for future requests.
    """
    # Authenticate user
    user = UserService.authenticate(
        db, email=form_data.username, password=form_data.password
    )
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Check user status
    if user.status == UserStatus.PENDING:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Your account is pending approval by an administrator",
        )
    elif user.status == UserStatus.INACTIVE:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Your account has been deactivated",
        )

    # Update last login timestamp
    UserService.update_last_login(db, user)

    # Create access and refresh tokens
    access_token, refresh_token = create_token_pair(user.id)

    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
    }


@router.post("/refresh", response_model=Token)
def refresh_token(token_data: RefreshToken, db: Session = Depends(get_db)) -> Any:
    """
    Refresh access token using a valid refresh token.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid refresh token",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        # Decode the refresh token
        payload = decode_token(token_data.refresh_token)

        # Verify it's a refresh token
        if not payload.get("refresh"):
            raise credentials_exception

        # Extract user ID
        user_id = payload.get("sub")
        if user_id is None:
            raise credentials_exception

        # Get the user
        user = UserService.get_by_id(db, user_id=user_id)
        if user is None:
            raise credentials_exception

        # Check user status
        if user.status != UserStatus.ACTIVE:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Inactive user account",
            )

        # Create new token pair
        access_token, refresh_token = create_token_pair(user.id)

        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
        }

    except JWTError:
        raise credentials_exception
