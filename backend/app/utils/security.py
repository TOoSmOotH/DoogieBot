from passlib.context import CryptContext
from datetime import datetime, timedelta, UTC
from typing import Any, Dict, Optional, Union, Tuple
from jose import jwt
import uuid
from app.core.config import settings

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a hash."""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Generate a password hash."""
    return pwd_context.hash(password)

# JWT token handling
def create_access_token(subject: Union[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token."""
    if expires_delta:
        expire = datetime.now(UTC) + expires_delta
    else:
        expire = datetime.now(UTC) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode = {"exp": expire, "sub": str(subject)}
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt

def create_refresh_token(subject: Union[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a JWT refresh token with longer expiration.
    
    Args:
        subject: The subject of the token (usually user ID)
        expires_delta: Optional custom expiration time. If not provided, uses the default from settings.
    
    Returns:
        str: The encoded JWT refresh token
    """
    if expires_delta:
        expire = datetime.now(UTC) + expires_delta
    else:
        expire = datetime.now(UTC) + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    
    to_encode = {"exp": expire, "sub": str(subject), "refresh": True}
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt

def create_token_pair(subject: Union[str, Any]) -> Tuple[str, str]:
    """Create both access and refresh tokens."""
    access_token = create_access_token(subject)
    refresh_token = create_refresh_token(subject)
    return access_token, refresh_token

def decode_token(token: str) -> Dict[str, Any]:
    """Decode a JWT token and return its payload."""
    return jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])

# ID generation
def generate_uuid() -> str:
    """Generate a UUID string."""
    return str(uuid.uuid4())