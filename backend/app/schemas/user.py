from typing import Optional
from pydantic import BaseModel, EmailStr, ConfigDict
from datetime import datetime
from app.models.user import UserRole, UserStatus

# Shared properties
class UserBase(BaseModel):
    email: Optional[EmailStr] = None
    role: Optional[UserRole] = None
    status: Optional[UserStatus] = None
    theme_preference: Optional[str] = None

# Properties to receive via API on creation
class UserCreate(BaseModel):
    email: EmailStr
    password: str

# Properties to receive via API on update
class UserUpdate(UserBase):
    password: Optional[str] = None

# Properties shared by models stored in DB
class UserInDBBase(UserBase):
    id: str
    email: EmailStr
    role: UserRole
    status: UserStatus
    created_at: datetime
    updated_at: datetime
    last_login: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)

# Properties to return to client
class User(UserInDBBase):
    pass

# Properties stored in DB
class UserInDB(UserInDBBase):
    hashed_password: str

# Response model for user creation/update
class UserResponse(BaseModel):
    id: str
    email: EmailStr
    role: UserRole
    status: UserStatus
    theme_preference: str
    created_at: datetime
    updated_at: datetime
    last_login: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)