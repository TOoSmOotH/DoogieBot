from sqlalchemy import Boolean, Column, String, DateTime, Enum
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
import enum
from app.db.base import Base

class UserRole(str, enum.Enum):
    USER = "user"
    ADMIN = "admin"

class UserStatus(str, enum.Enum):
    PENDING = "pending"
    ACTIVE = "active"
    INACTIVE = "inactive"

class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    role = Column(Enum(UserRole), default=UserRole.USER, nullable=False)
    status = Column(Enum(UserStatus), default=UserStatus.PENDING, nullable=False)
    theme_preference = Column(String, default="dark", nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    last_login = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    tags = relationship("Tag", back_populates="user", cascade="all, delete-orphan")
    mcp_configs = relationship("MCPServerConfig", back_populates="user", cascade="all, delete-orphan")
    documents = relationship("Document", back_populates="uploader", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<User {self.email}>"
