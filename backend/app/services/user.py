from typing import Optional, List
from sqlalchemy.orm import Session
from app.models.user import User, UserRole, UserStatus
from app.schemas.user import UserCreate, UserUpdate
from app.utils.security import get_password_hash, verify_password, generate_uuid
from datetime import datetime, UTC

class UserService:
    @staticmethod
    def get_by_id(db: Session, user_id: str) -> Optional[User]:
        """Get a user by ID."""
        return db.query(User).filter(User.id == user_id).first()
    
    @staticmethod
    def get_by_email(db: Session, email: str) -> Optional[User]:
        """Get a user by email."""
        return db.query(User).filter(User.email == email).first()
    
    @staticmethod
    def get_users(
        db: Session,
        skip: int = 0,
        limit: int = 100,
        status: Optional[UserStatus] = None
    ) -> List[User]:
        """Get a list of users with optional filtering by status."""
        query = db.query(User)
        if status:
            query = query.filter(User.status == status)
        return query.offset(skip).limit(limit).all()
    
    @staticmethod
    def count_users(
        db: Session,
        status: Optional[UserStatus] = None
    ) -> int:
        """Count the total number of users with optional filtering by status."""
        query = db.query(User)
        if status:
            query = query.filter(User.status == status)
        return query.count()
    
    @staticmethod
    def create_user(db: Session, user_in: UserCreate, role: UserRole = UserRole.USER, status: UserStatus = UserStatus.PENDING) -> User:
        """Create a new user."""
        user = User(
            id=generate_uuid(),
            email=user_in.email,
            hashed_password=get_password_hash(user_in.password),
            role=role,
            status=status,
            theme_preference="dark"
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        return user
    
    @staticmethod
    def update_user(db: Session, user: User, user_in: UserUpdate) -> User:
        """Update a user."""
        update_data = user_in.dict(exclude_unset=True)
        
        if "password" in update_data and update_data["password"]:
            update_data["hashed_password"] = get_password_hash(update_data.pop("password"))
        
        for field, value in update_data.items():
            setattr(user, field, value)
        
        user.updated_at = datetime.now(UTC)
        db.add(user)
        db.commit()
        db.refresh(user)
        return user
    
    @staticmethod
    def delete_user(db: Session, user_id: str) -> bool:
        """Delete a user."""
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            return False
        db.delete(user)
        db.commit()
        return True
    
    @staticmethod
    def authenticate(db: Session, email: str, password: str) -> Optional[User]:
        """Authenticate a user."""
        user = UserService.get_by_email(db, email)
        if not user:
            return None
        if not verify_password(password, user.hashed_password):
            return None
        return user
    
    @staticmethod
    def update_last_login(db: Session, user: User) -> User:
        """Update the last login timestamp for a user."""
        user.last_login = datetime.now(UTC)
        db.add(user)
        db.commit()
        db.refresh(user)
        return user
    
    @staticmethod
    def activate_user(db: Session, user: User) -> User:
        """Activate a user account."""
        user.status = UserStatus.ACTIVE
        db.add(user)
        db.commit()
        db.refresh(user)
        return user
    
    @staticmethod
    def deactivate_user(db: Session, user: User) -> User:
        """Deactivate a user account."""
        user.status = UserStatus.INACTIVE
        db.add(user)
        db.commit()
        db.refresh(user)
        return user
    
    @staticmethod
    def create_first_admin(db: Session, email: str, password: str) -> Optional[User]:
        """Create the first admin user if no users exist."""
        # Check if any users exist
        user_count = db.query(User).count()
        if user_count > 0:
            return None
        
        # Create admin user
        admin_user = User(
            id=generate_uuid(),
            email=email,
            hashed_password=get_password_hash(password),
            role=UserRole.ADMIN,
            status=UserStatus.ACTIVE,
            theme_preference="dark"
        )
        db.add(admin_user)
        db.commit()
        db.refresh(admin_user)
        return admin_user