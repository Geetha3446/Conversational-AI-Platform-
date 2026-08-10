"""
Authentication helpers: password hashing, JWT creation, and the FastAPI
dependency that turns an "Authorization: Bearer ..." header into a User row.

We call bcrypt directly rather than going through passlib, because recent
bcrypt releases broke passlib's version detection and produce noisy warnings.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from sqlalchemy.orm import Session

from backend.config import settings
from backend.database import get_db
from backend.models import User

# tokenUrl is only used by the /docs "Authorize" button.
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

# bcrypt refuses inputs longer than 72 bytes, so we truncate defensively.
_BCRYPT_MAX_BYTES = 72


def hash_password(plain: str) -> str:
    """Return a salted bcrypt hash, safe to store in the database."""
    payload = plain.encode("utf-8")[:_BCRYPT_MAX_BYTES]
    return bcrypt.hashpw(payload, bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    """Constant-time comparison of a candidate password against a stored hash."""
    try:
        payload = plain.encode("utf-8")[:_BCRYPT_MAX_BYTES]
        return bcrypt.checkpw(payload, hashed.encode("utf-8"))
    except (ValueError, TypeError):
        # Malformed hash in the DB: fail closed rather than crash.
        return False


def create_access_token(user_id: int, username: str) -> str:
    """Sign a JWT carrying the user id (sub) and an expiry claim."""
    expire = datetime.now(timezone.utc) + timedelta(
        minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
    )
    payload = {
        "sub": str(user_id),
        "username": username,
        "exp": expire,
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def decode_token(token: str) -> Optional[int]:
    """Return the user id inside a valid token, or None if it is bad/expired."""
    try:
        payload = jwt.decode(
            token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM]
        )
        sub = payload.get("sub")
        return int(sub) if sub is not None else None
    except (JWTError, ValueError):
        return None


def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> User:
    """
    FastAPI dependency. Any route that declares
    `user: User = Depends(get_current_user)` is automatically protected.
    """
    credentials_error = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials. Please log in again.",
        headers={"WWW-Authenticate": "Bearer"},
    )

    user_id = decode_token(token)
    if user_id is None:
        raise credentials_error

    user = db.get(User, user_id)
    if user is None or not user.is_active:
        raise credentials_error
    return user
