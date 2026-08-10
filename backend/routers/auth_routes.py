"""
/auth endpoints: register, login, and "who am I".

Login accepts the standard OAuth2 password form so the interactive docs at
/docs can authorise, and also a JSON body for convenience from Streamlit.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models import User
from backend.schemas import Token, UserCreate, UserLogin, UserOut
from backend.security import (
    create_access_token,
    get_current_user,
    hash_password,
    verify_password,
)

router = APIRouter(prefix="/auth", tags=["auth"])


def _issue_token(user: User) -> Token:
    return Token(
        access_token=create_access_token(user.id, user.username),
        token_type="bearer",
        user=UserOut.model_validate(user),
    )


@router.post("/register", response_model=Token, status_code=status.HTTP_201_CREATED)
def register(payload: UserCreate, db: Session = Depends(get_db)) -> Token:
    """Create an account and return a token, so the user is logged in immediately."""
    clash = (
        db.query(User)
        .filter((User.username == payload.username) | (User.email == payload.email))
        .first()
    )
    if clash:
        field = "Username" if clash.username == payload.username else "Email"
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"{field} is already registered. Try logging in instead.",
        )

    user = User(
        username=payload.username,
        email=payload.email,
        hashed_password=hash_password(payload.password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return _issue_token(user)


@router.post("/login", response_model=Token)
def login_form(
    form: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
) -> Token:
    """OAuth2 password flow. Used by the /docs Authorize button."""
    user = db.query(User).filter(User.username == form.username).first()
    if user is None or not verify_password(form.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return _issue_token(user)


@router.post("/login-json", response_model=Token)
def login_json(payload: UserLogin, db: Session = Depends(get_db)) -> Token:
    """Same as /login but with a JSON body. Used by the Streamlit client."""
    user = db.query(User).filter(User.username == payload.username).first()
    if user is None or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password.",
        )
    return _issue_token(user)


@router.get("/me", response_model=UserOut)
def me(user: User = Depends(get_current_user)) -> UserOut:
    """Validate a stored token and return the profile behind it."""
    return UserOut.model_validate(user)
