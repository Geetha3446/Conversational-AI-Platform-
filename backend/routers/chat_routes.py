"""
/chat endpoints: session management plus the streaming completion endpoint.

Streaming uses Server-Sent Events. Each line looks like:

    data: {"type": "token", "content": "Hello"}

The frontend parses those JSON payloads and renders tokens as they arrive.
Starlette runs a synchronous generator in a worker thread, which is exactly what
we want because LangGraph's SQLite checkpointer is synchronous.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Generator, List

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from backend.agent.graph import delete_thread, stream_chat
from backend.database import SessionLocal, get_db
from backend.models import ChatSession, Message, User
from backend.schemas import (
    ChatRequest,
    MessageOut,
    SessionCreate,
    SessionOut,
    SessionRename,
    SimpleMessage,
)
from backend.security import get_current_user

router = APIRouter(prefix="/chat", tags=["chat"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _owned_session(db: Session, session_id: str, user_id: int) -> ChatSession:
    """Fetch a session, 404 if it does not exist or belongs to someone else."""
    chat = db.get(ChatSession, session_id)
    if chat is None or chat.user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Chat session not found."
        )
    return chat


def _derive_title(text: str, limit: int = 48) -> str:
    """Turn the first user message into a readable sidebar title."""
    clean = " ".join(text.strip().split())
    return clean if len(clean) <= limit else clean[: limit - 1].rstrip() + "..."


# ---------------------------------------------------------------------------
# Session CRUD
# ---------------------------------------------------------------------------
@router.get("/sessions", response_model=List[SessionOut])
def list_sessions(
    db: Session = Depends(get_db), user: User = Depends(get_current_user)
) -> List[ChatSession]:
    """Most recently touched chats first. This is what fills the sidebar."""
    return (
        db.query(ChatSession)
        .filter(ChatSession.user_id == user.id)
        .order_by(ChatSession.updated_at.desc())
        .all()
    )


@router.post("/sessions", response_model=SessionOut, status_code=status.HTTP_201_CREATED)
def create_session(
    payload: SessionCreate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> ChatSession:
    chat = ChatSession(
        id=str(uuid.uuid4()),
        user_id=user.id,
        title=(payload.title or "New Chat").strip()[:200],
    )
    db.add(chat)
    db.commit()
    db.refresh(chat)
    return chat


@router.get("/sessions/{session_id}/messages", response_model=List[MessageOut])
def get_messages(
    session_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> List[Message]:
    """Full transcript, used to repaint a conversation the user comes back to."""
    chat = _owned_session(db, session_id, user.id)
    return chat.messages


@router.patch("/sessions/{session_id}", response_model=SessionOut)
def rename_session(
    session_id: str,
    payload: SessionRename,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> ChatSession:
    chat = _owned_session(db, session_id, user.id)
    chat.title = payload.title.strip()[:200]
    db.commit()
    db.refresh(chat)
    return chat


@router.delete("/sessions/{session_id}", response_model=SimpleMessage)
def delete_session(
    session_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> SimpleMessage:
    """Delete the transcript AND the agent's checkpointed memory for that thread."""
    chat = _owned_session(db, session_id, user.id)
    db.delete(chat)
    db.commit()
    delete_thread(session_id)
    return SimpleMessage(detail="Chat deleted.")


# ---------------------------------------------------------------------------
# Streaming completion
# ---------------------------------------------------------------------------
@router.post("/stream")
def chat_stream(
    payload: ChatRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> StreamingResponse:
    """
    Send one user message and stream the assistant's reply back as SSE.

    Persistence happens in two places on purpose:
      * LangGraph checkpoints the agent state (its working memory)
      * these ORM rows keep a clean transcript for the UI and for analytics
    """
    chat = _owned_session(db, payload.session_id, user.id)

    # Name the conversation from its first message.
    if chat.title in ("New Chat", "", None):
        chat.title = _derive_title(payload.message)

    db.add(Message(session_id=chat.id, role="user", content=payload.message))
    chat.updated_at = datetime.now(timezone.utc)
    db.commit()

    # Capture primitives now: the request-scoped session closes once streaming starts.
    user_id = user.id
    session_id = chat.id
    user_message = payload.message
    use_tools = payload.use_tools

    def event_source() -> Generator[str, None, None]:
        assistant_text = ""
        tools_used: List[str] = []
        errored = False

        try:
            for event in stream_chat(user_id, session_id, user_message, use_tools):
                if event["type"] == "done":
                    assistant_text = event.get("content", "")
                    tools_used = event.get("tools_used", [])
                elif event["type"] == "error":
                    errored = True
                    assistant_text = event.get("content", "Unknown error")
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as exc:  # last-resort guard so the stream always closes
            errored = True
            assistant_text = f"Streaming failed: {exc}"
            yield f"data: {json.dumps({'type': 'error', 'content': assistant_text})}\n\n"

        # Write the assistant turn with a fresh session of our own.
        if assistant_text and not errored:
            write_db = SessionLocal()
            try:
                write_db.add(
                    Message(
                        session_id=session_id,
                        role="assistant",
                        content=assistant_text,
                        tools_used=",".join(tools_used),
                    )
                )
                stored = write_db.get(ChatSession, session_id)
                if stored is not None:
                    stored.updated_at = datetime.now(timezone.utc)
                write_db.commit()
            except Exception as exc:
                write_db.rollback()
                print(f"[chat] Failed to persist assistant message: {exc}")
            finally:
                write_db.close()

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # stops nginx-style buffering if proxied
        },
    )
