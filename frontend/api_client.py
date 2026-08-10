"""
Thin HTTP client the Streamlit app uses to talk to FastAPI.

Everything the UI needs from the backend goes through this one module, so the
UI code stays free of URLs, headers and JSON handling. Every method returns
(ok, payload) so callers can render an error without try/except everywhere.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Generator, List, Optional, Tuple

import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")
TIMEOUT = 30


def _headers(token: Optional[str]) -> Dict[str, str]:
    return {"Authorization": f"Bearer {token}"} if token else {}


def _unwrap(response: requests.Response) -> Tuple[bool, Any]:
    """Normalise a response into (ok, data-or-error-string)."""
    try:
        body = response.json()
    except ValueError:
        body = {"detail": response.text or "Empty response from server"}

    if response.ok:
        return True, body

    detail = body.get("detail") if isinstance(body, dict) else str(body)
    if isinstance(detail, list) and detail:  # pydantic validation errors
        first = detail[0]
        detail = first.get("msg", str(first)) if isinstance(first, dict) else str(first)
    return False, detail or f"Request failed with status {response.status_code}"


# ---------------------------------------------------------------------------
# System
# ---------------------------------------------------------------------------
def health() -> Tuple[bool, Any]:
    try:
        return _unwrap(requests.get(f"{BASE_URL}/health", timeout=5))
    except requests.RequestException:
        return False, "Backend is unreachable. Is uvicorn running on port 8000?"


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------
def register(username: str, email: str, password: str) -> Tuple[bool, Any]:
    try:
        return _unwrap(
            requests.post(
                f"{BASE_URL}/auth/register",
                json={"username": username, "email": email, "password": password},
                timeout=TIMEOUT,
            )
        )
    except requests.RequestException as exc:
        return False, f"Could not reach the backend: {exc}"


def login(username: str, password: str) -> Tuple[bool, Any]:
    try:
        return _unwrap(
            requests.post(
                f"{BASE_URL}/auth/login-json",
                json={"username": username, "password": password},
                timeout=TIMEOUT,
            )
        )
    except requests.RequestException as exc:
        return False, f"Could not reach the backend: {exc}"


def me(token: str) -> Tuple[bool, Any]:
    try:
        return _unwrap(
            requests.get(f"{BASE_URL}/auth/me", headers=_headers(token), timeout=TIMEOUT)
        )
    except requests.RequestException as exc:
        return False, str(exc)


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------
def list_sessions(token: str) -> Tuple[bool, Any]:
    try:
        return _unwrap(
            requests.get(
                f"{BASE_URL}/chat/sessions", headers=_headers(token), timeout=TIMEOUT
            )
        )
    except requests.RequestException as exc:
        return False, str(exc)


def create_session(token: str, title: Optional[str] = None) -> Tuple[bool, Any]:
    try:
        return _unwrap(
            requests.post(
                f"{BASE_URL}/chat/sessions",
                headers=_headers(token),
                json={"title": title},
                timeout=TIMEOUT,
            )
        )
    except requests.RequestException as exc:
        return False, str(exc)


def get_messages(token: str, session_id: str) -> Tuple[bool, Any]:
    try:
        return _unwrap(
            requests.get(
                f"{BASE_URL}/chat/sessions/{session_id}/messages",
                headers=_headers(token),
                timeout=TIMEOUT,
            )
        )
    except requests.RequestException as exc:
        return False, str(exc)


def rename_session(token: str, session_id: str, title: str) -> Tuple[bool, Any]:
    try:
        return _unwrap(
            requests.patch(
                f"{BASE_URL}/chat/sessions/{session_id}",
                headers=_headers(token),
                json={"title": title},
                timeout=TIMEOUT,
            )
        )
    except requests.RequestException as exc:
        return False, str(exc)


def delete_session(token: str, session_id: str) -> Tuple[bool, Any]:
    try:
        return _unwrap(
            requests.delete(
                f"{BASE_URL}/chat/sessions/{session_id}",
                headers=_headers(token),
                timeout=TIMEOUT,
            )
        )
    except requests.RequestException as exc:
        return False, str(exc)


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------
def list_documents(token: str) -> Tuple[bool, Any]:
    try:
        return _unwrap(
            requests.get(
                f"{BASE_URL}/documents", headers=_headers(token), timeout=TIMEOUT
            )
        )
    except requests.RequestException as exc:
        return False, str(exc)


def upload_document(token: str, filename: str, data: bytes) -> Tuple[bool, Any]:
    try:
        return _unwrap(
            requests.post(
                f"{BASE_URL}/documents/upload",
                headers=_headers(token),
                files={"file": (filename, data, "application/pdf")},
                timeout=300,  # embedding a large PDF can take a while
            )
        )
    except requests.RequestException as exc:
        return False, str(exc)


def delete_document(token: str, document_id: int) -> Tuple[bool, Any]:
    try:
        return _unwrap(
            requests.delete(
                f"{BASE_URL}/documents/{document_id}",
                headers=_headers(token),
                timeout=TIMEOUT,
            )
        )
    except requests.RequestException as exc:
        return False, str(exc)


# ---------------------------------------------------------------------------
# Streaming chat
# ---------------------------------------------------------------------------
def stream_chat(
    token: str, session_id: str, message: str, use_tools: bool = True
) -> Generator[Dict[str, Any], None, None]:
    """
    Yield decoded SSE events from POST /chat/stream.

    Consumers see the same event dicts the backend emits: token, tool_start,
    tool_end, done, error.
    """
    try:
        with requests.post(
            f"{BASE_URL}/chat/stream",
            headers={**_headers(token), "Accept": "text/event-stream"},
            json={
                "session_id": session_id,
                "message": message,
                "use_tools": use_tools,
            },
            stream=True,
            timeout=300,
        ) as response:
            if not response.ok:
                ok, detail = _unwrap(response)
                yield {"type": "error", "content": str(detail)}
                return

            for raw_line in response.iter_lines(decode_unicode=True):
                if not raw_line or not raw_line.startswith("data: "):
                    continue
                payload = raw_line[6:]
                if payload == "[DONE]":
                    break
                try:
                    yield json.loads(payload)
                except json.JSONDecodeError:
                    continue
    except requests.RequestException as exc:
        yield {"type": "error", "content": f"Connection lost: {exc}"}
