"""
Streamlit entrypoint.

Run it from the project root, with the backend already up:

    streamlit run frontend/app.py

The app is a single page with two states: signed out (auth card) and signed in
(sidebar + chat). All server state lives in FastAPI and SQLite, so refreshing
the browser loses nothing except the in-memory token.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Allow "from frontend import ..." when Streamlit runs this file directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from frontend import api_client  # noqa: E402
from frontend.components.auth_view import render_auth  # noqa: E402
from frontend.components.chat_view import render_chat  # noqa: E402
from frontend.components.sidebar import render_sidebar  # noqa: E402
from frontend.components.styles import inject_css  # noqa: E402

st.set_page_config(
    page_title="Conversational AI Platform",
    page_icon="AI",
    layout="wide",
    initial_sidebar_state="expanded",
)


def bootstrap_state() -> None:
    """Give every session-state key a defined starting value exactly once."""
    defaults = {
        "token": None,
        "user": None,
        "sessions": [],
        "current_session": None,
        "messages": [],
        "use_tools": True,
        "upload_counter": 0,
        "flash": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def token_is_valid() -> bool:
    """Ask the backend whether the stored JWT still works (it may have expired)."""
    if not st.session_state.token:
        return False
    ok, data = api_client.me(st.session_state.token)
    if ok:
        st.session_state.user = data
        return True
    return False


def main() -> None:
    bootstrap_state()
    inject_css()

    if st.session_state.flash:
        st.warning(st.session_state.flash)
        st.session_state.flash = None

    if not token_is_valid():
        # Clear a stale token so the user is not stuck in a redirect loop.
        st.session_state.token = None
        render_auth()
        return

    render_sidebar()
    render_chat()


if __name__ == "__main__":
    main()
