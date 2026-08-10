"""
The left rail: new chat, chat history, PDF library, settings and sign out.

Sessions are fetched from the backend on every rerun, which keeps the list
correct after renames and deletes without any client-side cache to invalidate.
"""

from __future__ import annotations

from datetime import datetime

import streamlit as st

from frontend import api_client


def _pretty_date(iso: str) -> str:
    """Turn an ISO timestamp into 'Today', 'Yesterday' or a short date."""
    try:
        ts = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return ""
    delta = (datetime.utcnow().date() - ts.date()).days
    if delta == 0:
        return ts.strftime("Today %H:%M")
    if delta == 1:
        return "Yesterday"
    if delta < 7:
        return f"{delta} days ago"
    return ts.strftime("%d %b")


def _load_session(session_id: str) -> None:
    """Switch the main pane to a different conversation and repaint its history."""
    token = st.session_state.token
    ok, msgs = api_client.get_messages(token, session_id)
    st.session_state.current_session = session_id
    st.session_state.messages = (
        [{"role": m["role"], "content": m["content"], "tools_used": m.get("tools_used", "")}
         for m in msgs]
        if ok
        else []
    )
    if not ok:
        st.session_state.flash = f"Could not load that chat: {msgs}"


def render_sidebar() -> None:
    token = st.session_state.token
    user = st.session_state.user

    with st.sidebar:
        st.markdown(f"### {user['username']}")
        st.caption(user["email"])
        st.divider()

        # ---------------- New chat ----------------
        if st.button("New chat", use_container_width=True, type="primary"):
            ok, data = api_client.create_session(token)
            if ok:
                st.session_state.current_session = data["id"]
                st.session_state.messages = []
                st.rerun()
            else:
                st.error(str(data))

        # ---------------- History ----------------
        st.markdown("#### Recent chats")
        ok, sessions = api_client.list_sessions(token)
        if not ok:
            st.error(str(sessions))
            sessions = []
        st.session_state.sessions = sessions

        if not sessions:
            st.caption("No conversations yet. Start one above.")

        for chat in sessions:
            is_active = chat["id"] == st.session_state.current_session
            label = ("> " if is_active else "") + chat["title"]

            row_main, row_del = st.columns([5, 1])
            with row_main:
                if st.button(
                    label,
                    key=f"open_{chat['id']}",
                    use_container_width=True,
                    help=_pretty_date(chat["updated_at"]),
                ):
                    _load_session(chat["id"])
                    st.rerun()
            with row_del:
                if st.button("x", key=f"del_{chat['id']}", help="Delete this chat"):
                    ok, resp = api_client.delete_session(token, chat["id"])
                    if ok:
                        if st.session_state.current_session == chat["id"]:
                            st.session_state.current_session = None
                            st.session_state.messages = []
                        st.rerun()
                    else:
                        st.error(str(resp))

        st.divider()

        # ---------------- Documents ----------------
        with st.expander("Documents (RAG)", expanded=False):
            uploaded = st.file_uploader(
                "Upload a PDF",
                type=["pdf"],
                key=f"uploader_{st.session_state.upload_counter}",
                help="The text is chunked, embedded locally and made searchable "
                     "by the assistant.",
            )
            if uploaded is not None:
                with st.spinner(f"Indexing {uploaded.name}..."):
                    ok, data = api_client.upload_document(
                        token, uploaded.name, uploaded.getvalue()
                    )
                if ok:
                    st.success(
                        f"{data['filename']}: {data['num_pages']} pages, "
                        f"{data['num_chunks']} chunks indexed."
                    )
                    # Bump the key so the widget resets and does not re-upload.
                    st.session_state.upload_counter += 1
                    st.rerun()
                else:
                    st.error(str(data))

            ok, docs = api_client.list_documents(token)
            if ok and docs:
                st.caption(f"{len(docs)} document(s) in your knowledge base")
                for doc in docs:
                    c1, c2 = st.columns([5, 1])
                    c1.write(f"{doc['filename']}  \n"
                             f"<span class='badge'>{doc['num_chunks']} chunks</span>",
                             unsafe_allow_html=True)
                    if c2.button("x", key=f"docdel_{doc['id']}", help="Remove"):
                        api_client.delete_document(token, doc["id"])
                        st.rerun()
            elif ok:
                st.caption("No documents yet. Upload a PDF to enable document Q&A.")

        # ---------------- Settings ----------------
        with st.expander("Settings", expanded=False):
            st.session_state.use_tools = st.toggle(
                "Enable tools",
                value=st.session_state.use_tools,
                help="Document search, weather, Wikipedia, currency, calculator "
                     "and clock. Turn off for a plain LLM chat.",
            )
            ok, info = api_client.health()
            st.caption(f"Model: {info.get('model')}" if ok else "Backend offline")

        st.divider()
        if st.button("Sign out", use_container_width=True):
            for key in ("token", "user", "sessions", "current_session", "messages"):
                st.session_state[key] = None if key in ("token", "user", "current_session") else []
            st.rerun()
