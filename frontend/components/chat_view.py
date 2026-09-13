"""
The main pane: the welcome screen with starter tiles, the transcript, and the
streaming renderer.

Streaming works by writing into an `st.empty()` placeholder inside the assistant
chat bubble. Every token that arrives from the SSE stream is appended to a
buffer and the placeholder is redrawn, which produces the typewriter effect
without any JavaScript.
"""

from __future__ import annotations

from typing import List

import streamlit as st

from frontend import api_client

# Quick-start tiles, mirroring the persona cards in the product mockup.
STARTERS: List[dict] = [
    {"icon": "[doc]",  "title": "Document Q&A",
     "prompt": "Summarise the key points of the PDF I uploaded."},
    {"icon": "[sun]",  "title": "Live weather",
     "prompt": "What is the weather in Patna right now, and should I carry an umbrella?"},
    {"icon": "[web]",  "title": "Live web search",
     "prompt": "Search the web and give me a quick summary of what FAISS is."},
    {"icon": "[calc]", "title": "Number crunching",
     "prompt": "Add 128000 and 47500, then tell me the result."},
]


def _ensure_session() -> str | None:
    """Create a chat session lazily if the user starts typing without one."""
    if st.session_state.current_session:
        return st.session_state.current_session

    ok, data = api_client.create_session(st.session_state.token)
    if not ok:
        st.error(str(data))
        return None
    st.session_state.current_session = data["id"]
    st.session_state.messages = []
    return data["id"]


def _render_welcome() -> None:
    """Shown when the active conversation has no messages yet."""
    st.markdown('<div class="hero-title">Start Conversation</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-sub">Ask anything. The assistant can search your PDFs '
        "and call live tools when it needs facts.</div>",
        unsafe_allow_html=True,
    )

    for row_start in range(0, len(STARTERS), 3):
        cols = st.columns(3)
        for col, item in zip(cols, STARTERS[row_start:row_start + 3]):
            with col:
                if st.button(
                    f"{item['icon']}\n\n**{item['title']}**",
                    key=f"starter_{item['title']}",
                    use_container_width=True,
                    help=item["prompt"],
                ):
                    st.session_state.pending_prompt = item["prompt"]
                    st.rerun()


def _render_history() -> None:
    """Repaint every stored turn of the active conversation."""
    for msg in st.session_state.messages:
        with st.chat_message("user" if msg["role"] == "user" else "assistant"):
            st.markdown(msg["content"])
            tools = (msg.get("tools_used") or "").strip()
            if tools:
                pretty = ", ".join(t.replace("_", " ") for t in tools.split(","))
                st.markdown(f"<span class='tool-note'>Tools used: {pretty}</span>",
                            unsafe_allow_html=True)


def _stream_reply(prompt: str) -> None:
    """Send one message and render the response as it streams in."""
    session_id = _ensure_session()
    if session_id is None:
        return

    # Echo the user's turn immediately.
    st.session_state.messages.append({"role": "user", "content": prompt, "tools_used": ""})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        status_slot = st.empty()   # shows "calling get_weather..."
        text_slot = st.empty()     # shows the growing answer
        buffer = ""
        tools_used: List[str] = []
        failed = False

        for event in api_client.stream_chat(
            st.session_state.token, session_id, prompt, st.session_state.use_tools
        ):
            kind = event.get("type")

            if kind == "token":
                buffer += event.get("content", "")
                # The block cursor makes the typing visible on slow tokens.
                text_slot.markdown(buffer + " |")

            elif kind == "tool_start":
                name = event.get("name", "tool").replace("_", " ")
                tools_used.append(event.get("name", "tool"))
                status_slot.info(f"Calling {name}...")

            elif kind == "tool_end":
                status_slot.empty()

            elif kind == "done":
                final = event.get("content") or buffer
                buffer = final
                tools_used = event.get("tools_used", tools_used)
                status_slot.empty()
                text_slot.markdown(final)

            elif kind == "error":
                failed = True
                status_slot.empty()
                text_slot.error(event.get("content", "Something went wrong."))

        if not failed:
            if not buffer:
                text_slot.warning("The model returned an empty response. Try rephrasing.")
            else:
                if tools_used:
                    pretty = ", ".join(sorted({t.replace('_', ' ') for t in tools_used}))
                    st.markdown(f"<span class='tool-note'>Tools used: {pretty}</span>",
                                unsafe_allow_html=True)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": buffer,
                        "tools_used": ",".join(sorted(set(tools_used))),
                    }
                )


def render_chat() -> None:
    """Entry point called by app.py once the user is authenticated."""
    if not st.session_state.messages:
        _render_welcome()
    else:
        _render_history()

    # A starter tile queues a prompt, then this rerun consumes it.
    pending = st.session_state.pop("pending_prompt", None)
    if pending:
        _stream_reply(pending)
        st.rerun()

    typed = st.chat_input("Type a message")
    if typed:
        _stream_reply(typed)
        st.rerun()