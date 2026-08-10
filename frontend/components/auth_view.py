"""
The login / register gate.

While `st.session_state.token` is empty this is the only thing rendered. On a
successful login we stash the JWT and the user profile in session state and
rerun, at which point app.py shows the chat interface instead.
"""

from __future__ import annotations

import streamlit as st

from frontend import api_client


def _persist_login(data: dict) -> None:
    st.session_state.token = data["access_token"]
    st.session_state.user = data["user"]
    st.session_state.sessions = []
    st.session_state.current_session = None
    st.session_state.messages = []


def render_auth() -> None:
    """Draw the centred auth card."""
    left, middle, right = st.columns([1, 1.4, 1])

    with middle:
        st.markdown('<div class="hero-title">Conversational AI Platform</div>',
                    unsafe_allow_html=True)
        st.markdown(
            '<div class="hero-sub">Streaming chat, document Q&A and live tools, '
            "on your own machine.</div>",
            unsafe_allow_html=True,
        )

        # Connection badge, so a dead backend is obvious before you type anything.
        ok, info = api_client.health()
        if ok:
            key_ready = info.get("groq_key_configured")
            st.caption(
                f"Backend online. Model: {info.get('model')}. "
                + ("API key loaded." if key_ready else "GROQ_API_KEY missing in .env")
            )
        else:
            st.error(str(info))

        login_tab, register_tab = st.tabs(["Sign in", "Create account"])

        # ---------------- Sign in ----------------
        with login_tab:
            with st.form("login_form", clear_on_submit=False):
                username = st.text_input("Username", key="login_username")
                password = st.text_input("Password", type="password", key="login_password")
                submitted = st.form_submit_button("Sign in", use_container_width=True)

            if submitted:
                if not username or not password:
                    st.warning("Enter both a username and a password.")
                else:
                    with st.spinner("Signing in..."):
                        ok, data = api_client.login(username.strip(), password)
                    if ok:
                        _persist_login(data)
                        st.rerun()
                    else:
                        st.error(str(data))

        # ---------------- Register ----------------
        with register_tab:
            with st.form("register_form", clear_on_submit=False):
                new_username = st.text_input("Username", key="reg_username",
                                             help="Letters, digits, - and _ only")
                new_email = st.text_input("Email", key="reg_email")
                new_password = st.text_input("Password", type="password",
                                             key="reg_password",
                                             help="At least 8 characters")
                confirm = st.text_input("Confirm password", type="password",
                                        key="reg_confirm")
                created = st.form_submit_button("Create account",
                                                use_container_width=True)

            if created:
                if not all([new_username, new_email, new_password]):
                    st.warning("Fill in every field.")
                elif new_password != confirm:
                    st.warning("The two passwords do not match.")
                elif len(new_password) < 8:
                    st.warning("Use a password of at least 8 characters.")
                else:
                    with st.spinner("Creating your account..."):
                        ok, data = api_client.register(
                            new_username.strip(), new_email.strip(), new_password
                        )
                    if ok:
                        _persist_login(data)
                        st.success("Account created. Loading your workspace...")
                        st.rerun()
                    else:
                        st.error(str(data))
