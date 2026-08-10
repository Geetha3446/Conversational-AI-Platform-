"""
Custom CSS that gives Streamlit the dark, ChatGPT-like look from the mockup.

Streamlit rebuilds the DOM on every rerun, so this is injected once per render
from app.py via `inject_css()`.
"""

import streamlit as st

CSS = """
<style>
/* ---------- global dark canvas ---------- */
.stApp { background-color: #1f1f1f; }
section[data-testid="stSidebar"] { background-color: #171717; border-right: 1px solid #2e2e2e; }
section[data-testid="stSidebar"] * { color: #e8e8e8; }

/* ---------- headings ---------- */
h1, h2, h3, h4 { color: #f2f2f2 !important; letter-spacing: -0.01em; }

/* ---------- chat bubbles ---------- */
div[data-testid="stChatMessage"] {
    background-color: #262626;
    border: 1px solid #333333;
    border-radius: 14px;
    padding: 0.85rem 1.05rem;
    margin-bottom: 0.6rem;
}
div[data-testid="stChatMessage"] p { color: #ececec; line-height: 1.6; }

/* ---------- input box ---------- */
div[data-testid="stChatInput"] textarea {
    background-color: #2b2b2b !important;
    color: #f0f0f0 !important;
    border-radius: 12px;
}

/* ---------- buttons ---------- */
.stButton > button {
    border-radius: 10px;
    border: 1px solid #3a3a3a;
    background-color: #2b2b2b;
    color: #e8e8e8;
    transition: all 0.15s ease-in-out;
}
.stButton > button:hover {
    border-color: #4f7cff;
    color: #ffffff;
    background-color: #333333;
}

/* ---------- starter tiles ---------- */
.tile-grid-note { color: #9a9a9a; font-size: 0.86rem; margin-bottom: 0.4rem; }

/* ---------- hero ---------- */
.hero-title {
    text-align: center;
    font-size: 2.4rem;
    font-weight: 800;
    color: #f5f5f5;
    margin: 1.2rem 0 0.2rem 0;
}
.hero-sub { text-align: center; color: #9a9a9a; margin-bottom: 1.6rem; font-size: 0.95rem; }

/* ---------- misc ---------- */
.badge {
    display: inline-block; padding: 2px 9px; border-radius: 999px;
    font-size: 0.72rem; background: #2f3b57; color: #a9c1ff; margin-right: 6px;
}
.tool-note { color: #8fb3ff; font-size: 0.8rem; font-style: italic; }
footer, #MainMenu { visibility: hidden; }
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
