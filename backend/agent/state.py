"""
The graph's shared state.

`add_messages` is a LangGraph reducer: instead of overwriting the messages list
on every node return, it appends new messages and merges streamed chunks by id.
That single annotation is what gives the agent its conversational memory inside
one run, and combined with the SQLite checkpointer, across days.
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class ChatState(TypedDict):
    """State carried between nodes of the graph."""

    # Full conversation history. The reducer appends rather than replaces.
    messages: Annotated[list[BaseMessage], add_messages]
