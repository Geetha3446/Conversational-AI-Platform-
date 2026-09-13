"""
The LangGraph workflow.

    START -> agent -> (tools_condition) -> tools -> agent -> ... -> END

* `agent` calls the Groq-hosted Llama model with the toolset bound to it.
* `tools_condition` is LangGraph's prebuilt router: if the model's reply
  contains tool calls, go to the ToolNode, otherwise finish.
* `tools` executes every requested tool and appends ToolMessages to the state.
* The loop repeats until the model answers without asking for a tool.

Persistence
-----------
`SqliteSaver` checkpoints the state after every node, keyed by `thread_id`.
We use the chat session's uuid as the thread id, so closing the browser and
returning tomorrow resumes the exact same agent memory.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, Generator, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from backend.agent.state import ChatState
from backend.agent.tools import build_toolset
from backend.config import settings
from backend.database import raw_connection

SYSTEM_PROMPT = """You are a helpful, precise assistant inside a document-aware chat application.

Guidelines:
- When the user refers to their documents, files, PDFs or "the report", call the
  search_my_documents tool before answering, and cite the file name and page.
- Your training data has a cutoff. For current or changing facts, prices,
  who currently holds a role, ongoing events, call web_search rather than
  answering from memory. web_search covers known topics and entities well
  (definitions, "what is", "who is") but is not a full news feed; if it
  returns nothing useful for a narrow or breaking-news query, say so plainly
  rather than guessing.
- When you use web results, cite the source URLs and say when something is
  reported rather than confirmed. Search snippets are short and can mislead.
- Use the calculator for arithmetic rather than computing in your head. It
  takes two numbers and one operation (add, subtract, multiply, divide) per
  call, so for a multi-step calculation call it more than once, one step
  at a time, rather than trying to pass a full expression.
- Prefer calling a tool over guessing. If a tool returns nothing useful, say so
  plainly instead of inventing an answer.
- Format answers in clean markdown. Keep them as short as the question allows.
"""

# ---------------------------------------------------------------------------
# Shared singletons
# ---------------------------------------------------------------------------
_checkpointer: Optional[SqliteSaver] = None
_checkpointer_lock = threading.Lock()

# Compiled graphs cached by (user_id, use_tools) so we do not rebuild per request.
_graph_cache: Dict[tuple, Any] = {}
_graph_lock = threading.Lock()


def get_checkpointer() -> SqliteSaver:
    """One SqliteSaver for the whole process, writing into the same app.db file."""
    global _checkpointer
    if _checkpointer is None:
        with _checkpointer_lock:
            if _checkpointer is None:
                saver = SqliteSaver(raw_connection())
                # Older versions expose setup(); newer ones do it lazily.
                setup = getattr(saver, "setup", None)
                if callable(setup):
                    try:
                        setup()
                    except Exception:
                        pass
                _checkpointer = saver
    return _checkpointer


def _make_llm(tools: Optional[List] = None):
    """Create a ChatGroq client, optionally with tools bound."""
    from langchain_groq import ChatGroq

    if not settings.GROQ_API_KEY or settings.GROQ_API_KEY.startswith("gsk_replace"):
        raise RuntimeError(
            "GROQ_API_KEY is missing. Copy .env.example to .env and paste a real "
            "key from https://console.groq.com/keys"
        )

    llm = ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model=settings.GROQ_MODEL,
        temperature=settings.LLM_TEMPERATURE,
        max_retries=2,
        streaming=True,
    )
    return llm.bind_tools(tools) if tools else llm


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------
def build_graph(user_id: int, use_tools: bool = True):
    """Compile (and cache) the workflow for one user."""
    key = (user_id, use_tools)
    with _graph_lock:
        if key in _graph_cache:
            return _graph_cache[key]

    tools = build_toolset(user_id) if use_tools else []
    llm = _make_llm(tools if tools else None)

    def agent_node(state: ChatState) -> Dict[str, List]:
        """Call the model with the system prompt plus the running history.

        The system prompt is prepended at call time and deliberately not stored
        in state, so it never bloats the checkpoint and can be changed later
        without rewriting old threads.
        """
        response = llm.invoke([SystemMessage(content=SYSTEM_PROMPT)] + state["messages"])
        return {"messages": [response]}

    workflow = StateGraph(ChatState)
    workflow.add_node("agent", agent_node)
    workflow.add_edge(START, "agent")

    if tools:
        workflow.add_node("tools", ToolNode(tools))
        # tools_condition returns "tools" when the last AIMessage has tool_calls,
        # otherwise END.
        workflow.add_conditional_edges(
            "agent", tools_condition, {"tools": "tools", END: END}
        )
        workflow.add_edge("tools", "agent")  # feed results back for a final answer
    else:
        workflow.add_edge("agent", END)

    compiled = workflow.compile(checkpointer=get_checkpointer())

    with _graph_lock:
        _graph_cache[key] = compiled
    return compiled


def invalidate_graph_cache(user_id: int) -> None:
    """Called after a document upload so the retriever picks up the new index."""
    with _graph_lock:
        for key in list(_graph_cache):
            if key[0] == user_id:
                _graph_cache.pop(key, None)


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------
def stream_chat(
    user_id: int,
    session_id: str,
    user_message: str,
    use_tools: bool = True,
) -> Generator[Dict[str, Any], None, None]:
    """
    Run one turn and yield events as they happen.

    Event shapes:
        {"type": "token",      "content": "..."}     partial assistant text
        {"type": "tool_start", "name": "get_weather"}
        {"type": "tool_end",   "name": "get_weather"}
        {"type": "done",       "content": "...", "tools_used": [...]}
        {"type": "error",      "content": "..."}

    The caller is responsible for persisting the final text; this function only
    talks to the graph.
    """
    tools_used: List[str] = []
    final_text_parts: List[str] = []

    try:
        graph = build_graph(user_id, use_tools)
    except Exception as exc:
        yield {"type": "error", "content": str(exc)}
        return

    config = {
        "configurable": {"thread_id": session_id},
        # Guard against a model that loops on tools forever.
        "recursion_limit": 12,
    }
    inputs = {"messages": [HumanMessage(content=user_message)]}

    try:
        for mode, payload in graph.stream(
            inputs, config=config, stream_mode=["messages", "updates"]
        ):
            # ---- token-by-token text from the model ------------------------
            if mode == "messages":
                chunk, metadata = payload
                if metadata.get("langgraph_node") != "agent":
                    continue
                text = getattr(chunk, "content", "")
                if isinstance(text, list):
                    # Some providers return content as a list of blocks.
                    text = "".join(
                        b.get("text", "") for b in text if isinstance(b, dict)
                    )
                if text:
                    final_text_parts.append(text)
                    yield {"type": "token", "content": text}

            # ---- node-level updates, used to surface tool activity ---------
            elif mode == "updates":
                for node_name, node_output in (payload or {}).items():
                    if not isinstance(node_output, dict):
                        continue
                    for msg in node_output.get("messages", []) or []:
                        if node_name == "agent" and isinstance(msg, AIMessage):
                            for call in msg.tool_calls or []:
                                name = call.get("name", "tool")
                                tools_used.append(name)
                                yield {"type": "tool_start", "name": name}
                        elif node_name == "tools":
                            yield {
                                "type": "tool_end",
                                "name": getattr(msg, "name", "tool"),
                            }

        yield {
            "type": "done",
            "content": "".join(final_text_parts).strip(),
            "tools_used": sorted(set(tools_used)),
        }

    except Exception as exc:
        # Surface a readable message instead of a raw traceback in the UI.
        yield {
            "type": "error",
            "content": f"The assistant hit an error: {type(exc).__name__}: {exc}",
        }


def get_thread_messages(session_id: str) -> List:
    """Read the persisted LangGraph state for a thread (used for debugging)."""
    try:
        state = get_checkpointer().get({"configurable": {"thread_id": session_id}})
        if state is None:
            return []
        return state.get("channel_values", {}).get("messages", [])
    except Exception:
        return []


def delete_thread(session_id: str) -> None:
    """Remove all checkpoints for a thread when the user deletes a chat."""
    try:
        saver = get_checkpointer()
        deleter = getattr(saver, "delete_thread", None)
        if callable(deleter):
            deleter(session_id)
            return
        # Fallback for versions without delete_thread.
        conn = saver.conn
        for table in ("checkpoints", "writes"):
            try:
                conn.execute(f"DELETE FROM {table} WHERE thread_id = ?", (session_id,))
            except Exception:
                pass
        conn.commit()
    except Exception as exc:
        print(f"[graph] Could not delete thread {session_id}: {exc}")