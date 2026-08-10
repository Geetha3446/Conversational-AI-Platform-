# Conversational AI Platform

A production-grade, ChatGPT-style application built end to end: real-time token
streaming, RAG-based document Q&A over your own PDFs, JWT authentication, and
multi-session chat that resumes days later exactly where you left it.

Stack: **FastAPI + Pydantic** (backend), **Streamlit** (frontend), **SQLite**
(everything persistent), **LangGraph** (agent workflow), **FAISS +
sentence-transformers** (retrieval), **Groq-hosted Llama 3.3 70B** (inference).

---

## 1. Quick start

### 1.1 Create the environment

```bash
cd conversational-ai-platform

python -m venv .venv

# macOS / Linux
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -r requirements.txt
```

The install pulls PyTorch as a dependency of `sentence-transformers`, so budget
roughly 2 GB of download and a few minutes. This happens once.

### 1.2 Configure secrets


Open `.env` and set two values:

- `GROQ_API_KEY` from https://console.groq.com/keys (free tier is enough)
- `JWT_SECRET_KEY`, generated with:

```bash
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

Everything else has a working default.

### 1.3 Run it

Two terminals, both with the venv active, both from the project root.

**Terminal 1, backend:**

```bash
uvicorn backend.main:app --reload --port 8000
```

**Terminal 2, frontend:**

```bash
streamlit run frontend/app.py
```

Then open http://localhost:8501, create an account, and start talking.

Convenience scripts are included if you prefer: `./run_backend.sh` and
`./run_frontend.sh` on macOS or Linux, `run_backend.bat` and
`run_frontend.bat` on Windows.

The interactive API documentation lives at http://127.0.0.1:8000/docs, where you
can exercise every endpoint directly using the Authorize button.

### 1.4 First-run note

The very first message after uploading a PDF is slower than the rest, because
that is when the `all-MiniLM-L6-v2` embedding model downloads (about 90 MB) and
loads into memory. Every request afterwards uses the cached model.

---

## 2. What you get

| Capability | How it works |
|---|---|
| Register and log in | bcrypt password hashing, JWT bearer tokens, protected routes via a FastAPI dependency |
| Multi-session chat | Each conversation is a row in SQLite with a uuid that doubles as the LangGraph `thread_id` |
| Resume tomorrow | LangGraph's `SqliteSaver` checkpoints agent state after every node, keyed by thread |
| Token streaming | Server-Sent Events from FastAPI, rendered into a Streamlit placeholder for a typewriter effect |
| PDF document Q&A | pypdf extraction, recursive chunking, local embeddings, FAISS cosine search, per-user isolation |
| Tool-calling agent | LangGraph conditional edge routes between the model and a ToolNode until the model stops asking |
| Live tools | Web search, news search, document search, weather, Wikipedia, currency, calculator, clock |
| Session management | Rename, delete, auto-titling from the first message, most-recent-first sidebar |

---

## 3. Project layout

```
conversational-ai-platform/
├── backend/
│   ├── main.py                  FastAPI app, CORS, lifespan, health check
│   ├── config.py                Typed settings loaded once from .env
│   ├── database.py              SQLAlchemy engine, WAL pragmas, session factory
│   ├── models.py                User, ChatSession, Message, Document tables
│   ├── schemas.py               Pydantic request and response contracts
│   ├── security.py              Password hashing, JWT, get_current_user dependency
│   ├── routers/
│   │   ├── auth_routes.py       /auth/register, /auth/login, /auth/me
│   │   ├── chat_routes.py       Session CRUD plus the SSE streaming endpoint
│   │   └── document_routes.py   PDF upload, list, delete
│   ├── agent/
│   │   ├── state.py             ChatState with the add_messages reducer
│   │   ├── tools.py             Retrieval tool factory plus five public API tools
│   │   └── graph.py             Graph construction, checkpointer, stream_chat()
│   └── rag/
│       ├── vectorstore.py       Chunking, embedding, FAISS index lifecycle
│       └── ingest.py            CLI for bulk-loading a folder of PDFs
├── frontend/
│   ├── app.py                   Streamlit entrypoint and auth gate
│   ├── api_client.py            Every HTTP call to the backend lives here
│   └── components/
│       ├── auth_view.py         Sign in and register card
│       ├── sidebar.py           Session list, PDF library, settings
│       ├── chat_view.py         Welcome tiles, transcript, streaming renderer
│       └── styles.py            Dark theme CSS
├── .streamlit/config.toml       Theme and upload limits
├── requirements.txt
├── .env.example
└── README.md
```

---

## 4. Architecture

### 4.1 Request path for one message

```
Streamlit chat input
    |
    v
POST /chat/stream  (JWT in the Authorization header)
    |
    +-- persist the user turn to SQLite, auto-title the session if it is new
    |
    v
LangGraph: START -> agent -> tools_condition -> [tools -> agent]* -> END
    |
    +-- SqliteSaver checkpoints state after every node, keyed by thread_id
    |
    v
Server-Sent Events streamed back token by token
    |
    v
Streamlit appends into an st.empty() placeholder
    |
    v
On completion, persist the assistant turn plus which tools it used
```

### 4.2 The graph

```
                    +---------+
        START ----> |  agent  | ----(no tool calls)----> END
                    +---------+
                     ^       |
                     |       | (tool calls present)
                     |       v
                     |  +---------+
                     +--|  tools  |
                        +---------+
```

`agent` calls Groq with the toolset bound. `tools_condition` is LangGraph's
prebuilt router: it inspects the last `AIMessage` and sends the run to the
ToolNode if it contains tool calls, otherwise to `END`. Results flow back into
`agent`, which either answers or asks for another tool. A `recursion_limit` of
12 stops a misbehaving model from looping forever.

The system prompt is prepended at call time and deliberately not stored in
state, so it never bloats checkpoints and can be edited later without rewriting
old threads.

### 4.3 Two kinds of persistence, on purpose

There are two records of every conversation, and they are not redundant:

- **LangGraph checkpoints** hold the agent's raw working memory, including tool
  calls and tool results. This is what makes a conversation resumable with full
  context, and it is what the model actually sees.
- **The `messages` table** holds a clean human-readable transcript. This is what
  repaints the UI, what you would query for analytics, and what survives a
  change of agent framework.

Both live in the same `data/app.db` file. SQLite runs in WAL mode with a 30
second busy timeout so the ORM and the checkpointer can write concurrently
without tripping over each other.

### 4.4 Retrieval

Each user gets a private FAISS index at `data/vectorstores/user_<id>/`:

- `index.faiss` holds the vectors
- `meta.json` holds chunk text, source filename and page number, as readable JSON

Embeddings are L2-normalised and the index is inner-product, which makes the
returned score exactly cosine similarity. The retrieval tool is built by a
factory that captures the authenticated user id in a closure, so the model
cannot be prompted into reading another user's files: the user id is never one
of the tool's arguments.

### 4.5 Tools available to the model

| Tool | Source | Key needed |
|---|---|---|
| `search_my_documents` | The user's own FAISS index | No |
| `web_search` | DuckDuckGo via the `ddgs` package | No |
| `news_search` | DuckDuckGo news via `ddgs` | No |
| `get_weather` | Open-Meteo geocoding + forecast | No |
| `search_wikipedia` | Wikipedia REST API | No |
| `convert_currency` | open.er-api.com | No |
| `calculator` | Python `ast`, arithmetic only, no `eval` | No |
| `get_current_datetime` | Local clock | No |

Every tool catches its own exceptions and returns a string, because an
unhandled tool error inside the graph would abort the user's whole turn.

**A caveat on web search specifically.** `ddgs`, formerly published as
`duckduckgo-search`, scrapes search result pages rather than calling an official
API. There is no supported free DuckDuckGo web search API, so this is the
practical option, and it comes with real consequences:

- Queries can be rate limited, especially in bursts. The tool detects this and
  returns a message telling the model to say so rather than inventing an answer.
- Upstream HTML changes can break parsing without warning. Pin the version and
  expect to bump it occasionally.
- `backend="duckduckgo"` is tried first, with the library's `auto` mode as a
  fallback, which may quietly serve results from Bing or Brave instead. If
  strictly DuckDuckGo results matter to you, remove the fallback in
  `_ddgs_search`.

For anything where reliability is load-bearing, swap in a paid search API. The
tool interface stays identical; only the body of `_ddgs_search` changes.

---

## 5. Bulk PDF ingestion

Useful for seeding a demo without clicking through the uploader:

```bash
python -m backend.rag.ingest --username your_username --folder ./some_pdfs
```

Register the account in the app first. Already-ingested filenames are skipped.

---

## 6. Configuration reference

All of these live in `.env`:

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | none | Required. Get one free from Groq. |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Any Groq chat model that supports tool calling |
| `LLM_TEMPERATURE` | `0.3` | 0.0 is deterministic, 1.0 is creative |
| `JWT_SECRET_KEY` | placeholder | Required in production. Long random string. |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | `10080` | Seven days |
| `DATA_DIR` | `data` | Where the DB, uploads and indexes live |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Any sentence-transformers model |
| `BACKEND_URL` | `http://127.0.0.1:8000` | How the frontend reaches the API |

Chunking parameters (`CHUNK_SIZE`, `CHUNK_OVERLAP`, `RETRIEVER_K`) are in
`backend/config.py` and can be overridden by `.env` entries of the same name.

---

## 7. What has been tested, and what has not

Being straight about this, because it matters if you are going to build on it.

**Verified against the real code:**

- The full API surface through FastAPI's `TestClient`: registration, duplicate
  rejection (409), login, wrong-password rejection (401), unauthenticated access
  blocked (401), session creation, SSE event ordering, message persistence with
  tool attribution, auto-titling, cross-user access blocked (404), non-PDF
  upload rejected (415), cascade deletion.
- The real LangGraph workflow driven by a stubbed LLM: the agent requested the
  calculator, the ToolNode executed it, the result fed back, and the final answer
  streamed. Checkpointed state showed the correct four-message sequence, and a
  second turn on the same thread id saw the full prior history. That is the
  resume-tomorrow behaviour working.
- The PDF pipeline: a real generated PDF chunked with page numbers preserved,
  indexed into FAISS, searched, reloaded from disk after a cache clear, and
  confirmed invisible to a second user.
- `calculator` and `get_current_datetime` exercised live.
- The web search tool across every path: result formatting with both `href` and
  `url` key shapes, date rendering, empty results, rate limiting, timeouts,
  missing package, and generic failure. Also routed end to end through the real
  graph, confirming the ToolNode ran it, the result reached the model, and tool
  attribution was recorded.

**Not verified:**

- The Groq call itself, since the test environment had no API key. The graph was
  exercised with a fake LLM in its place.
- Real semantic embeddings. `sentence-transformers` pulls a large PyTorch
  dependency, so a deterministic hash-based embedder stood in. The FAISS
  plumbing is proven; retrieval quality is not.
- Any live network call: weather, Wikipedia, currency, and the actual DuckDuckGo
  fetch. The test environment blocked those domains, so search was verified with
  injected results rather than real ones. The formatting and error handling are
  proven; the live scrape is not. This is the single most likely thing to need
  attention on your machine, so test it first.
- The Streamlit UI, which cannot be driven headlessly. It compiles clean and the
  API layer beneath it is tested, but it has not been clicked through.

**Known behaviour worth knowing:** the vector store appends. Duplicate
protection lives in the upload endpoint, via the `uq_user_filename` constraint
and an explicit 409 check, not in the FAISS layer. Calling `add_chunks` twice
with the same content directly, bypassing the API, would produce duplicate
vectors.

---

## 8. Before this goes anywhere public

This is built to production patterns, but a few things are set for local
convenience and must change first.

1. **CORS** is `allow_origins=["*"]` in `backend/main.py`. Restrict it to your
   actual frontend origin.
2. **The JWT secret** has a placeholder default and the app will run happily
   without you changing it. Generate a real one.
3. **Pin your dependencies.** `requirements.txt` uses lower bounds. LangGraph
   and langchain-core have both shipped major versions recently. Once you have a
   working install, freeze it: `pip freeze > requirements.lock.txt`.
4. **SQLite** is the right call for single-machine deployment and genuinely fine
   for a lot of real traffic in WAL mode. If you outgrow it, the ORM layer moves
   to Postgres with a connection-string change, and LangGraph has a Postgres
   checkpointer that is a drop-in for `SqliteSaver`.
5. **Rate limiting** is absent. Add `slowapi` or equivalent on `/auth/login` and
   `/chat/stream` before exposing either.
6. **Uploads** are capped at 25 MB and filenames are sanitised against path
   traversal, but there is no virus scanning or content inspection.

---

## 9. Troubleshooting

**"Backend is unreachable" on the login screen**
The frontend cannot see uvicorn. Confirm it is running on port 8000 and that
`BACKEND_URL` in `.env` matches.

**"GROQ_API_KEY is missing"**
The key is absent or still the placeholder. It must begin with `gsk_`. Restart
uvicorn after editing `.env`, since settings load once at import.

**"No text could be extracted"**
The PDF is a scanned image with no text layer. Run OCR on it first, for example
with `ocrmypdf`, then re-upload.

**"database is locked"**
Should not happen given WAL and the 30 second timeout, but if it does, stop both
processes and confirm you are not running two backends against the same file.

**Streaming arrives all at once instead of token by token**
Something between the browser and uvicorn is buffering. The endpoint already
sends `X-Accel-Buffering: no`; check any proxy in front of it.

**Web search returns a rate limit message**
DuckDuckGo throttles scraped requests. Wait a minute and retry. If it persists,
try `backend="auto"` as the primary in `_ddgs_search`, or move to a paid search
API.

**Web search returns nothing at all, repeatedly**
Upstream HTML likely changed. Update the package: `pip install -U ddgs`. If that
does not fix it, check the project's issue tracker.

**Tool calls never happen**
Confirm "Enable tools" is on in the sidebar, and that `GROQ_MODEL` is a model
that supports tool calling. Not every Groq-hosted model does.

---