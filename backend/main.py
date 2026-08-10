"""
FastAPI application entrypoint.

Run it from the project root:

    uvicorn backend.main:app --reload --port 8000

Interactive API docs: http://127.0.0.1:8000/docs
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.config import settings
from backend.database import init_db
from backend.routers import auth_routes, chat_routes, document_routes


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Startup and shutdown hooks."""
    init_db()
    print("=" * 68)
    print("  Conversational AI Platform - backend ready")
    print(f"  Database : {settings.db_path}")
    print(f"  Model    : {settings.GROQ_MODEL}")
    key_state = "set" if settings.GROQ_API_KEY.startswith("gsk_") else "MISSING"
    print(f"  Groq key : {key_state}")
    print(f"  Docs     : {settings.BACKEND_URL}/docs")
    print("=" * 68)
    yield
    print("Backend shutting down.")


app = FastAPI(
    title="Conversational AI Platform API",
    description=(
        "A ChatGPT-style backend: JWT auth, multi-session chat, PDF RAG, "
        "a LangGraph tool-calling agent and token streaming over SSE."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Streamlit runs on a different port, so it is a cross-origin caller.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your Streamlit origin in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_routes.router)
app.include_router(chat_routes.router)
app.include_router(document_routes.router)


@app.get("/health", tags=["system"])
def health() -> dict:
    """Cheap liveness probe. The frontend uses it to show a connection badge."""
    return {
        "status": "ok",
        "model": settings.GROQ_MODEL,
        "groq_key_configured": settings.GROQ_API_KEY.startswith("gsk_"),
    }


@app.exception_handler(Exception)
async def unhandled_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    """Never leak a raw traceback to the client."""
    print(f"[unhandled] {type(exc).__name__}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Check the backend logs."},
    )
