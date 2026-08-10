#!/usr/bin/env bash
# Start the FastAPI backend on port 8000.
set -e
cd "$(dirname "$0")"
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
