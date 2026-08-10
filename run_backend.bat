@echo off
REM Start the FastAPI backend on port 8000.
cd /d "%~dp0"
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
