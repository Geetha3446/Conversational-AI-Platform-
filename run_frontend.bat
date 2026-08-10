@echo off
REM Start the Streamlit frontend on port 8501.
cd /d "%~dp0"
streamlit run frontend/app.py
