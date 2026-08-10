#!/usr/bin/env bash
# Start the Streamlit frontend on port 8501.
set -e
cd "$(dirname "$0")"
streamlit run frontend/app.py
