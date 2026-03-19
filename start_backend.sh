#!/bin/bash
# Start the FastAPI backend server
# This script must be run from the project root directory

cd "$(dirname "$0")"

# Activate virtual environment
source backend/venv/bin/activate

# Set PYTHONPATH to project root
export PYTHONPATH="$(pwd):${PYTHONPATH}"

# Start uvicorn using python -m to ensure proper module resolution
python -m uvicorn backend.main:app --reload --port 8000
