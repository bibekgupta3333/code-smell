#!/bin/bash
# Quick Start Script for Code Smell Detection System
# This script starts the FastAPI server with the new frontend

set -e

PROJECT_DIR="/Users/bibekgupta/Downloads/projects/code-smell"
VENV_PATH="$PROJECT_DIR/.venv"

echo "🚀 Starting Code Smell Detection System..."
echo ""

# Check if venv exists
if [ ! -d "$VENV_PATH" ]; then
    echo "❌ Virtual environment not found at $VENV_PATH"
    echo "Please create it with: python3 -m venv $VENV_PATH"
    exit 1
fi

# Activate venv
echo "📦 Activating virtual environment..."
source "$VENV_PATH/bin/activate"

# Navigate to project
cd "$PROJECT_DIR"

# Check if static files exist
if [ ! -d "src/static" ]; then
    echo "❌ Static files not found at src/static"
    echo "Frontend may not be properly set up"
    exit 1
fi

echo "✅ Virtual environment activated"
echo ""

# Start the server
echo "🎯 Starting FastAPI server..."
echo "📍 Frontend will be available at: http://localhost:8000"
echo "📍 API Docs at: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python3 -m uvicorn src.api_server:app --reload --host 0.0.0.0 --port 8000

# If we get here, the server stopped
echo ""
echo "🛑 Server stopped"
