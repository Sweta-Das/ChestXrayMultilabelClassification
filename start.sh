#!/bin/bash

# Quick Start Script for AI Chest X-ray Analysis v2.0
# Run this script to start both backend and frontend servers

echo "=========================================="
echo "🩺 AI Chest X-ray Analysis - Quick Start"
echo "=========================================="
echo ""

# Check if Python virtual environment exists
VENV_DIR=".venv-medfusion"
if [ -n "${PYTHON_BIN:-}" ] && command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v "$PYTHON_BIN")"
elif [ -x ".venv/bin/python" ]; then
    # Reuse the existing Python 3.10 interpreter if the old venv is still present.
    PYTHON_BIN=".venv/bin/python"
elif command -v python3.10 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3.10)"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
else
    echo "⚠️  No usable Python interpreter found."
    echo "This branch expects Python 3.10 for the MedFusionNet backend."
    echo "Set PYTHON_BIN to a Python 3.10 executable and try again."
    exit 1
fi

PYTHON_VERSION="$("$PYTHON_BIN" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"
case "$PYTHON_VERSION" in
    3.10|3.11|3.12)
        ;;
    *)
        echo "⚠️  Detected Python $PYTHON_VERSION, but this backend needs Python 3.10 to 3.12."
        echo "Set PYTHON_BIN to a compatible interpreter and try again."
        exit 1
        ;;
esac

if [ ! -d "$VENV_DIR" ]; then
    echo "⚠️  Virtual environment not found!"
    echo "Creating virtual environment at $VENV_DIR..."
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# Activate virtual environment
echo "📦 Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Install backend dependencies
echo "📥 Installing backend dependencies..."
pip install -q -r requirements.medfusion.txt

# Check for OpenAI API key
if [ -z "$OPENAI_API_KEY" ] && [ ! -f ".env" ]; then
    echo ""
    echo "⚠️  OpenAI API key not found!"
    echo "Please set OPENAI_API_KEY in .env file or as environment variable"
    echo "Example: export OPENAI_API_KEY='sk-your-key-here'"
    echo ""
fi

# Start backend in background
echo ""
echo "🚀 Starting FastAPI backend..."
python api.py > backend.log 2>&1 &
BACKEND_PID=$!
echo "   Backend PID: $BACKEND_PID"
echo "   Backend running at: http://localhost:8000"
echo "   Logs: backend.log"

# Wait for backend to start
sleep 3

# Start frontend
echo ""
echo "🚀 Starting Next.js frontend..."
cd frontend

# Install frontend dependencies if needed
if [ ! -d "node_modules" ]; then
    echo "📥 Installing frontend dependencies..."
    npm install
fi

echo ""
echo "✅ Application is starting!"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📍 Backend:  http://localhost:8000"
echo "📍 Frontend: http://localhost:3000"
echo "📍 API Docs: http://localhost:8000/docs"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Press Ctrl+C to stop both servers"
echo ""

# Start frontend (blocking)
npm run dev

# Cleanup backend when frontend is stopped
kill $BACKEND_PID 2>/dev/null
echo ""
echo "🛑 Servers stopped"
