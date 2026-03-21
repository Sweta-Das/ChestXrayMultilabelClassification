#!/bin/bash

# Quick Start Script for AI Chest X-ray Analysis v2.0
# Run this script to start both backend and frontend servers

echo "=========================================="
echo "🩺 AI Chest X-ray Analysis - Quick Start"
echo "=========================================="
echo ""

# Check if Python virtual environment exists
if [ ! -d "venv" ]; then
    echo "⚠️  Virtual environment not found!"
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "📦 Activating virtual environment..."
source venv/bin/activate

# Install backend dependencies
echo "📥 Installing backend dependencies..."
pip install -q -r requirements.txt

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
