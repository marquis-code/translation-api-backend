#!/bin/bash

# Medical Transcription System - Startup Script

echo "================================================"
echo "Medical Transcription System"
echo "================================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 16 or higher."
    exit 1
fi

echo "✅ Prerequisites check passed"
echo ""

# Start Backend
echo "🚀 Starting Backend Server..."
cd backend

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "📦 Installing Python dependencies..."
pip install -q -r requirements.txt

# Start backend in background
echo "🌐 Starting FastAPI server on http://localhost:8000"
python main.py &
BACKEND_PID=$!

# Wait for backend to start
sleep 3

# Start Frontend
cd ../frontend

echo ""
echo "🚀 Starting Frontend Server..."

# Install dependencies if needed
if [ ! -d "node_modules" ]; then
    echo "📦 Installing Node.js dependencies..."
    npm install
fi

# Start frontend
echo "🌐 Starting React app on http://localhost:3000"
npm start &
FRONTEND_PID=$!

echo ""
echo "================================================"
echo "✅ Application Started Successfully!"
echo "================================================"
echo ""
echo "Backend:  http://localhost:8000"
echo "Frontend: http://localhost:3000"
echo "API Docs: http://localhost:8000/docs"
echo ""
echo "Demo Login:"
echo "  Email:    doctor@clinic.com"
echo "  Password: password123"
echo ""
echo "Press Ctrl+C to stop all servers"
echo "================================================"

# Wait for Ctrl+C
trap "echo ''; echo '🛑 Stopping servers...'; kill $BACKEND_PID $FRONTEND_PID; exit" INT
wait