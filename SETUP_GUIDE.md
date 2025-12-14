# Quick Setup Guide

## Prerequisites Check

Before starting, ensure you have:

- ✅ Python 3.8+ installed (`python --version`)
- ✅ Node.js 16+ installed (`node --version`)
- ✅ npm installed (`npm --version`)
- ✅ Git installed (`git --version`)
- ✅ Microphone connected and working

## 5-Minute Setup

### Step 1: Clone or Download the Project

```bash
# If using Git
git clone <repository-url>
cd medical-transcription-system

# Or download and extract the ZIP file
```

### Step 2: Backend Setup (2 minutes)

```bash
# Navigate to backend directory
cd backend

# Create and activate virtual environment
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start the backend server
python main.py
```

**Expected Output:**
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete.
```

Keep this terminal window open!

### Step 3: Frontend Setup (2 minutes)

Open a NEW terminal window:

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies (this may take a minute)
npm install

# Start the development server
npm start
```

**Expected Output:**
```
Compiled successfully!
Local:            http://localhost:3000
```

Your browser should automatically open to http://localhost:3000

### Step 4: Login and Test (1 minute)

1. The login page should appear automatically
2. Use these demo credentials:
   - Email: `doctor@clinic.com`
   - Password: `password123`
3. Click "Login"
4. You should see the Dashboard!

## Troubleshooting

### Backend Issues

**Problem:** `ModuleNotFoundError: No module named 'fastapi'`

**Solution:**
```bash
# Make sure virtual environment is activated
# Look for (venv) in your terminal prompt
pip install -r requirements.txt
```

**Problem:** `Port 8000 is already in use`

**Solution:**
```bash
# Kill the process using port 8000
# On Windows:
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# On macOS/Linux:
lsof -ti:8000 | xargs kill -9
```

### Frontend Issues

**Problem:** `npm: command not found`

**Solution:**
Install Node.js from https://nodejs.org/

**Problem:** `Port 3000 is already in use`

**Solution:**
```bash
# The terminal will ask if you want to use another port
# Type 'y' and press Enter
# Or specify a different port:
PORT=3001 npm start
```

**Problem:** Cannot connect to backend

**Solution:**
1. Verify backend is running on http://localhost:8000
2. Check backend terminal for errors
3. Try accessing http://localhost:8000/docs in browser

### Microphone Issues

**Problem:** Microphone not working

**Solution:**
1. Check browser permissions (click lock icon in address bar)
2. Allow microphone access when prompted
3. Test microphone in browser settings
4. Try using Chrome/Edge (best WebRTC support)

## Verification Steps

### 1. Verify Backend is Running

Open http://localhost:8000/docs in your browser

You should see the FastAPI Swagger documentation.

### 2. Verify Frontend is Running

Open http://localhost:3000 in your browser

You should see the login page.

### 3. Test the Full Flow

1. **Login** with demo credentials
2. **Start New Consultation** from dashboard
3. **Click "Start Recording"** (allow microphone access)
4. **Speak into microphone** - you should see transcript appear
5. **Click "Stop Recording"**
6. **Click "Generate Summary"**
7. **Edit the summary sections**
8. **Click "Save"** then **"Mark as Completed"**
9. **Return to dashboard** - you should see your completed consultation

## Directory Structure

```
medical-transcription-system/
├── backend/
│   ├── main.py                 # FastAPI application
│   ├── streaming_service.py    # AssemblyAI integration
│   ├── websocket_manager.py    # WebSocket handling
│   ├── requirements.txt        # Python dependencies
│   └── .env.example           # Environment variables template
│
├── frontend/
│   ├── public/
│   │   └── index.html
│   ├── src/
│   │   ├── components/
│   │   │   ├── Login.jsx
│   │   │   ├── Dashboard.jsx
│   │   │   ├── TranscriptionPage.jsx
│   │   │   └── SummaryPage.jsx
│   │   ├── App.jsx
│   │   ├── App.css
│   │   └── index.js
│   ├── package.json
│   └── .env.example
│
└── README.md                   # Full documentation
```

## Next Steps

1. Read the full [README.md](README.md) for detailed documentation
2. Explore the API documentation at http://localhost:8000/docs
3. Test with real consultations
4. Review the code structure and architecture

## Getting Help

If you encounter issues:

1. Check the terminal logs for error messages
2. Verify all prerequisites are installed correctly
3. Make sure both backend and frontend are running
4. Check firewall settings aren't blocking ports 8000 or 3000
5. Try restarting both servers

## Production Deployment

For production deployment instructions, see the "Deployment" section in the main README.md.

---

**Setup Time:** ~5 minutes
**First Consultation:** Ready to go!

Happy transcribing! 🎉