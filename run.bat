@echo off
echo ================================================
echo Medical Transcription System
echo ================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo X Python is not installed. Please install Python 3.8 or higher.
    pause
    exit /b 1
)

REM Check if Node.js is installed
node --version >nul 2>&1
if errorlevel 1 (
    echo X Node.js is not installed. Please install Node.js 16 or higher.
    pause
    exit /b 1
)

echo [OK] Prerequisites check passed
echo.

REM Start Backend
echo [START] Starting Backend Server...
cd backend

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo [SETUP] Creating Python virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install dependencies
echo [SETUP] Installing Python dependencies...
pip install -q -r requirements.txt

REM Start backend
echo [RUN] Starting FastAPI server on http://localhost:8000
start "Backend Server" cmd /k "python main.py"

REM Wait for backend to start
timeout /t 3 /nobreak >nul

REM Start Frontend
cd ..\frontend

echo.
echo [START] Starting Frontend Server...

REM Install dependencies if needed
if not exist "node_modules" (
    echo [SETUP] Installing Node.js dependencies...
    call npm install
)

REM Start frontend
echo [RUN] Starting React app on http://localhost:3000
start "Frontend Server" cmd /k "npm start"

echo.
echo ================================================
echo [OK] Application Started Successfully!
echo ================================================
echo.
echo Backend:  http://localhost:8000
echo Frontend: http://localhost:3000
echo API Docs: http://localhost:8000/docs
echo.
echo Demo Login:
echo   Email:    doctor@clinic.com
echo   Password: password123
echo.
echo Close the command windows to stop the servers
echo ================================================
echo.
pause