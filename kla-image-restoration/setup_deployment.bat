@echo off
echo ========================================
echo   RestoreNet Deployment Setup
echo ========================================
echo.

REM Check Python
echo [1/6] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.10 or higher from https://www.python.org/
    pause
    exit /b 1
)
echo [OK] Python found
echo.

REM Check Node.js
echo [2/6] Checking Node.js installation...
node --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Node.js is not installed or not in PATH
    echo Please install Node.js 16+ from https://nodejs.org/
    pause
    exit /b 1
)
echo [OK] Node.js found
echo.

REM Create virtual environment
echo [3/6] Creating Python virtual environment...
if exist ".venv" (
    echo Virtual environment already exists, skipping...
) else (
    python -m venv .venv
    echo [OK] Virtual environment created
)
echo.

REM Install Python dependencies
echo [4/6] Installing Python dependencies...
call .venv\Scripts\activate.bat
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo [ERROR] Failed to install Python dependencies
    pause
    exit /b 1
)
echo [OK] Python dependencies installed
echo.

REM Install Node.js dependencies
echo [5/6] Installing Node.js dependencies...
cd frontend
if exist "node_modules" (
    echo Node modules already exist, skipping...
) else (
    call npm install
    if errorlevel 1 (
        echo [ERROR] Failed to install Node.js dependencies
        pause
        exit /b 1
    )
    echo [OK] Node.js dependencies installed
)
echo.

REM Build frontend
echo [6/6] Building frontend for production...
call npm run build
if errorlevel 1 (
    echo [ERROR] Failed to build frontend
    pause
    exit /b 1
)
echo [OK] Frontend built successfully
cd ..
echo.

echo ========================================
echo   Setup Complete!
echo ========================================
echo.
echo Next steps:
echo 1. Run start_production.bat to start the application
echo 2. Open http://localhost:4173 in your browser
echo.
echo For detailed deployment instructions, see DEPLOYMENT_GUIDE.md
echo.
pause
