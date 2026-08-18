@echo off
echo ========================================
echo   RestoreNet Production Startup Script
echo ========================================
echo.

REM Check if virtual environment exists
if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found!
    echo Please run setup_deployment.bat first
    pause
    exit /b 1
)

REM Activate virtual environment
call .venv\Scripts\activate.bat

echo [1/2] Starting Backend API Server...
echo.
start "RestoreNet Backend" cmd /k "python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 2"

timeout /t 5 /nobreak >nul

echo [2/2] Starting Frontend Server...
echo.
cd frontend
start "RestoreNet Frontend" cmd /k "npm run preview"

echo.
echo ========================================
echo   RestoreNet is starting up!
echo ========================================
echo.
echo Backend API:  http://localhost:8000
echo Frontend:     http://localhost:4173
echo API Health:   http://localhost:8000/api/health
echo.
echo Press any key to open the application in browser...
pause >nul

start http://localhost:4173

echo.
echo To stop the servers, close both terminal windows.
echo.
