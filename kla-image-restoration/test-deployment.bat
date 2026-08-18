@echo off
echo ========================================
echo   RestoreNet Deployment Test
echo ========================================
echo.

set /p BACKEND_URL="Enter your Render backend URL (e.g., https://restorenet-backend.onrender.com): "
set /p FRONTEND_URL="Enter your Vercel frontend URL (e.g., https://restorenet.vercel.app): "

echo.
echo Testing deployment...
echo.

REM Test backend health endpoint
echo [1/3] Testing backend health endpoint...
curl -s "%BACKEND_URL%/api/health" >nul 2>&1
if errorlevel 1 (
    echo [X] Backend health check failed!
    echo URL: %BACKEND_URL%/api/health
    echo.
    echo Possible issues:
    echo - Backend is still deploying (wait a few minutes)
    echo - URL is incorrect
    echo - Service is down
    echo.
) else (
    echo [OK] Backend is responding
    curl -s "%BACKEND_URL%/api/health"
    echo.
)
echo.

REM Test backend API documentation
echo [2/3] Checking API documentation...
echo Open this URL to view API docs:
echo %BACKEND_URL%/docs
echo.
start %BACKEND_URL%/docs
timeout /t 2 /nobreak >nul
echo.

REM Test frontend
echo [3/3] Opening frontend...
echo URL: %FRONTEND_URL%
echo.
start %FRONTEND_URL%
echo.

echo ========================================
echo   Test Checklist
echo ========================================
echo.
echo In your browser, verify:
echo.
echo [ ] Frontend loads without errors
echo [ ] No CORS errors in console (F12)
echo [ ] Can click "Load Synthetic Wafer"
echo [ ] Can click "RUN INFERENCE"
echo [ ] Image processes successfully
echo [ ] Metrics display correctly
echo [ ] Slider comparison works
echo.
echo If you see CORS errors:
echo 1. Update src/api/main.py CORS settings
echo 2. Add your Vercel domain to allow_origins
echo 3. Push changes to GitHub
echo 4. Wait for Render to auto-deploy
echo 5. Test again
echo.
echo ========================================
echo   Deployment URLs
echo ========================================
echo.
echo Backend API: %BACKEND_URL%/api
echo API Docs:    %BACKEND_URL%/docs
echo Frontend:    %FRONTEND_URL%
echo.
echo Backend Logs:  https://dashboard.render.com/
echo Frontend Logs: https://vercel.com/dashboard
echo.
pause
