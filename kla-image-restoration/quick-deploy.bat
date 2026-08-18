@echo off
echo ========================================
echo   RestoreNet Quick Deploy to GitHub
echo ========================================
echo.

REM Check if git is initialized
if not exist ".git" (
    echo Initializing git repository...
    git init
    echo.
)

REM Add all files
echo Adding all files to git...
git add .
echo.

REM Commit
set /p COMMIT_MSG="Enter commit message (or press Enter for default): "
if "%COMMIT_MSG%"=="" set COMMIT_MSG=Update for Vercel + Render deployment

echo Committing changes...
git commit -m "%COMMIT_MSG%"
echo.

REM Check if remote exists
git remote get-url origin >nul 2>&1
if errorlevel 1 (
    echo.
    echo ========================================
    echo   GitHub Remote Not Configured
    echo ========================================
    echo.
    echo 1. Create a new repository on GitHub:
    echo    https://github.com/new
    echo.
    echo 2. Name it: restorenet
    echo.
    echo 3. Run this command with your username:
    echo    git remote add origin https://github.com/YOUR_USERNAME/restorenet.git
    echo.
    echo 4. Then run this script again
    echo.
    pause
    exit /b 1
)

REM Push to GitHub
echo Pushing to GitHub...
git branch -M main
git push -u origin main

if errorlevel 1 (
    echo.
    echo [ERROR] Failed to push to GitHub
    echo.
    echo If this is your first push, you may need to authenticate.
    echo Or run: git push -u origin main --force
    echo.
    pause
    exit /b 1
)

echo.
echo ========================================
echo   Successfully pushed to GitHub!
echo ========================================
echo.
echo Next steps:
echo.
echo 1. Deploy Backend to Render:
echo    https://render.com/
echo    - New Web Service
echo    - Connect your GitHub repository
echo    - Python runtime, use render.yaml settings
echo.
echo 2. Deploy Frontend to Vercel:
echo    https://vercel.com/
echo    - New Project
echo    - Import your GitHub repository
echo    - Framework: Vite, Root: frontend
echo    - Add VITE_API_URL environment variable
echo.
echo 3. Update frontend\.env.production with Render URL
echo.
echo 4. Redeploy frontend to apply changes
echo.
pause
