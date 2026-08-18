@echo off
echo ========================================
echo   RestoreNet Deployment Checklist
echo ========================================
echo.

echo [1/10] Checking Git installation...
git --version >nul 2>&1
if errorlevel 1 (
    echo [X] Git not found! Install from https://git-scm.com/
    pause
    exit /b 1
)
echo [OK] Git installed
echo.

echo [2/10] Checking if repository is initialized...
if exist ".git" (
    echo [OK] Git repository exists
) else (
    echo [!] Initializing git repository...
    git init
    echo [OK] Repository initialized
)
echo.

echo [3/10] Checking .gitignore...
if exist ".gitignore" (
    echo [OK] .gitignore exists
) else (
    echo [!] .gitignore missing - create one!
)
echo.

echo [4/10] Checking required files...
set MISSING=0

if exist "requirements.txt" (
    echo [OK] requirements.txt found
) else (
    echo [X] requirements.txt missing!
    set MISSING=1
)

if exist "render.yaml" (
    echo [OK] render.yaml found
) else (
    echo [X] render.yaml missing!
    set MISSING=1
)

if exist "frontend\vercel.json" (
    echo [OK] frontend\vercel.json found
) else (
    echo [X] frontend\vercel.json missing!
    set MISSING=1
)

if exist "frontend\.env.production" (
    echo [OK] frontend\.env.production found
) else (
    echo [!] frontend\.env.production missing (optional)
)

if exist "checkpoints\best_model.pt" (
    echo [OK] Model checkpoint found
) else (
    echo [X] checkpoints\best_model.pt missing!
    set MISSING=1
)

if %MISSING%==1 (
    echo.
    echo [ERROR] Required files are missing!
    pause
    exit /b 1
)
echo.

echo [5/10] Checking frontend dependencies...
cd frontend
if exist "package.json" (
    echo [OK] package.json found
) else (
    echo [X] package.json missing!
    cd ..
    pause
    exit /b 1
)

echo [!] Testing frontend build...
call npm run build >nul 2>&1
if errorlevel 1 (
    echo [X] Frontend build failed! Run 'npm run build' to see errors
    cd ..
    pause
    exit /b 1
)
echo [OK] Frontend builds successfully
cd ..
echo.

echo [6/10] Checking Python dependencies...
python --version >nul 2>&1
if errorlevel 1 (
    echo [X] Python not found!
    pause
    exit /b 1
)
echo [OK] Python installed
echo.

echo [7/10] Checking Git remote...
git remote -v >nul 2>&1
if errorlevel 1 (
    echo [!] No git remote configured
    echo.
    echo To add GitHub remote, run:
    echo git remote add origin https://github.com/YOUR_USERNAME/restorenet.git
    echo.
) else (
    echo [OK] Git remote configured
    git remote -v
)
echo.

echo [8/10] Checking for uncommitted changes...
git diff --quiet
if errorlevel 1 (
    echo [!] You have uncommitted changes
    echo Run: git add . && git commit -m "Prepare for deployment"
) else (
    echo [OK] No uncommitted changes
)
echo.

echo [9/10] Checking model file size...
for %%I in (checkpoints\best_model.pt) do set SIZE=%%~zI
set /a SIZE_MB=%SIZE% / 1024 / 1024
echo Model size: %SIZE_MB% MB

if %SIZE_MB% GTR 100 (
    echo [!] Model file is large ^(%SIZE_MB% MB^)
    echo Consider using Git LFS:
    echo   git lfs install
    echo   git lfs track "checkpoints/*.pt"
    echo   git add .gitattributes
)
echo.

echo [10/10] Summary
echo ========================================
echo.
echo Backend Deployment (Render):
echo   1. Go to https://render.com
echo   2. New + ^> Web Service
echo   3. Connect your GitHub repository
echo   4. Use these settings:
echo      - Name: restorenet-backend
echo      - Runtime: Python 3
echo      - Build: pip install -r requirements.txt
echo      - Start: uvicorn src.api.main:app --host 0.0.0.0 --port $PORT
echo   5. Add environment variable: PYTHON_VERSION = 3.10.0
echo   6. Deploy!
echo.
echo Frontend Deployment (Vercel):
echo   1. Go to https://vercel.com
echo   2. New Project ^> Import Git Repository
echo   3. Select your repository
echo   4. Use these settings:
echo      - Framework: Vite
echo      - Root Directory: frontend
echo      - Build Command: npm run build
echo      - Output Directory: dist
echo   5. Add environment variable:
echo      VITE_API_URL = https://YOUR-BACKEND.onrender.com/api
echo   6. Deploy!
echo.
echo After deployment:
echo   - Update frontend\.env.production with actual Render URL
echo   - Test: https://YOUR-APP.vercel.app
echo   - Share your live app!
echo.
echo ========================================
echo   Deployment checklist complete!
echo ========================================
echo.
pause
