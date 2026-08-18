# 🚀 Deploy RestoreNet to Vercel + Render

Complete guide to deploy RestoreNet with:
- **Frontend** → Vercel (Free tier available)
- **Backend** → Render (Free tier available)

---

## 📋 Prerequisites

- GitHub account
- Vercel account ([vercel.com](https://vercel.com))
- Render account ([render.com](https://render.com))
- Git installed locally

---

## 🎯 Deployment Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Deployment Architecture               │
└─────────────────────────────────────────────────────────┘

User Browser
     │
     ├──────────> Frontend (Vercel)
     │            - React SPA
     │            - Static files
     │            - CDN delivery
     │                 │
     │                 │ API Calls
     │                 ▼
     └──────────> Backend (Render)
                  - FastAPI server
                  - Model inference
                  - PyTorch runtime
```

---

## 📦 Step 1: Prepare Your Repository

### 1.1 Create .gitignore (if not exists)

Create `kla-image-restoration/.gitignore`:

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
.venv/
venv/
ENV/
env/

# Node
node_modules/
npm-debug.log*
yarn-debug.log*
yarn-error.log*
.pnpm-debug.log*

# Build outputs
dist/
build/
*.egg-info/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
logs/
*.log

# Test coverage
.coverage
htmlcov/

# Model checkpoints (keep best_model.pt)
checkpoints/*.pt
!checkpoints/best_model.pt

# Data
data/GT/*.npy
data/NoisyLR/*.npy
results/

# Environment variables
.env
.env.local
```

### 1.2 Push to GitHub

```bash
# Navigate to project root
cd c:\Users\Lenovo\Downloads\restore-net\kla-image-restoration

# Initialize git (if not already)
git init

# Add all files
git add .

# Commit
git commit -m "Prepare for Vercel + Render deployment"

# Create GitHub repository (on github.com)
# Then add remote and push:
git remote add origin https://github.com/YOUR_USERNAME/restorenet.git
git branch -M main
git push -u origin main
```

---

## 🖥️ Step 2: Deploy Backend to Render

### 2.1 Create Render Configuration

Create `render.yaml` in project root:

```yaml
services:
  - type: web
    name: restorenet-backend
    env: python
    region: oregon
    plan: free
    branch: main
    buildCommand: "pip install -r requirements.txt"
    startCommand: "uvicorn src.api.main:app --host 0.0.0.0 --port $PORT"
    envVars:
      - key: PYTHON_VERSION
        value: 3.10.0
      - key: PORT
        value: 10000
```

### 2.2 Update Backend for Render

The backend needs to use Render's `$PORT` environment variable.

Update `src/api/main.py` if needed (already compatible):

```python
# main.py already uses uvicorn which respects PORT env var
# No changes needed!
```

### 2.3 Deploy to Render

1. **Go to [render.com](https://render.com)** and sign in

2. **Click "New +"** → **"Web Service"**

3. **Connect GitHub repository**:
   - Click "Connect account" if first time
   - Select your `restorenet` repository

4. **Configure service**:
   ```
   Name: restorenet-backend
   Region: Oregon (US West)
   Branch: main
   Root Directory: (leave blank or set to "kla-image-restoration")
   Runtime: Python 3
   Build Command: pip install -r requirements.txt
   Start Command: uvicorn src.api.main:app --host 0.0.0.0 --port $PORT
   ```

5. **Add Environment Variables**:
   - Click "Advanced"
   - Add: `PYTHON_VERSION` = `3.10.0`
   - Add: `PORT` = `10000` (Render sets this automatically)

6. **Select Plan**:
   - Choose **Free** (for testing) or **Starter** ($7/month for better performance)

7. **Click "Create Web Service"**

8. **Wait for deployment** (5-10 minutes first time)

9. **Get your backend URL**:
   ```
   https://restorenet-backend.onrender.com
   ```

### 2.4 Test Backend

```bash
# Test health endpoint
curl https://restorenet-backend.onrender.com/api/health

# Expected response:
# {"status":"ok","device":"cpu"}
```

⚠️ **Important Notes for Render Free Tier**:
- Service spins down after 15 minutes of inactivity
- First request after spin-down takes 30-60 seconds (cold start)
- Only CPU inference available on free tier
- 512 MB RAM limit

---

## 🌐 Step 3: Deploy Frontend to Vercel

### 3.1 Create Vercel Configuration

Create `frontend/vercel.json`:

```json
{
  "buildCommand": "npm run build",
  "outputDirectory": "dist",
  "devCommand": "npm run dev",
  "installCommand": "npm install",
  "framework": "vite",
  "rewrites": [
    {
      "source": "/(.*)",
      "destination": "/index.html"
    }
  ]
}
```

### 3.2 Update API URL for Production

Create `frontend/.env.production`:

```env
VITE_API_URL=https://restorenet-backend.onrender.com/api
```

### 3.3 Update API Client

Modify `frontend/src/api/client.js`:

```javascript
// Use environment variable or fallback to localhost
const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000/api';

export const restoreImage = async (originalFile) => {
    const formData = new FormData();
    formData.append('file', originalFile);

    const response = await fetch(`${API_BASE}/restore`, {
        method: 'POST',
        body: formData,
    });

    // ... rest of the code stays the same
};

// ... rest of the file
```

### 3.4 Update CORS on Backend

Update `src/api/main.py` to allow Vercel domain:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:4173",
        "https://*.vercel.app",  # Allow all Vercel preview deployments
        "https://restorenet.vercel.app",  # Your custom domain (update this)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

Commit and push this change:

```bash
git add src/api/main.py
git commit -m "Update CORS for Vercel"
git push
```

Wait for Render to auto-deploy the update.

### 3.5 Deploy to Vercel

1. **Go to [vercel.com](https://vercel.com)** and sign in

2. **Click "Add New"** → **"Project"**

3. **Import Git Repository**:
   - Click "Import" on your GitHub repository
   - If not listed, click "Adjust GitHub App Permissions"

4. **Configure Project**:
   ```
   Project Name: restorenet
   Framework Preset: Vite
   Root Directory: frontend
   Build Command: npm run build
   Output Directory: dist
   Install Command: npm install
   ```

5. **Add Environment Variables**:
   - Click "Environment Variables"
   - Add: `VITE_API_URL` = `https://restorenet-backend.onrender.com/api`
   - Make sure to set for "Production, Preview, and Development"

6. **Click "Deploy"**

7. **Wait for deployment** (2-3 minutes)

8. **Get your frontend URL**:
   ```
   https://restorenet.vercel.app
   (or your-unique-name.vercel.app)
   ```

---

## ✅ Step 4: Verify Deployment

### 4.1 Test Backend

```bash
# Health check
curl https://restorenet-backend.onrender.com/api/health

# Expected: {"status":"ok","device":"cpu"}
```

### 4.2 Test Frontend

1. Open `https://restorenet.vercel.app` in browser
2. Click "Load Synthetic Wafer"
3. Click "RUN INFERENCE"
4. Verify you see the restored image

### 4.3 Check Browser Console

Press `F12` and check for any CORS errors. If you see:
```
Access to fetch at 'https://restorenet-backend.onrender.com/api/restore' 
from origin 'https://restorenet.vercel.app' has been blocked by CORS policy
```

→ Go back to Step 3.4 and update CORS settings.

---

## 🔧 Step 5: Configure Custom Domain (Optional)

### 5.1 Frontend Custom Domain (Vercel)

1. Go to your Vercel project → **Settings** → **Domains**
2. Add your domain (e.g., `restorenet.yourdomain.com`)
3. Follow DNS configuration instructions
4. Update CORS in backend to include your custom domain

### 5.2 Backend Custom Domain (Render)

Render free tier doesn't support custom domains. Upgrade to paid plan for this feature.

---

## 📊 Step 6: Monitor and Optimize

### 6.1 Render Dashboard

- **Logs**: View real-time logs in Render dashboard
- **Metrics**: Check CPU, memory usage
- **Events**: See deployment history

### 6.2 Vercel Analytics

- **Performance**: Check Core Web Vitals
- **Edge Network**: Monitor CDN performance
- **Real User Monitoring**: See actual user experience

### 6.3 Performance Tips

**Backend (Render)**:
- Upgrade to Starter plan for always-on service (no cold starts)
- Consider GPU instance for faster inference (paid plans only)
- Enable Redis caching for repeated requests

**Frontend (Vercel)**:
- Already optimized with Edge CDN
- Enable Analytics for monitoring
- Use Image Optimization for screenshots

---

## 🐛 Troubleshooting

### Issue 1: Backend Cold Start (Free Tier)

**Problem**: First request takes 30-60 seconds  
**Solution**: 
- Upgrade to paid plan ($7/month) for always-on
- OR implement keep-alive ping from frontend
- OR use UptimeRobot to ping every 14 minutes

**Keep-Alive Implementation**:

Add to `frontend/src/App.jsx`:

```javascript
useEffect(() => {
  // Keep backend alive (free tier only)
  const keepAlive = setInterval(async () => {
    try {
      await fetch(`${API_BASE}/health`);
    } catch (e) {
      console.log('Keep-alive ping failed');
    }
  }, 14 * 60 * 1000); // Every 14 minutes

  return () => clearInterval(keepAlive);
}, []);
```

### Issue 2: CORS Errors

**Problem**: Browser shows CORS policy errors  
**Solution**:
1. Update `allow_origins` in `src/api/main.py`
2. Push to GitHub
3. Wait for Render auto-deploy
4. Clear browser cache and test

### Issue 3: Build Fails on Render

**Problem**: Dependencies fail to install  
**Solution**:
```bash
# Check requirements.txt has exact versions
pip freeze > requirements.txt

# Or use constraints file
echo "--index-url https://pypi.org/simple" > constraints.txt
```

### Issue 4: Frontend Build Fails on Vercel

**Problem**: `npm run build` fails  
**Solution**:
```bash
# Test build locally first
cd frontend
npm install
npm run build

# Check for TypeScript errors or linting issues
npm run lint
```

### Issue 5: Large Model File

**Problem**: Git rejects large checkpoint file  
**Solution**:

```bash
# Use Git LFS for large files
git lfs install
git lfs track "checkpoints/*.pt"
git add .gitattributes
git add checkpoints/best_model.pt
git commit -m "Add model with LFS"
```

Or host model file externally:
```python
# In src/api/main.py
import requests

MODEL_URL = "https://your-storage.com/best_model.pt"

def download_model():
    response = requests.get(MODEL_URL)
    with open("checkpoints/best_model.pt", "wb") as f:
        f.write(response.content)
```

### Issue 6: Environment Variables Not Loading

**Problem**: VITE_API_URL not working  
**Solution**:
```bash
# Make sure .env.production exists in frontend/
# Rebuild on Vercel
# Clear cache: Settings → General → Clear Cache and Redeploy
```

---

## 💰 Pricing Summary

### Free Tier (Good for Development)

| Service | Free Tier Limits |
|---------|-----------------|
| **Render Backend** | 512 MB RAM, spins down after 15 min, CPU only |
| **Vercel Frontend** | Unlimited bandwidth, 100 GB/month, 100 deployments |
| **Total** | **$0/month** |

**Limitations**:
- Cold starts (30-60s first request)
- CPU inference only (slower)
- Limited concurrent requests

### Paid Tier (Production Ready)

| Service | Plan | Cost | Benefits |
|---------|------|------|----------|
| **Render Backend** | Starter | $7/month | Always-on, no cold starts, 512 MB RAM |
| **Render Backend** | Standard | $25/month | 2 GB RAM, faster CPU |
| **Vercel Frontend** | Pro | $20/month | Advanced analytics, more bandwidth |
| **Total (Basic)** | | **$7/month** | Backend always-on |
| **Total (Pro)** | | **$27/month** | Better performance |

---

## 🚀 Quick Deploy Commands

```bash
# 1. Prepare repository
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/restorenet.git
git push -u origin main

# 2. Deploy backend (via Render dashboard)
# - Connect GitHub repo
# - Set Python 3.10, start command

# 3. Deploy frontend (via Vercel dashboard)
# - Import GitHub repo
# - Set framework to Vite
# - Add VITE_API_URL env var

# 4. Test
curl https://restorenet-backend.onrender.com/api/health
# Open https://restorenet.vercel.app
```

---

## 📚 Additional Resources

- [Render Documentation](https://render.com/docs)
- [Vercel Documentation](https://vercel.com/docs)
- [FastAPI Deployment](https://fastapi.tiangolo.com/deployment/)
- [Vite Deployment](https://vitejs.dev/guide/static-deploy.html)

---

## 🎉 Success Checklist

- [ ] GitHub repository created and pushed
- [ ] Backend deployed to Render
- [ ] Backend health endpoint working
- [ ] Frontend deployed to Vercel
- [ ] Frontend loads without errors
- [ ] Can upload .npy file
- [ ] Inference completes successfully
- [ ] Metrics display correctly
- [ ] No CORS errors in browser console
- [ ] Custom domain configured (optional)
- [ ] Monitoring set up

---

**Your RestoreNet is now live! 🚀**

Share your links:
- Frontend: `https://restorenet.vercel.app`
- Backend API: `https://restorenet-backend.onrender.com/api`
- API Docs: `https://restorenet-backend.onrender.com/docs`
