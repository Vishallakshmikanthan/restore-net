# RestoreNet Deployment Guide

Complete step-by-step guide to deploy RestoreNet application to production.

---

## 📋 Table of Contents
1. [Deployment Options](#deployment-options)
2. [Local Production Setup](#local-production-setup)
3. [Cloud Deployment (Recommended)](#cloud-deployment)
4. [Docker Deployment](#docker-deployment)
5. [Production Checklist](#production-checklist)

---

## 🎯 Deployment Options

### Option A: Local Production Server
- **Best for**: Internal company deployment, development testing
- **Cost**: Free (uses your infrastructure)
- **Difficulty**: Easy
- **Time**: 15-30 minutes

### Option B: Cloud Deployment (AWS/GCP/Azure)
- **Best for**: Public access, scalability, professional deployment
- **Cost**: ~$20-50/month
- **Difficulty**: Moderate
- **Time**: 1-2 hours

### Option C: Docker Container
- **Best for**: Portable deployment, microservices architecture
- **Cost**: Free (container orchestration may cost)
- **Difficulty**: Moderate
- **Time**: 30-60 minutes

---

## 🚀 Option A: Local Production Setup

### Step 1: Verify Prerequisites
```bash
# Check Python version (need 3.10+)
python --version

# Check Node.js version (need 16+)
node --version
npm --version

# Check if you have CUDA (optional, for GPU)
nvidia-smi
```

### Step 2: Backend Setup

#### 2.1 Navigate to project directory
```bash
cd c:\Users\Lenovo\Downloads\restore-net\kla-image-restoration
```

#### 2.2 Create Python virtual environment
```bash
# Create virtual environment
python -m venv .venv

# Activate it
.venv\Scripts\activate
```

#### 2.3 Install Python dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### 2.4 Verify model checkpoint exists
```bash
# Check if model file exists (should be ~4-8 MB)
dir checkpoints\best_model.pt
```

#### 2.5 Test backend API
```bash
# Start the FastAPI backend server
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

**✅ Success Check**: Open browser to `http://localhost:8000/api/health`  
You should see: `{"status":"ok","device":"cuda"}`

### Step 3: Frontend Setup

#### 3.1 Open new terminal and navigate to frontend
```bash
cd c:\Users\Lenovo\Downloads\restore-net\kla-image-restoration\frontend
```

#### 3.2 Install Node.js dependencies
```bash
npm install
```

#### 3.3 Build production frontend
```bash
npm run build
```

This creates optimized files in `frontend/dist/` folder.

#### 3.4 Preview production build
```bash
npm run preview
```

**✅ Success Check**: Open browser to `http://localhost:4173`  
You should see the RestoreNet interface.

### Step 4: Production Server Setup

#### 4.1 Install production server
```bash
# Install serve globally
npm install -g serve
```

#### 4.2 Start backend (Terminal 1)
```bash
cd c:\Users\Lenovo\Downloads\restore-net\kla-image-restoration
.venv\Scripts\activate
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 2
```

#### 4.3 Start frontend (Terminal 2)
```bash
cd c:\Users\Lenovo\Downloads\restore-net\kla-image-restoration\frontend
serve -s dist -l 3000
```

**✅ Your app is now running!**
- Frontend: `http://localhost:3000`
- Backend API: `http://localhost:8000`

### Step 5: Configure for Network Access

To allow other computers on your network to access:

#### 5.1 Find your local IP address
```bash
ipconfig
```
Look for "IPv4 Address" (e.g., `192.168.1.100`)

#### 5.2 Update frontend API endpoint
Edit `frontend/src/api/client.js`:
```javascript
// Change this line:
const API_BASE = 'http://localhost:8000/api';

// To your IP address:
const API_BASE = 'http://192.168.1.100:8000/api';
```

#### 5.3 Rebuild frontend
```bash
cd frontend
npm run build
```

#### 5.4 Restart servers with network access
```bash
# Backend
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# Frontend (new terminal)
serve -s dist -l 3000 -L
```

**✅ Now accessible at**: `http://192.168.1.100:3000` from any device on your network

---

## ☁️ Option B: Cloud Deployment

### Deployment on Render.com (Free Tier Available)

#### Step 1: Prepare Repository

1. **Create `.gitignore`** (if not exists):
```bash
echo ".venv/
__pycache__/
*.pyc
node_modules/
.env
checkpoints/*.pt
!checkpoints/best_model.pt" > .gitignore
```

2. **Push to GitHub**:
```bash
git init
git add .
git commit -m "Prepare for deployment"
git remote add origin https://github.com/YOUR_USERNAME/restore-net.git
git push -u origin main
```

#### Step 2: Deploy Backend on Render

1. Go to [render.com](https://render.com) and sign up
2. Click **"New +"** → **"Web Service"**
3. Connect your GitHub repository
4. Configure:
   - **Name**: `restorenet-backend`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn src.api.main:app --host 0.0.0.0 --port $PORT`
   - **Instance Type**: Free (or paid for better performance)

5. Add Environment Variables:
   - `PYTHON_VERSION`: `3.10.0`

6. Click **"Create Web Service"**

**✅ Success**: You'll get a URL like `https://restorenet-backend.onrender.com`

#### Step 3: Deploy Frontend on Vercel

1. Go to [vercel.com](https://vercel.com) and sign up
2. Click **"Add New"** → **"Project"**
3. Import your GitHub repository
4. Configure:
   - **Framework Preset**: Vite
   - **Root Directory**: `frontend`
   - **Build Command**: `npm run build`
   - **Output Directory**: `dist`

5. Add Environment Variable:
   - Go to **Settings** → **Environment Variables**
   - Add: `VITE_API_URL` = `https://restorenet-backend.onrender.com/api`

6. Update `frontend/src/api/client.js`:
```javascript
const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000/api';
```

7. Rebuild and deploy

**✅ Success**: You'll get a URL like `https://restorenet.vercel.app`

---

## 🐳 Option C: Docker Deployment

### Step 1: Create Dockerfile for Backend

Create `Dockerfile` in root:
```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY src/ ./src/
COPY checkpoints/ ./checkpoints/

# Expose port
EXPOSE 8000

# Start server
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Step 2: Create Dockerfile for Frontend

Create `frontend/Dockerfile`:
```dockerfile
FROM node:18-alpine AS builder

WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

### Step 3: Create Docker Compose

Create `docker-compose.yml` in root:
```yaml
version: '3.8'

services:
  backend:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./checkpoints:/app/checkpoints
    environment:
      - WORKERS=2

  frontend:
    build: ./frontend
    ports:
      - "3000:80"
    depends_on:
      - backend
    environment:
      - VITE_API_URL=http://localhost:8000/api
```

### Step 4: Build and Run

```bash
# Build containers
docker-compose build

# Start services
docker-compose up -d

# Check logs
docker-compose logs -f

# Stop services
docker-compose down
```

**✅ Access at**: `http://localhost:3000`

---

## ✅ Production Checklist

### Security
- [ ] Change default ports if exposed to internet
- [ ] Add HTTPS/SSL certificate (use Let's Encrypt)
- [ ] Set up firewall rules
- [ ] Enable CORS only for your frontend domain
- [ ] Add rate limiting to API endpoints
- [ ] Set up authentication if needed

### Performance
- [ ] Enable GPU for backend inference (if available)
- [ ] Set up CDN for frontend static files
- [ ] Configure caching headers
- [ ] Monitor memory usage
- [ ] Set up auto-scaling (cloud deployments)

### Monitoring
- [ ] Set up application logging
- [ ] Configure error tracking (e.g., Sentry)
- [ ] Add uptime monitoring (e.g., UptimeRobot)
- [ ] Set up performance monitoring
- [ ] Create health check endpoints

### Backup
- [ ] Backup model checkpoints
- [ ] Version control all code
- [ ] Document deployment process
- [ ] Create rollback plan

---

## 🔧 Troubleshooting

### Backend won't start
```bash
# Check if port 8000 is already in use
netstat -ano | findstr :8000

# Kill process if needed
taskkill /PID <PID_NUMBER> /F

# Try different port
python -m uvicorn src.api.main:app --port 8001
```

### Frontend shows CORS errors
Edit `src/api/main.py`:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://yourdomain.com"],  # Add your domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Model file not found
```bash
# Verify checkpoint exists
ls -lh checkpoints/best_model.pt

# If missing, you need to train the model first
python scripts/train.py --config configs/train.yaml
```

### Out of Memory (OOM)
- Reduce batch size in backend
- Use CPU instead of GPU (set `device="cpu"`)
- Restart the backend process
- Upgrade server memory

---

## 📞 Support & Resources

- **Documentation**: See README.md
- **Issues**: Create issue on GitHub
- **Logs**: Check `logs/` directory for errors

---

## 🎉 Quick Start Commands

### Development
```bash
# Terminal 1: Backend
cd kla-image-restoration
.venv\Scripts\activate
python -m uvicorn src.api.main:app --reload

# Terminal 2: Frontend  
cd kla-image-restoration/frontend
npm run dev
```

### Production
```bash
# Terminal 1: Backend
cd kla-image-restoration
.venv\Scripts\activate
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 2

# Terminal 2: Frontend
cd kla-image-restoration/frontend
npm run build
serve -s dist -l 3000
```

---

**Deployment Complete! 🚀**

Your RestoreNet application is now ready for production use.
