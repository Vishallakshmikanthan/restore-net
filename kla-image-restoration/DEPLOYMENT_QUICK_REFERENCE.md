# 🚀 RestoreNet Deployment Quick Reference

## 📋 Pre-Deployment Checklist

```bash
# Run this first
deploy-checklist.bat
```

✅ Git installed  
✅ Requirements.txt exists  
✅ Model checkpoint present  
✅ Frontend builds successfully  
✅ GitHub repository created  

---

## 🔗 Quick Deploy Steps

### 1️⃣ Push to GitHub

```bash
# Run this script
quick-deploy.bat

# Or manually:
git init
git add .
git commit -m "Deploy RestoreNet"
git remote add origin https://github.com/YOUR_USERNAME/restorenet.git
git push -u origin main
```

### 2️⃣ Deploy Backend (Render)

**URL**: https://render.com/

```yaml
Service Type: Web Service
Name: restorenet-backend
Runtime: Python 3
Build: pip install -r requirements.txt
Start: uvicorn src.api.main:app --host 0.0.0.0 --port $PORT
Branch: main
Plan: Free (or Starter for $7/month)

Environment Variables:
- PYTHON_VERSION: 3.10.0
```

⏱️ **Deploy time**: 5-10 minutes  
🔗 **Result**: `https://restorenet-backend.onrender.com`

### 3️⃣ Deploy Frontend (Vercel)

**URL**: https://vercel.com/

```yaml
Framework: Vite
Root Directory: frontend
Build Command: npm run build
Output Directory: dist
Install Command: npm install

Environment Variables:
- VITE_API_URL: https://restorenet-backend.onrender.com/api
```

⏱️ **Deploy time**: 2-3 minutes  
🔗 **Result**: `https://restorenet.vercel.app`

### 4️⃣ Test Deployment

```bash
# Run this script
test-deployment.bat

# Or manually test:
curl https://restorenet-backend.onrender.com/api/health
# Open https://restorenet.vercel.app in browser
```

---

## 🔧 Configuration Files

| File | Purpose | Required? |
|------|---------|-----------|
| `render.yaml` | Render backend config | ✅ Yes |
| `frontend/vercel.json` | Vercel frontend config | ✅ Yes |
| `frontend/.env.production` | Production API URL | ✅ Yes |
| `.gitignore` | Exclude files from git | ✅ Yes |

---

## 🐛 Common Issues & Fixes

### Issue: CORS Error

**Error**: `Access to fetch blocked by CORS policy`

**Fix**:
```python
# In src/api/main.py, update:
allow_origins=[
    "https://*.vercel.app",
    "https://YOUR-APP.vercel.app",
]
```

Then: `git push` → Wait for Render auto-deploy

---

### Issue: Backend Cold Start (Free Tier)

**Symptom**: First request takes 30-60 seconds

**Fix**: Upgrade to Starter plan ($7/month) OR add keep-alive:

```javascript
// In frontend/src/App.jsx
useEffect(() => {
  const keepAlive = setInterval(() => {
    fetch(`${API_BASE}/health`).catch(() => {});
  }, 14 * 60 * 1000); // Every 14 min
  return () => clearInterval(keepAlive);
}, []);
```

---

### Issue: Frontend Not Loading API URL

**Symptom**: API calls go to localhost instead of Render

**Fix**:
1. Check `frontend/.env.production` has correct URL
2. Rebuild on Vercel: Settings → Deployments → Redeploy
3. Clear cache: Settings → General → Clear Cache

---

### Issue: Model File Too Large

**Error**: Git push rejected (file > 100MB)

**Fix**: Use Git LFS
```bash
git lfs install
git lfs track "checkpoints/*.pt"
git add .gitattributes
git add checkpoints/best_model.pt
git commit -m "Add model with LFS"
git push
```

---

## 💰 Pricing Summary

### Free Tier
- **Render**: 512MB RAM, spins down after 15min
- **Vercel**: Unlimited, 100GB bandwidth
- **Total**: $0/month

### Recommended Production
- **Render Starter**: $7/month (always-on)
- **Vercel Free**: $0/month
- **Total**: $7/month

---

## 📊 Monitoring URLs

| Service | Dashboard | Logs |
|---------|-----------|------|
| **Render** | https://dashboard.render.com/ | Services → Logs |
| **Vercel** | https://vercel.com/dashboard | Project → Deployments |

---

## 🎯 Success Checklist

After deployment, verify:

- [ ] Backend health: `https://YOUR-BACKEND.onrender.com/api/health`
- [ ] API docs: `https://YOUR-BACKEND.onrender.com/docs`
- [ ] Frontend loads: `https://YOUR-APP.vercel.app`
- [ ] No console errors (F12)
- [ ] Upload .npy works
- [ ] Inference runs
- [ ] Metrics display
- [ ] No CORS errors

---

## 📞 Support Resources

- **Render Docs**: https://render.com/docs
- **Vercel Docs**: https://vercel.com/docs
- **Detailed Guide**: See `VERCEL_RENDER_DEPLOYMENT.md`

---

## ⚡ Quick Commands

```bash
# Check deployment readiness
deploy-checklist.bat

# Deploy to GitHub
quick-deploy.bat

# Test after deployment
test-deployment.bat

# View backend logs (Render dashboard)
# View frontend logs (Vercel dashboard)
```

---

**🎉 Your RestoreNet will be live in ~15 minutes!**

Share: `https://YOUR-APP.vercel.app`
