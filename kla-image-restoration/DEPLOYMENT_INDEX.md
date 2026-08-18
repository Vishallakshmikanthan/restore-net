# 📚 RestoreNet Deployment Resources Index

> **Quick Start**: Run `deploy-checklist.bat` → `quick-deploy.bat` → Deploy to Render & Vercel

---

## 🎯 Choose Your Path

### 🚀 Just Want to Deploy? (Fastest)
1. Read: **`DEPLOYMENT_QUICK_REFERENCE.md`** (1 page)
2. Run: **`deploy-checklist.bat`**
3. Run: **`quick-deploy.bat`**
4. Follow: Steps 2 & 3 in Quick Reference

⏱️ **Time**: 20-25 minutes

---

### 📖 Want Detailed Instructions? (Recommended)
1. Read: **`VERCEL_RENDER_DEPLOYMENT.md`** (Complete guide)
2. Run: **`deploy-checklist.bat`**
3. Follow: All steps in the guide
4. Run: **`test-deployment.bat`**

⏱️ **Time**: 30-40 minutes (first time)

---

### 🔍 Need to Understand Everything? (Deep Dive)
1. Read: **`DEPLOYMENT_SUMMARY.md`** (Overview)
2. View: **`DEPLOYMENT_FLOWCHART.txt`** (Visual guide)
3. Read: **`VERCEL_RENDER_DEPLOYMENT.md`** (Detailed steps)
4. Study: Configuration files below

⏱️ **Time**: 1-2 hours (learning + deploy)

---

## 📁 Documentation Files

| File | Purpose | When to Use |
|------|---------|-------------|
| **DEPLOYMENT_QUICK_REFERENCE.md** | 1-page cheat sheet | Quick deploy, troubleshooting |
| **DEPLOYMENT_SUMMARY.md** | Overview & next steps | After reading full guide |
| **VERCEL_RENDER_DEPLOYMENT.md** | Complete step-by-step | First-time deployment |
| **DEPLOYMENT_FLOWCHART.txt** | Visual process flow | Understanding workflow |
| **DEPLOYMENT_INDEX.md** | This file | Finding right resource |

---

## 🛠️ Helper Scripts

| Script | Purpose | When to Run |
|--------|---------|-------------|
| **deploy-checklist.bat** | Validate before deploy | Before GitHub push |
| **quick-deploy.bat** | Fast GitHub deployment | Push to GitHub |
| **test-deployment.bat** | Verify deployment | After Render + Vercel |
| **setup_deployment.bat** | Local development setup | First time only |
| **start_production.bat** | Start local servers | Local testing |

---

## ⚙️ Configuration Files

| File | Purpose | Auto-Created? |
|------|---------|---------------|
| **render.yaml** | Backend deployment config | ✅ Yes |
| **frontend/vercel.json** | Frontend build config | ✅ Yes |
| **frontend/.env.production** | Production API URL | ✅ Yes |
| **frontend/.env.example** | Env variable template | ✅ Yes |
| **.gitignore** | Git exclusion rules | ✅ Yes |

---

## 🎓 Learning Path

### For Beginners
```
1. DEPLOYMENT_FLOWCHART.txt  (5 min)  - Visual overview
2. DEPLOYMENT_QUICK_REFERENCE.md (10 min) - Quick steps
3. deploy-checklist.bat (2 min) - Validate setup
4. quick-deploy.bat (3 min) - Push to GitHub
5. Follow Render + Vercel steps (15 min)
```

### For Experienced Users
```
1. deploy-checklist.bat - Validate
2. quick-deploy.bat - Deploy
3. Render dashboard - Backend
4. Vercel dashboard - Frontend
5. test-deployment.bat - Verify
```

### For Troubleshooters
```
1. VERCEL_RENDER_DEPLOYMENT.md → Troubleshooting section
2. Check browser console (F12) for errors
3. Review Render logs
4. Review Vercel logs
5. Verify environment variables
```

---

## 🔗 Quick Links After Deployment

### Your Services
- **Frontend**: `https://YOUR-APP.vercel.app`
- **Backend API**: `https://YOUR-BACKEND.onrender.com/api`
- **API Docs**: `https://YOUR-BACKEND.onrender.com/docs`

### Dashboards
- **Render Dashboard**: https://dashboard.render.com/
- **Vercel Dashboard**: https://vercel.com/dashboard

### Documentation
- **Render Docs**: https://render.com/docs
- **Vercel Docs**: https://vercel.com/docs

---

## 🆘 Troubleshooting Guide

| Problem | Quick Fix | Detailed Guide |
|---------|-----------|----------------|
| **CORS errors** | Update `src/api/main.py` | VERCEL_RENDER_DEPLOYMENT.md → Issue 2 |
| **Cold starts** | Upgrade to paid plan | DEPLOYMENT_QUICK_REFERENCE.md → Issues |
| **Build fails** | Run `deploy-checklist.bat` | VERCEL_RENDER_DEPLOYMENT.md → Issue 3 |
| **Env vars missing** | Check `.env.production` | DEPLOYMENT_QUICK_REFERENCE.md → Config |
| **Model too large** | Use Git LFS | VERCEL_RENDER_DEPLOYMENT.md → Issue 5 |

---

## 💡 Pro Tips

### Before Deployment
- ✅ Run `deploy-checklist.bat` to catch issues early
- ✅ Test frontend build locally: `cd frontend && npm run build`
- ✅ Verify model file size (should be in checkpoints/)
- ✅ Check `.gitignore` to avoid uploading unnecessary files

### During Deployment
- ⏱️ Backend takes 5-10 minutes (first deployment)
- ⏱️ Frontend takes 2-3 minutes
- 📝 Copy your backend URL immediately after Render deploys
- 📝 Add backend URL to Vercel environment variables

### After Deployment
- 🧪 Run `test-deployment.bat` to verify
- 🔍 Check browser console (F12) for errors
- 📊 Monitor logs in Render and Vercel dashboards
- 🔄 Test with synthetic wafer data first

---

## 📊 Deployment Comparison

### Local Deployment
- **Setup**: `setup_deployment.bat`
- **Start**: `start_production.bat`
- **Access**: `http://localhost:4173`
- **Cost**: Free
- **Use Case**: Development, testing

### Cloud Deployment (Vercel + Render)
- **Setup**: `quick-deploy.bat` + dashboards
- **Access**: Public URLs
- **Cost**: Free tier or $7/month
- **Use Case**: Production, sharing, demos

### Docker Deployment
- **Setup**: `docker-compose up`
- **Access**: `http://localhost:3000`
- **Cost**: Free (your server)
- **Use Case**: Containerized environments

---

## 🎯 Decision Tree

```
Do you want to deploy?
│
├─ YES → Do you have 30+ minutes?
│        │
│        ├─ YES → Read VERCEL_RENDER_DEPLOYMENT.md
│        │        Follow all steps
│        │        Deep understanding
│        │
│        └─ NO → Read DEPLOYMENT_QUICK_REFERENCE.md
│                 Follow quick steps
│                 Deploy fast
│
└─ NO → Want to learn only?
         │
         ├─ Read DEPLOYMENT_SUMMARY.md
         ├─ View DEPLOYMENT_FLOWCHART.txt
         └─ Understand the process
```

---

## ✅ Success Checklist

Use this after deployment:

```bash
# Run test script
test-deployment.bat

# Manual checks:
□ Backend health responds: curl YOUR-BACKEND/api/health
□ Frontend loads without errors
□ Can upload .npy file
□ Inference runs successfully
□ Metrics display correctly
□ No CORS errors in console
□ Comparison slider works
□ Share link works for others
```

---

## 🚀 Next Steps After Deployment

1. **Share Your App**
   - Send link to team: `https://YOUR-APP.vercel.app`
   - Add to hackathon submission
   - Include in README

2. **Monitor Performance**
   - Check Render logs for errors
   - Monitor Vercel analytics
   - Track user feedback

3. **Consider Upgrades**
   - Render Starter ($7/month) for no cold starts
   - Custom domain for professional look
   - SSL certificate (free with both platforms)

4. **Document**
   - Update README with live URLs
   - Add screenshots of live app
   - Create demo video

---

## 📞 Get Help

### Scripts Not Working?
- Make sure you're in the right directory
- Check Python and Node.js are installed
- Run as Administrator if needed

### Deployment Failing?
- Check `VERCEL_RENDER_DEPLOYMENT.md` troubleshooting
- Review Render/Vercel logs
- Verify environment variables

### Still Stuck?
- Re-read relevant documentation section
- Check platform status pages
- Review error messages carefully

---

## 🎉 You're All Set!

Everything you need is here. Choose your path and start deploying!

**Recommended First Steps**:
1. Open `DEPLOYMENT_QUICK_REFERENCE.md`
2. Run `deploy-checklist.bat`
3. Follow the 3-step process

**Good luck with your deployment! 🚀**

---

**Team VibeSync** | **KLA Hackathon 2026**  
*Made with ❤️ by Vishal Lakshmikanthan & Sneha C*
