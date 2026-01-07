# 🚀 START HERE - Run the Integrated System

## ✅ Integration Complete!

The ML model from `crash_webapp` is now integrated with the frontend.

---

## ⚡ Quick Start (2 Steps)

### Step 1: Start Backend (Flask)

**Terminal 1:**
```bash
cd "Smart_Surveillance/Smart-Surveillance-System-for-Vehicle-Crash-Detection"
python app.py
```

✅ **Expected Output:**
```
✅ Found model at: crash_webapp/weights/best.pt
✅ Model loaded successfully
📡 Backend running on: http://localhost:5000
```

### Step 2: Start Frontend (React)

**Terminal 2:**
```bash
cd frontend
npm install  # First time only
npm start
```

✅ **Expected Output:**
```
Compiled successfully!
Local: http://localhost:3000
```

---

## 🎯 Access Points

- **Frontend Dashboard**: http://localhost:3000 ⭐
- **Backend API**: http://localhost:5000
- **Health Check**: http://localhost:5000/health
- **Video Stream**: http://localhost:5000/video?conf=0.6

---

## 📋 What's Integrated

✅ ML Model: `crash_webapp/weights/best.pt`  
✅ Backend: Flask (port 5000)  
✅ Frontend: React (port 3000)  
✅ All API endpoints connected  
✅ Video streaming enabled  

---

## 🆘 Quick Troubleshooting

**Model not found?**
- Check: `crash_webapp/weights/best.pt` exists
- Backend will use YOLOv8n as fallback

**Frontend can't connect?**
- Verify backend is running on port 5000
- Check browser console for errors

**Port conflicts?**
- Backend: Change port in `app.py`
- Frontend: React will prompt for alternative port

---

## 📚 Full Documentation

- **INTEGRATION_GUIDE.md** - Complete integration details
- **ROADMAP.md** (in crash_webapp) - Full setup guide

---

**Ready to run!** 🎉






