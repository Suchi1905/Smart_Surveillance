# 🔗 Integration Guide: ML Model + Frontend

## ✅ Integration Complete!

The ML model from `crash_webapp` has been successfully integrated with the frontend in the parent directory.

---

## 🔧 What Was Changed

### 1. **Backend (app.py)**
- ✅ Updated to load model from `crash_webapp/weights/best.pt`
- ✅ Auto-detects model in multiple locations
- ✅ Added missing endpoints:
  - `/health` - Health check
  - `/api/system/status` - System status for dashboard
  - `/api/crashes/recent/<hours>` - Recent crashes endpoint

### 2. **Frontend**
- ✅ Updated API_URL from `localhost:3001` to `localhost:5000`
- ✅ Updated proxy in `package.json` to point to port 5000
- ✅ All components now connect to Flask backend

---

## 🚀 How to Run

### Step 1: Start Backend (Flask)

```bash
cd "Smart_Surveillance/Smart-Surveillance-System-for-Vehicle-Crash-Detection"
python app.py
```

**Expected Output:**
```
✅ Found model at: crash_webapp/weights/best.pt
✅ Model loaded successfully from crash_webapp/weights/best.pt
📡 Backend running on: http://localhost:5000
```

### Step 2: Start Frontend (React)

**Terminal 2:**
```bash
cd frontend
npm install  # First time only
npm start
```

**Expected Output:**
```
Compiled successfully!
Local: http://localhost:3000
```

### Step 3: Access Application

- **Frontend Dashboard**: http://localhost:3000
- **Backend API**: http://localhost:5000
- **Health Check**: http://localhost:5000/health
- **System Status**: http://localhost:5000/api/system/status

---

## 📊 Architecture

```
┌─────────────────────┐
│  React Frontend     │  Port 3000
│  (Parent Directory) │
│  - Dashboard        │
│  - Camera Grid      │
│  - Live Detection   │
└──────────┬──────────┘
           │
           │ HTTP/REST (proxy)
           │ localhost:5000
           ▼
┌─────────────────────┐
│  Flask Backend      │  Port 5000
│  (app.py)           │
│  - Video Stream     │
│  - API Endpoints    │
└──────────┬──────────┘
           │
           │ Loads Model
           ▼
┌─────────────────────┐
│  ML Model           │
│  crash_webapp/      │
│  weights/best.pt    │
│  (YOLOv8)           │
└─────────────────────┘
```

---

## 🔍 Model Loading Priority

The backend automatically searches for the model in this order:

1. `crash_webapp/weights/best.pt` ✅ **Primary Location**
2. `weights/best.pt` (local)
3. Absolute path resolution
4. Falls back to YOLOv8n if not found

---

## 📝 API Endpoints

### Backend Endpoints (Flask - Port 5000)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/status` | GET | Basic status |
| `/api/system/status` | GET | System status (for dashboard) |
| `/api/crashes/recent/<hours>` | GET | Recent crash events |
| `/api/config` | GET | Configuration |
| `/video?conf=0.6` | GET | MJPEG video stream |

---

## ✅ Verification Checklist

- [x] Backend loads model from `crash_webapp/weights/best.pt`
- [x] Frontend connects to backend on port 5000
- [x] All required endpoints added
- [x] Video streaming works
- [x] Dashboard can fetch system status

---

## 🐛 Troubleshooting

### Model Not Found

**Error**: `⚠️ Warning: Model not found`

**Solution**:
1. Verify `crash_webapp/weights/best.pt` exists
2. Check file permissions
3. Backend will fallback to YOLOv8n

### Frontend Can't Connect

**Error**: `Failed to fetch` or `Network error`

**Solution**:
1. Ensure backend is running on port 5000
2. Check `package.json` proxy setting
3. Verify CORS is enabled in Flask

### Port Conflicts

**Error**: `Address already in use`

**Solution**:
- Change Flask port in `app.py`: `app.run(port=5001)`
- Update frontend `API_URL` and proxy accordingly

---

## 📚 Files Modified

1. `app.py` - Model loading and new endpoints
2. `frontend/package.json` - Proxy configuration
3. `frontend/src/App.js` - API_URL
4. `frontend/src/components/LiveDetection.js` - API_URL

---

## 🎉 Status

**Integration Status**: ✅ **COMPLETE**

The ML model from `crash_webapp` is now fully integrated with the parent frontend. You can:

- ✅ Run the frontend dashboard
- ✅ See live video detection
- ✅ Monitor system status
- ✅ View crash detections

**Ready to use!** 🚀






