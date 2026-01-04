# Setup Guide - Node.js/Express + PostgreSQL + React

This guide will help you set up the complete system with Node.js backend, PostgreSQL database, and React frontend.

## Architecture Overview

```
React Frontend (Port 3000)
    ↓
Node.js/Express API (Port 3001)
    ├─→ PostgreSQL Database
    └─→ Python ML Service (Port 5000)
        └─→ YOLO + OpenCV Processing
```

## Prerequisites

1. **Node.js** (v14 or higher) and npm
2. **Python** 3.8+ with pip
3. **PostgreSQL** (v12 or higher)
4. **Camera** (webcam or IP camera)

## Step 1: Install PostgreSQL

### Windows:
1. Download from https://www.postgresql.org/download/windows/
2. Install with default settings
3. Remember the password you set for `postgres` user

### Linux:
```bash
sudo apt-get update
sudo apt-get install postgresql postgresql-contrib
```

### macOS:
```bash
brew install postgresql
brew services start postgresql
```

## Step 2: Create Database

1. Open PostgreSQL command line or pgAdmin
2. Run the following:

```sql
CREATE DATABASE crash_detection;
```

3. Connect to the database and run the schema:

```bash
psql -U postgres -d crash_detection -f backend/database/schema.sql
```

Or manually run the SQL from `backend/database/schema.sql`

## Step 3: Configure Environment Variables

### Backend (.env file)

Create `backend/.env`:

```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=crash_detection
DB_USER=postgres
DB_PASSWORD=your_postgres_password
PORT=3001
ML_SERVICE_URL=http://localhost:5000
```

### Frontend

Create `frontend/.env` (optional):

```env
REACT_APP_API_URL=http://localhost:3001
```

## Step 4: Install Backend Dependencies

```bash
cd backend
npm install
```

## Step 5: Install Frontend Dependencies

```bash
cd frontend
npm install
```

## Step 6: Install Python ML Service Dependencies

```bash
# From project root
pip install -r requirements.txt
```

## Step 7: Start All Services

You need to run 3 services simultaneously:

### Terminal 1: PostgreSQL (if not running as service)
```bash
# Usually runs automatically, but if needed:
# Windows: Start PostgreSQL service from Services
# Linux: sudo systemctl start postgresql
# macOS: brew services start postgresql
```

### Terminal 2: Node.js Backend
```bash
cd backend
npm start
# Or for development with auto-reload:
npm run dev
```

### Terminal 3: Python ML Service
```bash
cd ml-service
python ml_service.py
```

### Terminal 4: React Frontend
```bash
cd frontend
npm start
```

## Step 8: Access the Application

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:3001
- **ML Service**: http://localhost:5000
- **Video Stream**: http://localhost:3001/video?conf=0.6

## Verification

1. Check backend is running:
   ```bash
   curl http://localhost:3001/health
   ```

2. Check ML service:
   ```bash
   curl http://localhost:5000/api/status
   ```

3. Check database connection:
   - Backend should show: `[✅] Connected to PostgreSQL database`

## Troubleshooting

### Database Connection Issues

**Error**: `Connection refused` or `password authentication failed`

**Solution**:
1. Verify PostgreSQL is running:
   ```bash
   # Windows
   services.msc (look for PostgreSQL)
   
   # Linux
   sudo systemctl status postgresql
   ```

2. Check credentials in `backend/.env`
3. Test connection:
   ```bash
   psql -U postgres -h localhost
   ```

### ML Service Not Starting

**Error**: `Model not found`

**Solution**:
1. Ensure model file exists at `weights/best.pt`
2. Or update `MODEL_PATH` in `ml-service/ml_service.py`

### Video Stream Not Loading

**Error**: Video shows error message

**Solution**:
1. Check camera is connected
2. Verify ML service is running on port 5000
3. Check browser console for CORS errors
4. Try accessing video directly: `http://localhost:5000/video?conf=0.6`

### Frontend Can't Connect to Backend

**Error**: Network error or CORS

**Solution**:
1. Verify backend is running on port 3001
2. Check `frontend/.env` has correct `REACT_APP_API_URL`
3. Restart React dev server after changing .env

## Production Deployment

For production:

1. Build React frontend:
   ```bash
   cd frontend
   npm run build
   ```

2. Use process manager (PM2) for Node.js:
   ```bash
   npm install -g pm2
   pm2 start backend/server.js
   pm2 start ml-service/ml_service.py --interpreter python3
   ```

3. Configure PostgreSQL for production (security, backups, etc.)

## Next Steps

- Train your YOLO model using `modeltrain.py`
- Configure Telegram bot for alerts
- Set up SSL certificates for HTTPS
- Configure firewall rules

## Support

If you encounter issues:
1. Check all services are running
2. Verify environment variables
3. Check logs in each terminal
4. Ensure PostgreSQL is accessible


