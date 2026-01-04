# Smart Crash Detection System - Node.js/Express Edition

A complete vehicle crash detection system with Node.js/Express backend, PostgreSQL database, and React frontend.

## 🏗️ Architecture

```
┌─────────────────┐
│  React Frontend │ (Port 3000)
│   (Dashboard)   │
└────────┬────────┘
         │ HTTP/REST
         ▼
┌─────────────────┐
│ Node.js/Express │ (Port 3001)
│     Backend     │
└────┬────────┬───┘
     │        │
     │        └───► PostgreSQL Database
     │              (Port 5432)
     │
     │ HTTP Proxy
     ▼
┌─────────────────┐
│  Python ML      │ (Port 5000)
│    Service      │
│  (YOLO/OpenCV)  │
└─────────────────┘
```

## ✨ Features

- **Real-Time Detection**: AI-powered crash detection at 30 FPS
- **Severity Triage**: Automatic classification (Severe/Moderate/Mild)
- **Privacy Protection**: GDPR-compliant anonymization
- **Database Storage**: PostgreSQL for event history
- **Modern Dashboard**: React-based UI with WiseGuard-style design
- **RESTful API**: Node.js/Express backend
- **Real-Time Updates**: Live statistics and recent events

## 📁 Project Structure

```
Smart-Surveillance-System-for-Vehicle-Crash-Detection/
├── backend/                 # Node.js/Express API
│   ├── config/            # Database configuration
│   ├── models/            # Database models
│   ├── routes/            # API routes
│   ├── database/          # SQL schema
│   └── server.js          # Main server file
├── frontend/              # React application
│   ├── src/
│   │   ├── components/    # React components
│   │   └── App.js
│   └── package.json
├── ml-service/            # Python ML service
│   └── ml_service.py      # Video processing
└── README.md
```

## 🚀 Quick Start

See [SETUP.md](./SETUP.md) for detailed installation instructions.

### Quick Commands

```bash
# 1. Install PostgreSQL and create database
createdb crash_detection
psql -d crash_detection -f backend/database/schema.sql

# 2. Install dependencies
cd backend && npm install
cd ../frontend && npm install
pip install -r requirements.txt

# 3. Configure environment
cp backend/.env.example backend/.env
# Edit backend/.env with your PostgreSQL credentials

# 4. Start services (in separate terminals)
cd backend && npm start          # Terminal 1
cd ml-service && python ml_service.py  # Terminal 2
cd frontend && npm start         # Terminal 3
```

## 📊 API Endpoints

### System
- `GET /api/system/status` - System status
- `GET /api/system/config` - Configuration

### Crash Events
- `POST /api/crashes` - Create crash event
- `GET /api/crashes` - Get all events
- `GET /api/crashes/:id` - Get event by ID
- `GET /api/crashes/stats/summary` - Statistics
- `GET /api/crashes/recent/:hours` - Recent events

### Video
- `GET /video?conf=0.6` - MJPEG video stream

## 🗄️ Database Schema

### crash_events
- `id` - Primary key
- `severity` - Severe/Moderate/Mild
- `severity_index` - Calculated severity (0-1)
- `confidence` - Detection confidence
- `track_id` - Vehicle track ID
- `frame_number` - Frame when detected
- `location` - JSONB with coordinates
- `created_at` - Timestamp

See `backend/database/schema.sql` for full schema.

## 🎨 Frontend Features

- **Hero Section**: Main landing with start/stop detection
- **Features Grid**: System capabilities overview
- **Live Detection**: Real-time video stream with controls
- **Statistics Dashboard**: Crash event statistics
- **Recent Events**: Latest crash detections
- **Responsive Design**: Works on desktop and mobile

## 🔧 Configuration

### Backend (.env)
```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=crash_detection
DB_USER=postgres
DB_PASSWORD=your_password
PORT=3001
ML_SERVICE_URL=http://localhost:5000
```

### Frontend (.env)
```env
REACT_APP_API_URL=http://localhost:3001
```

## 📝 Development

### Backend Development
```bash
cd backend
npm run dev  # Uses nodemon for auto-reload
```

### Frontend Development
```bash
cd frontend
npm start  # Hot-reload enabled
```

## 🐛 Troubleshooting

See [SETUP.md](./SETUP.md) for common issues and solutions.

## 📄 License

MIT License

## 👥 Contributors

Your Name Here


