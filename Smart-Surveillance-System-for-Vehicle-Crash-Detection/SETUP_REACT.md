# React Frontend Setup Guide

This guide will help you set up and run the React frontend for the Crash Detection System.

## Prerequisites

- Node.js (v14 or higher) and npm installed
- Python 3.8+ with Flask backend running

## Quick Start

### 1. Install Frontend Dependencies

Navigate to the frontend directory and install dependencies:

```bash
cd frontend
npm install
```

### 2. Start the Flask Backend

In a separate terminal, start the Flask backend:

```bash
# From the project root directory
python app.py
```

The backend will run on `http://localhost:5000`

### 3. Start the React Development Server

In the frontend directory:

```bash
npm start
```

The React app will automatically open at `http://localhost:3000`

## Development Workflow

1. **Backend (Flask)**: Handles video processing, detection, and API endpoints
   - Runs on port 5000
   - Provides `/video` stream endpoint
   - Provides `/api/status` and `/api/config` endpoints

2. **Frontend (React)**: Provides the user interface
   - Runs on port 3000 (development)
   - Automatically proxies API calls to backend
   - Hot-reloads on code changes

## Production Build

To build the React app for production:

```bash
cd frontend
npm run build
```

This creates an optimized build in `frontend/build/`. The Flask backend is configured to serve these static files in production mode.

## Troubleshooting

### Port Already in Use

If port 3000 is already in use, React will prompt you to use a different port. You can also specify a port:

```bash
PORT=3001 npm start
```

### Backend Connection Issues

If the frontend can't connect to the backend:

1. Ensure Flask backend is running on port 5000
2. Check that CORS is enabled in `app.py` (flask-cors installed)
3. Verify the proxy setting in `package.json` points to `http://localhost:5000`

### Video Stream Not Loading

1. Check browser console for errors
2. Verify camera is connected and accessible
3. Ensure model file exists at `weights/best.pt`
4. Check Flask backend logs for errors

## Project Structure

```
Smart-Surveillance-System-for-Vehicle-Crash-Detection/
├── app.py                    # Flask backend API
├── frontend/                 # React frontend
│   ├── public/
│   ├── src/
│   │   ├── components/       # React components
│   │   │   ├── VideoStream.js
│   │   │   ├── ControlPanel.js
│   │   │   └── StatusPanel.js
│   │   ├── App.js
│   │   └── index.js
│   └── package.json
└── requirements.txt
```

## Features

- ✅ Real-time video streaming
- ✅ Adjustable confidence threshold
- ✅ System status indicators
- ✅ Responsive design
- ✅ Modern dark theme UI
- ✅ Error handling and loading states

## Next Steps

1. Install dependencies: `cd frontend && npm install`
2. Start backend: `python app.py`
3. Start frontend: `cd frontend && npm start`
4. Open browser to `http://localhost:3000`

Enjoy your React-powered crash detection system! 🚗


