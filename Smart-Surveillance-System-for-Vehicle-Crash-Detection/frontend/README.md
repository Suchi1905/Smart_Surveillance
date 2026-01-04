# React Frontend for Crash Detection System

This is the React frontend for the Privacy-Preserving Vehicle Crash Detection System.

## Installation

```bash
npm install
```

## Development

Start the development server:

```bash
npm start
```

The app will open at [http://localhost:3000](http://localhost:3000)

The frontend will automatically proxy API requests to the Flask backend running on `http://localhost:5000`.

## Building for Production

Build the React app for production:

```bash
npm run build
```

This creates an optimized production build in the `build` folder. The Flask backend can serve these static files.

## Project Structure

```
frontend/
├── public/          # Static files
├── src/
│   ├── components/  # React components
│   │   ├── VideoStream.js
│   │   ├── ControlPanel.js
│   │   └── StatusPanel.js
│   ├── App.js       # Main app component
│   └── index.js     # Entry point
└── package.json
```

## Features

- **Real-time Video Streaming**: MJPEG stream from Flask backend
- **Confidence Control**: Adjustable detection threshold slider
- **System Status**: Real-time status indicators
- **Responsive Design**: Works on desktop and mobile devices
- **Modern UI**: Dark theme with smooth animations

## API Endpoints

The frontend communicates with the Flask backend via:

- `GET /api/status` - Get system status
- `GET /api/config` - Get configuration
- `GET /video?conf=0.6` - MJPEG video stream


