const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');
const { createProxyMiddleware } = require('http-proxy-middleware');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 3001;

// Middleware
app.use(cors());
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));

// Routes
const crashEventsRoutes = require('./routes/crashEvents');
const systemRoutes = require('./routes/system');

app.use('/api/crashes', crashEventsRoutes);
app.use('/api/system', systemRoutes);

// Proxy video stream from Python ML service
const ML_SERVICE_URL = process.env.ML_SERVICE_URL || 'http://localhost:5000';
app.use('/video', createProxyMiddleware({
  target: ML_SERVICE_URL,
  changeOrigin: true,
  ws: true,
  logLevel: 'debug'
}));

// Health check
app.get('/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

// Start server
app.listen(PORT, () => {
  console.log('\n' + '='.repeat(60));
  console.log('🚗 Smart Crash Detection System - Node.js Backend');
  console.log('='.repeat(60));
  console.log(`📡 Backend API running on: http://localhost:${PORT}`);
  console.log(`🎥 Video stream proxy: http://localhost:${PORT}/video`);
  console.log(`📊 API endpoints:`);
  console.log(`   - GET  /api/system/status`);
  console.log(`   - GET  /api/crashes`);
  console.log(`   - POST /api/crashes`);
  console.log(`   - GET  /api/crashes/stats/summary`);
  console.log('='.repeat(60) + '\n');
});


