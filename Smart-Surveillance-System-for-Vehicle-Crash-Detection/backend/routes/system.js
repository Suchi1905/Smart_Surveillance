const express = require('express');
const router = express.Router();
const axios = require('axios');

const ML_SERVICE_URL = process.env.ML_SERVICE_URL || 'http://localhost:5000';

// Get system status
router.get('/status', async (req, res) => {
  try {
    // Check ML service status
    let mlStatus = { available: false };
    try {
      const response = await axios.get(`${ML_SERVICE_URL}/api/status`, { timeout: 2000 });
      mlStatus = { ...response.data, available: true };
    } catch (error) {
      console.log('[⚠️] ML service not available');
    }

    // Check database status
    let dbStatus = { connected: false };
    try {
      const pool = require('../config/database');
      await Promise.race([
        pool.query('SELECT 1'),
        new Promise((_, reject) => setTimeout(() => reject(new Error('timeout')), 2000))
      ]);
      dbStatus = { connected: true };
    } catch (error) {
      // Database not available - app continues without it
      dbStatus = { connected: false, message: 'Database optional' };
    }

    res.json({
      ml_service: mlStatus,
      database: dbStatus,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    res.status(500).json({ error: 'Failed to get system status' });
  }
});

// Get system configuration
router.get('/config', async (req, res) => {
  try {
    let mlConfig = {};
    try {
      const response = await axios.get(`${ML_SERVICE_URL}/api/config`, { timeout: 2000 });
      mlConfig = response.data;
    } catch (error) {
      console.log('[⚠️] ML service not available');
    }

    res.json({
      ml_service: mlConfig,
      database: {
        host: process.env.DB_HOST || 'localhost',
        name: process.env.DB_NAME || 'crash_detection'
      }
    });
  } catch (error) {
    res.status(500).json({ error: 'Failed to get configuration' });
  }
});

module.exports = router;


