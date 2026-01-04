const express = require('express');
const router = express.Router();
const CrashEvent = require('../models/CrashEvent');

// Create new crash event (called by Python ML service)
router.post('/', async (req, res) => {
  try {
    const event = await CrashEvent.create(req.body);
    console.log(`[✅] Crash event saved to database: ${event.id}`);
    res.status(201).json(event);
  } catch (error) {
    // If database is not available, still return success but log warning
    if (error.code === 'ECONNREFUSED' || error.message.includes('timeout')) {
      console.log('[⚠️] Database not available - event logged in memory only');
      res.status(201).json({ ...req.body, id: Date.now(), saved: false, message: 'Database unavailable' });
    } else {
      console.error('[❌] Error creating crash event:', error);
      res.status(500).json({ error: 'Failed to create crash event' });
    }
  }
});

// Get all crash events
router.get('/', async (req, res) => {
  try {
    const limit = parseInt(req.query.limit) || 100;
    const offset = parseInt(req.query.offset) || 0;
    const events = await CrashEvent.getAll(limit, offset);
    res.json(events);
  } catch (error) {
    console.error('[❌] Error fetching crash events:', error);
    res.status(500).json({ error: 'Failed to fetch crash events' });
  }
});

// Get crash event by ID
router.get('/:id', async (req, res) => {
  try {
    const event = await CrashEvent.getById(req.params.id);
    if (!event) {
      return res.status(404).json({ error: 'Crash event not found' });
    }
    res.json(event);
  } catch (error) {
    console.error('[❌] Error fetching crash event:', error);
    res.status(500).json({ error: 'Failed to fetch crash event' });
  }
});

// Get events by severity
router.get('/severity/:severity', async (req, res) => {
  try {
    const events = await CrashEvent.getBySeverity(req.params.severity);
    res.json(events);
  } catch (error) {
    console.error('[❌] Error fetching events by severity:', error);
    res.status(500).json({ error: 'Failed to fetch events' });
  }
});

// Get statistics
router.get('/stats/summary', async (req, res) => {
  try {
    const stats = await CrashEvent.getStats();
    res.json(stats);
  } catch (error) {
    console.error('[❌] Error fetching stats:', error);
    res.status(500).json({ error: 'Failed to fetch statistics' });
  }
});

// Get recent events (last 24 hours by default)
router.get('/recent/:hours?', async (req, res) => {
  try {
    const hours = parseInt(req.params.hours) || 24;
    const events = await CrashEvent.getRecentEvents(hours);
    res.json(events);
  } catch (error) {
    console.error('[❌] Error fetching recent events:', error);
    res.status(500).json({ error: 'Failed to fetch recent events' });
  }
});

module.exports = router;


