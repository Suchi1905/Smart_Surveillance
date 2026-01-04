const { Pool } = require('pg');
require('dotenv').config();

// Create pool with connection timeout and retry logic
const pool = new Pool({
  user: process.env.DB_USER || 'postgres',
  host: process.env.DB_HOST || 'localhost',
  database: process.env.DB_NAME || 'crash_detection',
  password: process.env.DB_PASSWORD || 'postgres',
  port: process.env.DB_PORT || 5432,
  connectionTimeoutMillis: 2000, // 2 second timeout
  idleTimeoutMillis: 30000,
});

// Test connection with error handling
pool.on('connect', () => {
  console.log('[✅] Connected to PostgreSQL database');
});

pool.on('error', (err) => {
  // Don't crash the app if DB is unavailable
  console.log('[⚠️] Database connection error (app will continue without DB):', err.message);
});

// Test connection on startup
pool.query('SELECT 1').then(() => {
  console.log('[✅] Database connection verified');
}).catch((err) => {
  console.log('[⚠️] Database not available - app will run without database storage');
  console.log('[ℹ️] To enable database: Install PostgreSQL and update backend/.env');
});

module.exports = pool;


