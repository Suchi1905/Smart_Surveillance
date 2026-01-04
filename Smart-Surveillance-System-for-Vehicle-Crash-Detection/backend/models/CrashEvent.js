const pool = require('../config/database');

class CrashEvent {
  static async create(eventData) {
    const {
      severity,
      severity_index,
      confidence,
      track_id,
      frame_number,
      location,
      image_path,
      anonymized = true
    } = eventData;

    const query = `
      INSERT INTO crash_events 
      (severity, severity_index, confidence, track_id, frame_number, location, image_path, anonymized, created_at)
      VALUES ($1, $2, $3, $4, $5, $6, $7, $8, NOW())
      RETURNING *
    `;

    const values = [
      severity,
      severity_index,
      confidence,
      track_id,
      frame_number,
      JSON.stringify(location),
      image_path,
      anonymized
    ];

    try {
      const result = await pool.query(query, values);
      return result.rows[0];
    } catch (error) {
      console.error('Error creating crash event:', error);
      throw error;
    }
  }

  static async getAll(limit = 100, offset = 0) {
    const query = `
      SELECT * FROM crash_events 
      ORDER BY created_at DESC 
      LIMIT $1 OFFSET $2
    `;

    try {
      const result = await pool.query(query, [limit, offset]);
      return result.rows;
    } catch (error) {
      console.error('Error fetching crash events:', error);
      throw error;
    }
  }

  static async getById(id) {
    const query = 'SELECT * FROM crash_events WHERE id = $1';
    
    try {
      const result = await pool.query(query, [id]);
      return result.rows[0];
    } catch (error) {
      console.error('Error fetching crash event:', error);
      throw error;
    }
  }

  static async getBySeverity(severity, limit = 50) {
    const query = `
      SELECT * FROM crash_events 
      WHERE severity = $1 
      ORDER BY created_at DESC 
      LIMIT $2
    `;

    try {
      const result = await pool.query(query, [severity, limit]);
      return result.rows;
    } catch (error) {
      console.error('Error fetching crash events by severity:', error);
      throw error;
    }
  }

  static async getStats() {
    const query = `
      SELECT 
        COUNT(*) as total_events,
        COUNT(*) FILTER (WHERE severity = 'Severe') as severe_count,
        COUNT(*) FILTER (WHERE severity = 'Moderate') as moderate_count,
        COUNT(*) FILTER (WHERE severity = 'Mild') as mild_count,
        AVG(confidence) as avg_confidence,
        AVG(severity_index) as avg_severity_index,
        MAX(created_at) as last_event
      FROM crash_events
    `;

    try {
      const result = await pool.query(query);
      return result.rows[0];
    } catch (error) {
      console.error('Error fetching stats:', error);
      throw error;
    }
  }

  static async getRecentEvents(hours = 24) {
    const query = `
      SELECT * FROM crash_events 
      WHERE created_at >= NOW() - INTERVAL '${hours} hours'
      ORDER BY created_at DESC
    `;

    try {
      const result = await pool.query(query);
      return result.rows;
    } catch (error) {
      console.error('Error fetching recent events:', error);
      throw error;
    }
  }
}

module.exports = CrashEvent;


