-- Create database (run this manually first)
-- CREATE DATABASE crash_detection;

-- Crash events table
CREATE TABLE IF NOT EXISTS crash_events (
    id SERIAL PRIMARY KEY,
    severity VARCHAR(20) NOT NULL,
    severity_index DECIMAL(5,2),
    confidence DECIMAL(5,2) NOT NULL,
    track_id INTEGER,
    frame_number INTEGER,
    location JSONB,
    image_path VARCHAR(255),
    anonymized BOOLEAN DEFAULT true,
    telegram_sent BOOLEAN DEFAULT false,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- System logs table
CREATE TABLE IF NOT EXISTS system_logs (
    id SERIAL PRIMARY KEY,
    log_level VARCHAR(20) NOT NULL,
    message TEXT NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Detection statistics table (for periodic updates)
CREATE TABLE IF NOT EXISTS detection_stats (
    id SERIAL PRIMARY KEY,
    frames_processed INTEGER DEFAULT 0,
    detections_count INTEGER DEFAULT 0,
    false_positives INTEGER DEFAULT 0,
    avg_processing_time DECIMAL(10,3),
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_crash_events_created_at ON crash_events(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_crash_events_severity ON crash_events(severity);
CREATE INDEX IF NOT EXISTS idx_crash_events_track_id ON crash_events(track_id);
CREATE INDEX IF NOT EXISTS idx_system_logs_created_at ON system_logs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_detection_stats_recorded_at ON detection_stats(recorded_at DESC);

-- Function to get recent crash statistics
CREATE OR REPLACE FUNCTION get_crash_stats(hours INTEGER DEFAULT 24)
RETURNS TABLE (
    total_events BIGINT,
    severe_count BIGINT,
    moderate_count BIGINT,
    mild_count BIGINT,
    avg_confidence NUMERIC,
    avg_severity_index NUMERIC
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        COUNT(*) as total_events,
        COUNT(*) FILTER (WHERE severity = 'Severe') as severe_count,
        COUNT(*) FILTER (WHERE severity = 'Moderate') as moderate_count,
        COUNT(*) FILTER (WHERE severity = 'Mild') as mild_count,
        AVG(confidence) as avg_confidence,
        AVG(severity_index) as avg_severity_index
    FROM crash_events
    WHERE created_at >= NOW() - (hours || ' hours')::INTERVAL;
END;
$$ LANGUAGE plpgsql;


