import React, { useState, useRef, useEffect } from 'react';
import './LiveDetection.css';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

const LiveDetection = ({ isStreaming, onStreamingChange }) => {
  const [confidence, setConfidence] = useState(0.6);
  const [error, setError] = useState(null);
  const videoRef = useRef(null);

  useEffect(() => {
    if (!isStreaming) return;

    // Create image element for MJPEG stream
    const img = new Image();
    const streamUrl = `${API_URL}/video?conf=${confidence}`;
    
    img.onload = () => {
      setError(null);
      if (videoRef.current) {
        videoRef.current.src = streamUrl;
      }
    };

    img.onerror = () => {
      setError('Failed to load video stream. Make sure the ML service is running.');
    };

    img.src = streamUrl;
  }, [isStreaming, confidence]);

  return (
    <section id="detection" className="live-detection">
      <div className="container">
        <div className="detection-content">
          <div className="detection-text">
            <h2 className="section-title">Keep An Eye On Everything With Our Video Security!</h2>
            <p className="detection-description">
              Real-time vehicle crash detection with AI-powered analysis. Monitor traffic incidents 
              with automatic severity classification and privacy-preserving anonymization.
            </p>
            <ul className="detection-features">
              <li>✓ 24/7 Real-Time Monitoring</li>
              <li>✓ AI-Powered Crash Detection</li>
              <li>✓ Automatic Severity Classification</li>
              <li>✓ Privacy-Compliant Anonymization</li>
              <li>✓ Instant Alert Notifications</li>
              <li>✓ Historical Event Playback</li>
            </ul>
            <div className="confidence-control">
              <label htmlFor="confidence">
                Confidence Threshold: <strong>{confidence.toFixed(1)}</strong>
              </label>
              <input
                type="range"
                id="confidence"
                min="0.1"
                max="1.0"
                step="0.1"
                value={confidence}
                onChange={(e) => setConfidence(parseFloat(e.target.value))}
                className="slider"
              />
            </div>
            <button 
              className="btn btn-primary"
              onClick={() => onStreamingChange(!isStreaming)}
            >
              {isStreaming ? '⏹ Stop Detection' : '▶ Start Detection'}
            </button>
          </div>
          <div className="detection-video">
            <div className="video-wrapper">
              {error && (
                <div className="video-error">
                  <p>⚠️ {error}</p>
                  <p className="error-hint">Make sure the ML service is running on port 5000</p>
                </div>
              )}
              {!isStreaming && !error && (
                <div className="video-placeholder">
                  <div className="placeholder-icon">📹</div>
                  <p>Click "Start Detection" to begin monitoring</p>
                </div>
              )}
              <img
                ref={videoRef}
                src={isStreaming ? `${API_URL}/video?conf=${confidence}` : ''}
                alt="Live detection feed"
                className="video-stream"
                style={{ display: isStreaming && !error ? 'block' : 'none' }}
              />
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default LiveDetection;


