import React, { useState, useEffect } from 'react';
import './CameraCard.css';

const CameraCard = ({ camera, detection, apiUrl }) => {
  const [isLive, setIsLive] = useState(true);
  const [streamError, setStreamError] = useState(false);

  useEffect(() => {
    const img = new Image();
    img.onload = () => {
      setIsLive(true);
      setStreamError(false);
    };
    img.onerror = () => {
      setIsLive(false);
      setStreamError(true);
    };
    img.src = `${apiUrl}/video?conf=${camera.confidence}`;
  }, [apiUrl, camera.confidence]);

  const getSeverityColor = (severity) => {
    switch (severity) {
      case 'Severe':
        return 'var(--severity-severe)';
      case 'Moderate':
        return 'var(--severity-moderate)';
      case 'Mild':
        return 'var(--severity-mild)';
      default:
        return 'transparent';
    }
  };

  const hasSeverityAlert = detection && detection.severity !== 'Monitoring';

  const formatTimestamp = () => {
    const now = new Date();
    return {
      date: now.toLocaleDateString('en-GB', { day: '2-digit', month: '2-digit', year: 'numeric' }),
      time: now.toLocaleTimeString('en-US', { hour12: true, hour: '2-digit', minute: '2-digit', second: '2-digit' })
    };
  };

  const timestamp = formatTimestamp();

  return (
    <div className="camera-card">
      {/* Live Badge - Top Right with pulsing animation */}
      {isLive && (
        <div className="live-badge">
          <span className="live-dot-pulse"></span>
          <span className="live-text">LIVE</span>
        </div>
      )}
      
      {/* Camera Feed */}
      <div className="camera-feed">
        {streamError ? (
          <div className="feed-error">
            <div className="error-icon">⚠️</div>
            <p>Stream Unavailable</p>
            <p className="error-hint">Check ML service connection</p>
          </div>
        ) : (
          <img
            src={`${apiUrl}/video?conf=${camera.confidence}`}
            alt={`${camera.id} feed`}
            className="feed-image"
            onError={() => setStreamError(true)}
          />
        )}
        
        {/* Severity Alert Area - Glows red only when accident detected */}
        {hasSeverityAlert && (
          <div 
            className="severity-alert-area"
            style={{ 
              backgroundColor: getSeverityColor(detection.severity),
              boxShadow: `0 0 25px ${getSeverityColor(detection.severity)}`
            }}
          >
            <span className="alert-icon">🚨</span>
            <span className="alert-text">{detection.severity} ACCIDENT</span>
            <span className="alert-confidence">{(detection.confidence * 100).toFixed(0)}%</span>
          </div>
        )}

        {/* Timestamp overlay - Top Left */}
        <div className="timestamp-overlay">
          <div className="timestamp-text">{camera.id}</div>
          <div className="timestamp-text">{timestamp.date}</div>
          <div className="timestamp-text">{timestamp.time}</div>
        </div>

        {/* WiFi/Battery Icons - Top Left (like evizz) */}
        <div className="status-icons">
          <span className="status-icon wifi">📶</span>
          <span className="status-icon battery">🔋</span>
        </div>
      </div>
      
      {/* Camera Identifier - Bottom Left (CAM-01 - North Entrance) */}
      <div className="camera-identifier">
        <span className="camera-id">{camera.id}</span>
        <span className="camera-separator">-</span>
        <span className="camera-name">{camera.name}</span>
      </div>
    </div>
  );
};

export default CameraCard;
