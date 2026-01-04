import React, { useRef, useEffect, useState } from 'react';
import './VideoStream.css';

const VideoStream = ({ confidence }) => {
  const videoRef = useRef(null);
  const [error, setError] = useState(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    if (!videoRef.current) return;

    const video = videoRef.current;
    const streamUrl = `http://localhost:5000/video?conf=${confidence}`;
    
    // Create image element to load the stream
    const img = new Image();
    img.src = streamUrl;
    
    setIsLoading(true);
    setError(null);

    const handleLoad = () => {
      setIsLoading(false);
      if (video) {
        // For MJPEG stream, we'll use an img tag instead
        video.src = streamUrl;
      }
    };

    const handleError = (e) => {
      setIsLoading(false);
      setError('Failed to load video stream. Make sure the backend is running.');
      console.error('Video stream error:', e);
    };

    img.onload = handleLoad;
    img.onerror = handleError;

    // Cleanup
    return () => {
      img.onload = null;
      img.onerror = null;
    };
  }, [confidence]);

  return (
    <div className="video-stream-container">
      <h2>Live Detection Feed</h2>
      <div className="video-wrapper">
        {isLoading && (
          <div className="loading-overlay">
            <div className="spinner"></div>
            <p>Connecting to video stream...</p>
          </div>
        )}
        {error && (
          <div className="error-overlay">
            <p>⚠️ {error}</p>
          </div>
        )}
        <img
          ref={videoRef}
          src={`http://localhost:5000/video?conf=${confidence}`}
          alt="Live detection feed"
          className="video-stream"
          onLoad={() => setIsLoading(false)}
          onError={() => {
            setIsLoading(false);
            setError('Failed to load video stream');
          }}
        />
      </div>
      <div className="video-info">
        <p>Confidence Threshold: <strong>{confidence.toFixed(1)}</strong></p>
        <p className="info-text">The video stream shows real-time crash detection with severity triage analysis.</p>
      </div>
    </div>
  );
};

export default VideoStream;


