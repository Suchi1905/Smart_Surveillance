import React from 'react';
import './Hero.css';

const Hero = ({ isStreaming, onStreamingChange }) => {
  return (
    <section id="home" className="hero">
      <div className="container">
        <div className="hero-content">
          <div className="hero-text">
            <h1 className="hero-title">Unique & Trusted Crash Detection Solutions</h1>
            <p className="hero-subtitle">
              Advanced AI-powered vehicle crash detection system with real-time monitoring, 
              severity triage, and privacy-preserving anonymization.
            </p>
            <div className="hero-buttons">
              <button 
                className="btn btn-primary"
                onClick={() => onStreamingChange(!isStreaming)}
              >
                {isStreaming ? '⏹ Stop Detection' : '▶ Start Detection'}
              </button>
              <a href="#detection" className="btn btn-secondary">Learn More</a>
            </div>
          </div>
          <div className="hero-visual">
            <div className="camera-visual">
              <div className="camera-icon">📹</div>
              <div className="camera-status">
                {isStreaming ? (
                  <span className="status-active">● Live</span>
                ) : (
                  <span className="status-inactive">○ Offline</span>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Hero;


