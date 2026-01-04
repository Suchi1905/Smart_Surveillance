import React, { useState, useEffect } from 'react';
import './Header.css';

const Header = ({ systemStatus, selectedLocation, onLocationChange }) => {
  const [currentTime, setCurrentTime] = useState(new Date());

  useEffect(() => {
    const timer = setInterval(() => {
      setCurrentTime(new Date());
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  const locations = ['All', 'Basement', 'Backyard', 'Front Door', "Kid's Room", 'Kitchen'];

  const formatTime = (date) => {
    return date.toLocaleTimeString('en-US', {
      hour12: true,
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit'
    });
  };

  return (
    <header className="dashboard-header">
      <div className="header-left">
        <div className="header-logo">
          <span className="logo-text">SmartGuard</span>
          <span className="logo-subtitle">Surveillance System</span>
        </div>
      </div>
      
      <div className="header-center">
        <div className="location-tabs">
          {locations.map(location => (
            <button
              key={location}
              className={`location-tab ${selectedLocation === location ? 'active' : ''}`}
              onClick={() => onLocationChange(location)}
            >
              {location}
            </button>
          ))}
        </div>
      </div>
      
      <div className="header-right">
        <div className="digital-clock">
          <span className="clock-time">{formatTime(currentTime)}</span>
        </div>
        <div className={`status-pill ${systemStatus.aiEngineActive ? 'active' : 'inactive'}`}>
          <span className="status-dot"></span>
          <span className="status-text">AI Engine: {systemStatus.aiEngineActive ? 'Active' : 'Inactive'}</span>
        </div>
      </div>
    </header>
  );
};

export default Header;
