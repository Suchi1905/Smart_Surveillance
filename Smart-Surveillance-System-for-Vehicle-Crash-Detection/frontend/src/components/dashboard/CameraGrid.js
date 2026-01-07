import React, { useState, useEffect } from 'react';
import './CameraGrid.css';
import CameraCard from './CameraCard';

const CameraGrid = ({ selectedLocation, onLocationChange, emergencyMode, apiUrl }) => {
  const [cameras] = useState({
    'All': [
      { id: 'CAM-01', name: 'Main Entrance', location: 'Basement', confidence: 0.6 },
      { id: 'CAM-02', name: 'Highway North', location: 'Backyard', confidence: 0.6 },
      { id: 'CAM-03', name: 'Highway South', location: 'Front Door', confidence: 0.6 },
      { id: 'CAM-04', name: 'Intersection A', location: 'Kitchen', confidence: 0.6 }
    ],
    'Basement': [
      { id: 'CAM-01', name: 'Main Entrance', location: 'Basement', confidence: 0.6 }
    ],
    'Backyard': [
      { id: 'CAM-02', name: 'Highway North', location: 'Backyard', confidence: 0.6 }
    ],
    'Front Door': [
      { id: 'CAM-03', name: 'Highway South', location: 'Front Door', confidence: 0.6 }
    ],
    'Kitchen': [
      { id: 'CAM-04', name: 'Intersection A', location: 'Kitchen', confidence: 0.6 }
    ]
  });

  const [detections, setDetections] = useState({});

  useEffect(() => {
    // Simulate detection updates
    const interval = setInterval(() => {
      setDetections(prev => {
        const newDetections = { ...prev };
        cameras[selectedLocation]?.forEach(cam => {
          // Randomly simulate detections
          if (Math.random() > 0.95) {
            const severities = ['Severe', 'Moderate', 'Mild', 'Monitoring'];
            const severity = severities[Math.floor(Math.random() * severities.length)];
            newDetections[cam.id] = {
              severity,
              confidence: Math.random() * 0.3 + 0.7,
              timestamp: new Date()
            };
          }
        });
        return newDetections;
      });
    }, 5000);

    return () => clearInterval(interval);
  }, [selectedLocation]);

  const locations = ['All', 'Basement', 'Backyard', 'Front Door', 'Kitchen'];

  return (
    <main className="camera-grid-container">
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

      {locations.filter(loc => loc !== 'All').map(location => {
        const locationCameras = cameras[location] || [];
        if (selectedLocation !== 'All' && selectedLocation !== location) {
          return null;
        }
        if (locationCameras.length === 0) {
          return null;
        }

        return (
          <div key={location} className="location-section">
            <h2 className="location-heading">{location}</h2>
            <div className="camera-grid">
              {locationCameras.map(camera => (
                <CameraCard
                  key={camera.id}
                  camera={camera}
                  detection={detections[camera.id]}
                  emergencyMode={emergencyMode}
                  apiUrl={apiUrl}
                />
              ))}
            </div>
          </div>
        );
      })}
    </main>
  );
};

export default CameraGrid;
