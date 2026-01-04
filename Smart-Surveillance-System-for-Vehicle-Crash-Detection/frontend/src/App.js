import React, { useState, useEffect } from 'react';
import './App.css';
import Sidebar from './components/dashboard/Sidebar';
import Header from './components/dashboard/Header';
import CameraGrid from './components/dashboard/CameraGrid';
import ActivityFeed from './components/dashboard/ActivityFeed';
import Footer from './components/dashboard/Footer';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:3001';

function App() {
  const [systemStatus, setSystemStatus] = useState({
    healthy: true,
    emergencyMode: false
  });
  const [fps, setFps] = useState(30);
  const [latency, setLatency] = useState(0);
  const [networkStatus, setNetworkStatus] = useState('Connected');
  const [selectedLocation, setSelectedLocation] = useState('All');
  const [recentEvents, setRecentEvents] = useState([]);

  useEffect(() => {
    // Update FPS and latency
    const metricsInterval = setInterval(() => {
      setFps(prev => {
        const variation = Math.random() * 2 - 1;
        return Math.max(25, Math.min(35, prev + variation));
      });
      setLatency(Math.floor(Math.random() * 10 + 15));
    }, 2000);

    // Check network status
    const checkNetwork = async () => {
      try {
        const response = await fetch(`${API_URL}/health`, { 
          method: 'GET',
          signal: AbortSignal.timeout(3000)
        });
        if (response.ok) {
          setNetworkStatus('Connected');
        } else {
          setNetworkStatus('Warning');
        }
      } catch (error) {
        setNetworkStatus('Disconnected');
      }
    };

    // Fetch system status
    const fetchStatus = async () => {
      try {
        const response = await fetch(`${API_URL}/api/system/status`);
        const data = await response.json();
        setSystemStatus(prev => ({
          ...prev,
          healthy: data.ml_service?.available && data.database?.connected
        }));
      } catch (error) {
        console.error('Error fetching status:', error);
        setSystemStatus(prev => ({ ...prev, healthy: false }));
      }
    };

    // Fetch recent events
    const fetchEvents = async () => {
      try {
        const response = await fetch(`${API_URL}/api/crashes/recent/24`);
        const data = await response.json();
        setRecentEvents(data.slice(0, 20));
      } catch (error) {
        // If database not available, use mock data
        setRecentEvents([]);
      }
    };

    fetchStatus();
    fetchEvents();
    checkNetwork();
    const statusInterval = setInterval(() => {
      fetchStatus();
      fetchEvents();
      checkNetwork();
    }, 10000);

    return () => {
      clearInterval(metricsInterval);
      clearInterval(statusInterval);
    };
  }, [API_URL]);

  return (
    <div className="app-container">
      <Sidebar />
      <div className="main-content">
        <Header systemStatus={systemStatus} />
        <div className="content-wrapper">
          <CameraGrid 
            selectedLocation={selectedLocation}
            onLocationChange={setSelectedLocation}
            emergencyMode={systemStatus.emergencyMode}
            apiUrl={API_URL}
          />
          <ActivityFeed events={recentEvents} />
        </div>
        <Footer 
          fps={fps}
          latency={latency}
          networkStatus={networkStatus}
        />
      </div>
    </div>
  );
}

export default App;
