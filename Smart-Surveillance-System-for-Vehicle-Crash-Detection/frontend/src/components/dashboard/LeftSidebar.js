import React, { useState } from 'react';
import './LeftSidebar.css';

const LeftSidebar = () => {
  const [activeItem, setActiveItem] = useState('live-view');
  const [settingsOpen, setSettingsOpen] = useState(false);

  const menuItems = [
    { id: 'live-view', icon: '🏠', label: 'Live View' },
    { id: 'incident-logs', icon: '📋', label: 'Incident Logs' },
    { id: 'settings', icon: '⚙️', label: 'Settings' }
  ];

  return (
    <aside className={`left-sidebar ${settingsOpen ? 'settings-open' : ''}`}>
      <div className="sidebar-header">
        <div className="sidebar-logo">
          <span className="logo-text">SmartGuard</span>
        </div>
        {settingsOpen && (
          <button 
            className="close-settings"
            onClick={() => setSettingsOpen(false)}
          >
            ✕
          </button>
        )}
      </div>
      
      {!settingsOpen ? (
        <>
          <div className="sidebar-quick-icons">
            <button className="icon-btn" title="Home">
              <span>🏠</span>
            </button>
            <button className="icon-btn" title="Gallery">
              <span>🖼️</span>
            </button>
            <button className="icon-btn" title="Notifications">
              <span>🔔</span>
            </button>
            <button className="icon-btn" title="Add">
              <span>➕</span>
            </button>
          </div>
          
          <nav className="sidebar-nav">
            {menuItems.map(item => (
              <button
                key={item.id}
                className={`nav-item ${activeItem === item.id ? 'active' : ''}`}
                onClick={() => {
                  setActiveItem(item.id);
                  if (item.id === 'settings') {
                    setSettingsOpen(true);
                  }
                }}
              >
                <span className="nav-icon">{item.icon}</span>
                <span className="nav-label">{item.label}</span>
                <span className="nav-arrow">→</span>
              </button>
            ))}
          </nav>
        </>
      ) : (
        <div className="settings-panel">
          <div className="settings-header">
            <h2>Settings</h2>
          </div>
          <div className="settings-content">
            <div className="settings-menu">
              <button className="settings-item">
                <span>My Profile</span>
                <span>→</span>
              </button>
              <button className="settings-item">
                <span>General Settings</span>
                <span>→</span>
              </button>
              <button className="settings-item">
                <span>Layout Settings</span>
                <span>→</span>
              </button>
              <button className="settings-item">
                <span>Device Network Tools</span>
                <span>→</span>
              </button>
              <button className="settings-item">
                <span>LAN Live View</span>
                <span>→</span>
              </button>
              <button className="settings-item">
                <span>System Permission Settings</span>
                <span>→</span>
              </button>
              <button className="settings-item">
                <span>About SmartGuard</span>
                <span>→</span>
              </button>
              <button className="settings-item logout">
                <span>Log Out</span>
              </button>
            </div>
          </div>
        </div>
      )}
    </aside>
  );
};

export default LeftSidebar;

