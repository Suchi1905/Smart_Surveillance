import React from 'react';
import './Footer.css';

const Footer = ({ fps, latency, networkStatus }) => {
  const getNetworkStatusColor = (status) => {
    switch (status) {
      case 'Connected':
        return 'var(--status-healthy)';
      case 'Warning':
        return 'var(--status-warning)';
      case 'Disconnected':
        return 'var(--status-danger)';
      default:
        return 'var(--text-muted)';
    }
  };

  return (
    <footer className="dashboard-footer">
      <div className="footer-content">
        <div className="footer-item">
          <span className="footer-label">FPS</span>
          <span className="footer-value">{fps.toFixed(0)}</span>
        </div>
        
        <div className="footer-divider"></div>
        
        <div className="footer-item">
          <span className="footer-label">Inference Latency</span>
          <span className="footer-value">{latency}ms</span>
        </div>
        
        <div className="footer-divider"></div>
        
        <div className="footer-item">
          <span className="footer-label">Network Status</span>
          <span 
            className="footer-value network-status"
            style={{ color: getNetworkStatusColor(networkStatus) }}
          >
            {networkStatus}
          </span>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
