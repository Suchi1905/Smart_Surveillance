import React from 'react';
import './Footer.css';

const Footer = () => {
  return (
    <footer className="footer">
      <div className="container">
        <div className="footer-content">
          <div className="footer-section">
            <h3>SmartGuard</h3>
            <p>Advanced AI-powered vehicle crash detection system with real-time monitoring and privacy protection.</p>
          </div>
          <div className="footer-section">
            <h4>Quick Links</h4>
            <ul>
              <li><a href="#home">Home</a></li>
              <li><a href="#features">Features</a></li>
              <li><a href="#detection">Live Detection</a></li>
              <li><a href="#statistics">Statistics</a></li>
            </ul>
          </div>
          <div className="footer-section">
            <h4>System</h4>
            <ul>
              <li>Real-Time Detection</li>
              <li>Severity Triage</li>
              <li>Privacy Anonymization</li>
              <li>Database Storage</li>
            </ul>
          </div>
        </div>
        <div className="footer-bottom">
          <p>&copy; 2025 SmartGuard Crash Detection System. All rights reserved.</p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;


