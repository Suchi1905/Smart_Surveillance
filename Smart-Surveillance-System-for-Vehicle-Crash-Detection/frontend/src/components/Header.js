import React from 'react';
import './Header.css';

const Header = () => {
  return (
    <header className="header">
      <div className="container">
        <div className="header-content">
          <div className="logo">
            <div className="logo-icon">🛡️</div>
            <span className="logo-text">SmartGuard</span>
          </div>
          <nav className="nav">
            <a href="#home" className="nav-link">Home</a>
            <a href="#features" className="nav-link">Features</a>
            <a href="#detection" className="nav-link">Live Detection</a>
            <a href="#statistics" className="nav-link">Statistics</a>
            <a href="#contact" className="nav-link">Contact</a>
          </nav>
          <button className="btn btn-primary">Get Started</button>
        </div>
      </div>
    </header>
  );
};

export default Header;


