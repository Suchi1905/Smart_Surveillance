import React from 'react';
import './Features.css';

const Features = ({ systemStatus }) => {
  const features = [
    {
      icon: '🎯',
      title: 'Real-Time Detection',
      description: 'Advanced AI-powered crash detection with 30 FPS processing',
      status: systemStatus.ml_service?.available
    },
    {
      icon: '📊',
      title: 'Severity Triage',
      description: 'Automatic classification: Severe, Moderate, or Mild crashes',
      status: systemStatus.ml_service?.triage
    },
    {
      icon: '🔒',
      title: 'Privacy Protection',
      description: 'GDPR-compliant anonymization of faces and license plates',
      status: systemStatus.ml_service?.anonymization
    },
    {
      icon: '💾',
      title: 'Data Storage',
      description: 'PostgreSQL database for crash event history and analytics',
      status: systemStatus.database?.connected
    }
  ];

  return (
    <section id="features" className="features">
      <div className="container">
        <h2 className="section-title">Protect Your Roads With Our Unique Solutions</h2>
        <div className="features-grid">
          {features.map((feature, index) => (
            <div key={index} className="feature-card card">
              <div className="feature-icon">{feature.icon}</div>
              <h3 className="feature-title">{feature.title}</h3>
              <p className="feature-description">{feature.description}</p>
              <div className="feature-status">
                {feature.status ? (
                  <span className="status-badge active">● Active</span>
                ) : (
                  <span className="status-badge inactive">○ Inactive</span>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default Features;


