import React from 'react';
import './StatusPanel.css';

const StatusPanel = ({ status }) => {
  return (
    <div className="status-panel">
      <h2>System Status</h2>
      <div className="status-items">
        <span className={`status-badge ${status.detection ? 'active' : 'inactive'}`}>
          {status.detection ? '✓' : '✗'} Detection {status.detection ? 'Active' : 'Inactive'}
        </span>
        <span className={`status-badge ${status.triage ? 'active' : 'inactive'}`}>
          {status.triage ? '✓' : '✗'} Severity Triage {status.triage ? 'Enabled' : 'Disabled'}
        </span>
        <span className={`status-badge ${status.anonymization ? 'active' : 'inactive'}`}>
          {status.anonymization ? '✓' : '✗'} Privacy Anonymization {status.anonymization ? 'Active' : 'Inactive'}
        </span>
      </div>
    </div>
  );
};

export default StatusPanel;


