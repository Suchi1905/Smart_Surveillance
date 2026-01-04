import React from 'react';
import './RecentEvents.css';

const RecentEvents = ({ events }) => {
  const getSeverityColor = (severity) => {
    switch (severity) {
      case 'Severe':
        return '#f44336';
      case 'Moderate':
        return '#ff9800';
      case 'Mild':
        return '#ffeb3b';
      default:
        return '#666';
    }
  };

  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return date.toLocaleString();
  };

  return (
    <section className="recent-events">
      <div className="container">
        <h2 className="section-title">Recent Crash Events</h2>
        {events.length === 0 ? (
          <div className="no-events">
            <p>No crash events recorded yet.</p>
            <p className="hint">Events will appear here when crashes are detected.</p>
          </div>
        ) : (
          <div className="events-list">
            {events.map((event) => (
              <div key={event.id} className="event-card card">
                <div 
                  className="event-severity-indicator"
                  style={{ backgroundColor: getSeverityColor(event.severity) }}
                ></div>
                <div className="event-content">
                  <div className="event-header">
                    <h3 className="event-severity">{event.severity}</h3>
                    <span className="event-time">{formatDate(event.created_at)}</span>
                  </div>
                  <div className="event-details">
                    <div className="event-detail">
                      <span className="detail-label">Confidence:</span>
                      <span className="detail-value">{(event.confidence * 100).toFixed(1)}%</span>
                    </div>
                    <div className="event-detail">
                      <span className="detail-label">Severity Index:</span>
                      <span className="detail-value">{parseFloat(event.severity_index || 0).toFixed(2)}</span>
                    </div>
                    {event.track_id && (
                      <div className="event-detail">
                        <span className="detail-label">Track ID:</span>
                        <span className="detail-value">{event.track_id}</span>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </section>
  );
};

export default RecentEvents;


