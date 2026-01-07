import React, { useState } from 'react';
import './ActivityFeed.css';

const ActivityFeed = ({ events }) => {
  const [selectedDate, setSelectedDate] = useState(new Date());

  // Generate date options (last 7 days)
  const dateOptions = [];
  for (let i = 6; i >= 0; i--) {
    const date = new Date();
    date.setDate(date.getDate() - i);
    dateOptions.push(date);
  }

  const formatTime = (dateString) => {
    if (!dateString) return 'N/A';
    try {
      const date = new Date(dateString);
      if (isNaN(date.getTime())) return 'N/A';
      return date.toLocaleTimeString('en-US', {
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
        hour12: true
      });
    } catch (error) {
      console.error('Error formatting time:', error);
      return 'N/A';
    }
  };

  const formatDateLabel = (date) => {
    const days = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
    return `${days[date.getDay()]} ${date.getDate().toString().padStart(2, '0')}`;
  };

  const getEventTypeIcon = (severity) => {
    switch (severity) {
      case 'Severe':
        return '🚨';
      case 'Moderate':
        return '⚠️';
      case 'Mild':
        return '🔔';
      default:
        return '📹';
    }
  };

  const getEventTypeLabel = (severity) => {
    if (!severity || severity === 'Monitoring') return 'Motion Detected';
    return `${severity} Crash`;
  };

  return (
    <aside className="activity-feed">
      <div className="feed-header">
        <h3 className="feed-title">Feed</h3>
        <div className="feed-controls">
          <button className="feed-icon-btn" title="Filter">
            <span>🔽</span>
          </button>
          <button className="feed-icon-btn" title="Sort">
            <span>☰</span>
          </button>
        </div>
      </div>

      <div className="date-navigation">
        {dateOptions.map((date, index) => (
          <button
            key={index}
            className={`date-tab ${selectedDate.toDateString() === date.toDateString() ? 'active' : ''}`}
            onClick={() => setSelectedDate(date)}
          >
            {formatDateLabel(date)}
          </button>
        ))}
      </div>

      <div className="events-list">
        {!events || events.length === 0 ? (
          <div className="no-events">
            <p>No events recorded</p>
            <p className="hint">Events will appear here when detected</p>
          </div>
        ) : (
          events.filter(event => event != null).map((event, index) => {
            // Safely handle event properties
            const eventSeverity = event?.severity || 'Monitoring';
            const eventLocation = event?.location;
            const eventTime = event?.created_at || event?.timestamp || new Date().toISOString();
            
            return (
              <div key={event?.id || `event-${index}`} className="event-item">
                <div className="event-dot"></div>
                <div className="event-thumbnail">
                  <div className="thumbnail-placeholder">
                    <span className="thumbnail-icon">{getEventTypeIcon(eventSeverity)}</span>
                  </div>
                </div>
                <div className="event-details">
                  <div className="event-camera">
                    {typeof eventLocation === 'object' ? (eventLocation?.name || 'Camera') : (eventLocation || 'Camera')}
                  </div>
                  <div className="event-type">{getEventTypeLabel(eventSeverity)}</div>
                  <div className="event-time">{formatTime(eventTime)}</div>
                </div>
              </div>
            );
          })
        )}
      </div>
    </aside>
  );
};

export default ActivityFeed;
