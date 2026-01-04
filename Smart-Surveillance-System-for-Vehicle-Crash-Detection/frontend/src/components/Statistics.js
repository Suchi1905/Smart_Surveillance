import React from 'react';
import './Statistics.css';

const Statistics = ({ stats }) => {
  if (!stats) {
    return (
      <section id="statistics" className="statistics">
        <div className="container">
          <h2 className="section-title">System Statistics</h2>
          <p>Loading statistics...</p>
        </div>
      </section>
    );
  }

  return (
    <section id="statistics" className="statistics">
      <div className="container">
        <h2 className="section-title">Reduce Cost And Increase Your Total Work Efficiency</h2>
        <div className="stats-grid">
          <div className="stat-card card">
            <div className="stat-value">{stats.total_events || 0}</div>
            <div className="stat-label">Total Events</div>
          </div>
          <div className="stat-card card stat-severe">
            <div className="stat-value">{stats.severe_count || 0}</div>
            <div className="stat-label">Severe Crashes</div>
          </div>
          <div className="stat-card card stat-moderate">
            <div className="stat-value">{stats.moderate_count || 0}</div>
            <div className="stat-label">Moderate Crashes</div>
          </div>
          <div className="stat-card card stat-mild">
            <div className="stat-value">{stats.mild_count || 0}</div>
            <div className="stat-label">Mild Crashes</div>
          </div>
          <div className="stat-card card">
            <div className="stat-value">
              {stats.avg_confidence ? (parseFloat(stats.avg_confidence) * 100).toFixed(1) : 0}%
            </div>
            <div className="stat-label">Avg Confidence</div>
          </div>
          <div className="stat-card card">
            <div className="stat-value">
              {stats.avg_severity_index ? parseFloat(stats.avg_severity_index).toFixed(2) : 0}
            </div>
            <div className="stat-label">Avg Severity Index</div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Statistics;


