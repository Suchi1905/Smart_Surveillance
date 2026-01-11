import React, { useEffect, useState } from 'react';
import './IncidentsView.css';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

/**
 * Incidents/Crash Logs View
 * Shows history of detected crashes from the backend
 */
const IncidentsView = () => {
    const [incidents, setIncidents] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [filter, setFilter] = useState('all');

    // Fetch incidents from backend
    useEffect(() => {
        const fetchIncidents = async () => {
            try {
                setLoading(true);
                const response = await fetch(`${API_URL}/api/v1/crashes`);

                if (response.ok) {
                    const data = await response.json();
                    // Handle various response formats
                    let incidentsList = [];
                    if (Array.isArray(data)) {
                        incidentsList = data;
                    } else if (data && Array.isArray(data.crashes)) {
                        incidentsList = data.crashes;
                    } else if (data && Array.isArray(data.items)) {
                        incidentsList = data.items;
                    }
                    setIncidents(incidentsList);
                    setError(null);
                } else {
                    setError('Failed to fetch incidents');
                    setIncidents([]);
                }
            } catch (err) {
                console.error('Failed to fetch incidents:', err);
                setError('Unable to connect to server');
                setIncidents([]);
            } finally {
                setLoading(false);
            }
        };

        fetchIncidents();

        // Refresh every 30 seconds
        const interval = setInterval(fetchIncidents, 30000);
        return () => clearInterval(interval);
    }, []);

    // Get severity color
    const getSeverityColor = (severity) => {
        switch (severity?.toLowerCase()) {
            case 'severe': case 'critical': return '#ff4444';
            case 'moderate': case 'high': return '#ff8800';
            case 'mild': case 'low': return '#ffcc00';
            default: return '#44ff44';
        }
    };

    // Filter incidents - ensure incidents is an array
    const filteredIncidents = Array.isArray(incidents)
        ? incidents.filter(incident => {
            if (filter === 'all') return true;
            return incident.severity?.toLowerCase() === filter;
        })
        : [];

    if (loading) {
        return (
            <div className="incidents-view">
                <div className="incidents-view__loading">
                    <span className="spinner"></span>
                    Loading incidents...
                </div>
            </div>
        );
    }

    return (
        <div className="incidents-view">
            <header className="incidents-view__header">
                <div className="incidents-view__title-group">
                    <h1 className="incidents-view__title">⚠️ Incident Logs</h1>
                    <p className="incidents-view__subtitle">
                        Historical crash events and severity reports
                    </p>
                </div>

                <div className="incidents-view__controls">
                    <select
                        className="incidents-view__filter"
                        value={filter}
                        onChange={(e) => setFilter(e.target.value)}
                    >
                        <option value="all">All Incidents</option>
                        <option value="severe">Severe Only</option>
                        <option value="moderate">Moderate</option>
                        <option value="mild">Mild</option>
                    </select>

                    <span className="incidents-view__count">
                        {filteredIncidents.length} incidents
                    </span>
                </div>
            </header>

            {error && (
                <div className="incidents-view__error">
                    ⚠️ {error}
                </div>
            )}

            <div className="incidents-view__list">
                {filteredIncidents.length === 0 ? (
                    <div className="incidents-view__empty">
                        <span className="incidents-view__empty-icon">✅</span>
                        <h3>No Incidents Recorded</h3>
                        <p>No crash events have been detected yet. Start the detection feed to monitor for incidents.</p>
                    </div>
                ) : (
                    filteredIncidents.map((incident, idx) => (
                        <article
                            key={incident.id || idx}
                            className="incident-card glass"
                        >
                            <div
                                className="incident-card__severity-bar"
                                style={{ backgroundColor: getSeverityColor(incident.severity) }}
                            />

                            <div className="incident-card__content">
                                <header className="incident-card__header">
                                    <div className="incident-card__meta">
                                        <span className="incident-card__id">#{incident.id || idx + 1}</span>
                                        <span
                                            className="incident-card__severity"
                                            style={{ color: getSeverityColor(incident.severity) }}
                                        >
                                            {incident.severity || 'Unknown'}
                                        </span>
                                    </div>
                                    <time className="incident-card__time">
                                        {incident.timestamp
                                            ? new Date(incident.timestamp).toLocaleString()
                                            : 'Unknown time'
                                        }
                                    </time>
                                </header>

                                <div className="incident-card__details">
                                    <div className="incident-card__stat">
                                        <span className="incident-card__stat-label">Confidence</span>
                                        <span className="incident-card__stat-value">
                                            {(incident.confidence * 100 || 0).toFixed(1)}%
                                        </span>
                                    </div>

                                    <div className="incident-card__stat">
                                        <span className="incident-card__stat-label">Severity Index</span>
                                        <span className="incident-card__stat-value">
                                            {(incident.severity_index || 0).toFixed(2)}
                                        </span>
                                    </div>

                                    {incident.location && (
                                        <div className="incident-card__stat">
                                            <span className="incident-card__stat-label">Location</span>
                                            <span className="incident-card__stat-value">
                                                {incident.location}
                                            </span>
                                        </div>
                                    )}
                                </div>

                                {incident.description && (
                                    <p className="incident-card__description">
                                        {incident.description}
                                    </p>
                                )}
                            </div>
                        </article>
                    ))
                )}
            </div>
        </div>
    );
};

export default IncidentsView;
