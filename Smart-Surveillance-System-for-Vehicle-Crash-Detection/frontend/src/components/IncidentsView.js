import React, { useState, useEffect, useCallback } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import './IncidentsView.css';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
        return (
            <div className="chart-tooltip glass">
                <p className="chart-tooltip__label">{payload[0].payload.time}</p>
                <p className="chart-tooltip__value">{payload[0].value.toFixed(1)}</p>
            </div>
        );
    }
    return null;
};

const IncidentsView = () => {
    const [incidents, setIncidents] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [severityFilter, setSeverityFilter] = useState('all');
    const [selectedIncident, setSelectedIncident] = useState(null);

    const fetchIncidents = useCallback(async () => {
        try {
            const response = await fetch(`${API_URL}/api/v1/crashes`);
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            const data = await response.json();

            let crashList = [];
            if (Array.isArray(data)) {
                crashList = data;
            } else if (data.crashes && Array.isArray(data.crashes)) {
                crashList = data.crashes;
            } else if (data.data && Array.isArray(data.data)) {
                crashList = data.data;
            }

            setIncidents(crashList);
            setError(null);
        } catch (err) {
            console.error('Failed to fetch incidents:', err);
            setError(err.message);
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => {
        fetchIncidents();
        const interval = setInterval(fetchIncidents, 30000);
        return () => clearInterval(interval);
    }, [fetchIncidents]);

    const filteredIncidents = severityFilter === 'all'
        ? incidents
        : incidents.filter(i => i.severity === severityFilter);

    const getSeverityClass = (severity) => {
        switch (severity) {
            case 'Severe': return 'incident--severe';
            case 'Moderate': return 'incident--moderate';
            case 'Mild': return 'incident--mild';
            default: return 'incident--unknown';
        }
    };

    const getSeverityEmoji = (severity) => {
        switch (severity) {
            case 'Severe': return '🔴';
            case 'Moderate': return '🟠';
            case 'Mild': return '🔵';
            default: return '⚪';
        }
    };

    // CSV Export
    const handleExport = () => {
        const csv = [
            ['ID', 'Severity', 'Confidence', 'Severity Index', 'Timestamp', 'Location', 'Description'],
            ...filteredIncidents.map(i => [
                i.id,
                i.severity,
                i.confidence,
                i.severity_index,
                i.timestamp,
                i.location || '',
                i.description || ''
            ])
        ].map(row => row.join(',')).join('\n');

        const blob = new Blob([csv], { type: 'text/csv' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `incidents-${new Date().toISOString().split('T')[0]}.csv`;
        a.click();
        window.URL.revokeObjectURL(url);
    };

    // Delete incident
    const handleDelete = async (id) => {
        try {
            await fetch(`${API_URL}/api/v1/crashes/${id}`, { method: 'DELETE' });
            setIncidents(prev => prev.filter(i => i.id !== id));
            setSelectedIncident(null);
        } catch (err) {
            console.error('Failed to delete incident:', err);
        }
    };

    // Mock severity timeline for header chart
    const mockSeverityTimeline = [
        { time: '0s', severity: 2.1 },
        { time: '2s', severity: 3.5 },
        { time: '4s', severity: 5.2 },
        { time: '6s', severity: 7.8 },
        { time: '8s', severity: 8.9 }
    ];

    if (loading) {
        return (
            <div className="incidents-view">
                <div className="incidents-view__loading glass">
                    <div className="incidents-view__spinner"></div>
                    <p>Loading incident data...</p>
                </div>
            </div>
        );
    }

    return (
        <div className="incidents-view">
            {/* Page Header */}
            <div className="incidents-view__page-header">
                <div>
                    <h2 className="incidents-view__page-title">Incident Log</h2>
                    <p className="incidents-view__page-subtitle">Historical crash detection records</p>
                </div>
            </div>

            {/* Severity Timeline Chart */}
            <div className="incidents-view__timeline-chart glass">
                <div className="incidents-view__timeline-header">
                    <span>📈</span>
                    <h3>Severity Timeline</h3>
                </div>
                <div className="incidents-view__timeline-area">
                    <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={mockSeverityTimeline}>
                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                            <XAxis dataKey="time" stroke="#94a3b8" tick={{ fill: '#94a3b8', fontSize: 12 }} />
                            <YAxis
                                stroke="#94a3b8"
                                tick={{ fill: '#94a3b8', fontSize: 12 }}
                                label={{ value: 'Severity', angle: -90, position: 'insideLeft', fill: '#94a3b8', fontSize: 11 }}
                            />
                            <Tooltip content={<CustomTooltip />} />
                            <Line
                                type="monotone"
                                dataKey="severity"
                                stroke="#ef4444"
                                strokeWidth={3}
                                dot={{ fill: '#ef4444', r: 4 }}
                            />
                        </LineChart>
                    </ResponsiveContainer>
                </div>
            </div>

            {/* Filter Bar */}
            <div className="incidents-view__filter-bar glass">
                <div className="incidents-view__filter-left">
                    <span className="incidents-view__filter-icon">🔍</span>
                    <span className="incidents-view__filter-label">Filter:</span>
                    <select
                        value={severityFilter}
                        onChange={(e) => setSeverityFilter(e.target.value)}
                        className="incidents-view__filter-select"
                    >
                        <option value="all">All Severities</option>
                        <option value="Severe">🔴 Severe</option>
                        <option value="Moderate">🟠 Moderate</option>
                        <option value="Mild">🔵 Mild</option>
                    </select>
                    <div className="incidents-view__count-badge">
                        {filteredIncidents.length} incidents
                    </div>
                </div>
                <button className="btn btn--outline incidents-view__export-btn" onClick={handleExport}>
                    📥 Export CSV
                </button>
            </div>

            {/* Error state */}
            {error && (
                <div className="incidents-view__error glass">
                    <span>⚠️</span>
                    <span>Failed to load incidents: {error}</span>
                    <button className="btn btn--primary" onClick={fetchIncidents}>Retry</button>
                </div>
            )}

            {/* Incidents List */}
            <div className="incidents-view__list">
                {filteredIncidents.length === 0 ? (
                    <div className="incidents-view__empty glass">
                        <div className="incidents-view__empty-icon">📋</div>
                        <p className="incidents-view__empty-text">No incidents found</p>
                        <p className="incidents-view__empty-sub">Adjust filters or wait for new detections</p>
                    </div>
                ) : (
                    filteredIncidents.map((incident, index) => (
                        <div
                            key={incident.id || index}
                            className={`incident-card glass ${getSeverityClass(incident.severity)} incident-card--animate`}
                            style={{ animationDelay: `${index * 0.05}s` }}
                            onClick={() => setSelectedIncident(incident)}
                        >
                            <div className="incident-card__severity-bar"></div>
                            <div className="incident-card__body">
                                <div className="incident-card__main">
                                    <div className="incident-card__header">
                                        <span className={`incident-card__badge ${getSeverityClass(incident.severity)}`}>
                                            {getSeverityEmoji(incident.severity)} {incident.severity}
                                        </span>
                                        <span className="incident-card__id">ID: {incident.id}</span>
                                    </div>
                                    {incident.description && (
                                        <p className="incident-card__desc">{incident.description}</p>
                                    )}
                                    {incident.location && (
                                        <div className="incident-card__location">
                                            📍 {incident.location}
                                        </div>
                                    )}
                                </div>

                                <div className="incident-card__stats">
                                    <div className="incident-card__stat">
                                        <span className="incident-card__stat-label">Confidence</span>
                                        <span className="incident-card__stat-value incident-card__stat-value--cyan">
                                            {typeof incident.confidence === 'number'
                                                ? `${(incident.confidence * 100).toFixed(1)}%`
                                                : incident.confidence || 'N/A'}
                                        </span>
                                    </div>
                                    <div className="incident-card__stat">
                                        <span className="incident-card__stat-label">Severity Index</span>
                                        <span className="incident-card__stat-value incident-card__stat-value--warning">
                                            {typeof incident.severity_index === 'number'
                                                ? incident.severity_index.toFixed(1)
                                                : incident.severity_index || 'N/A'}
                                        </span>
                                    </div>
                                    <div className="incident-card__stat incident-card__stat--time">
                                        <span className="incident-card__stat-label">📅 Time</span>
                                        <span className="incident-card__stat-value">
                                            {incident.timestamp ? new Date(incident.timestamp).toLocaleString() : 'N/A'}
                                        </span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    ))
                )}
            </div>

            {/* Detail Modal */}
            {selectedIncident && (
                <div className="modal-overlay" onClick={() => setSelectedIncident(null)}>
                    <div className="modal-content" onClick={(e) => e.stopPropagation()}>
                        <div className="modal-header">
                            <div className="modal-header__title">
                                <span>🚨</span>
                                <span>Incident Details</span>
                                <span className={`incident-card__badge ${getSeverityClass(selectedIncident.severity)}`}>
                                    {selectedIncident.severity}
                                </span>
                            </div>
                            <button className="modal-close" onClick={() => setSelectedIncident(null)}>×</button>
                        </div>

                        <div className="modal-grid">
                            <div className="modal-field">
                                <div className="modal-field__label">Incident ID</div>
                                <div className="modal-field__value">{selectedIncident.id}</div>
                            </div>
                            <div className="modal-field">
                                <div className="modal-field__label">Timestamp</div>
                                <div className="modal-field__value">
                                    {selectedIncident.timestamp ? new Date(selectedIncident.timestamp).toLocaleString() : 'N/A'}
                                </div>
                            </div>
                            <div className="modal-field">
                                <div className="modal-field__label">Confidence</div>
                                <div className="modal-field__value modal-field__value--cyan">
                                    {typeof selectedIncident.confidence === 'number'
                                        ? `${(selectedIncident.confidence * 100).toFixed(1)}%`
                                        : 'N/A'}
                                </div>
                            </div>
                            <div className="modal-field">
                                <div className="modal-field__label">Severity Index</div>
                                <div className="modal-field__value modal-field__value--warning">
                                    {typeof selectedIncident.severity_index === 'number'
                                        ? selectedIncident.severity_index.toFixed(1)
                                        : 'N/A'}
                                </div>
                            </div>
                        </div>

                        {selectedIncident.location && (
                            <div className="modal-field" style={{ marginBottom: '12px' }}>
                                <div className="modal-field__label">Location</div>
                                <div className="modal-field__value">📍 {selectedIncident.location}</div>
                            </div>
                        )}

                        {selectedIncident.description && (
                            <div className="modal-field" style={{ marginBottom: '12px' }}>
                                <div className="modal-field__label">Description</div>
                                <div className="modal-field__value" style={{ fontSize: '0.85rem', fontFamily: 'Inter' }}>
                                    {selectedIncident.description}
                                </div>
                            </div>
                        )}

                        <div className="modal-actions">
                            <button
                                className="btn btn--danger"
                                onClick={() => handleDelete(selectedIncident.id)}
                            >
                                🗑 Delete Incident
                            </button>
                            <button
                                className="btn btn--outline"
                                onClick={() => setSelectedIncident(null)}
                            >
                                Close
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

export default IncidentsView;
