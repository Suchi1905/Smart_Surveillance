import React, { useState, useEffect } from 'react';
import './IncidentMap.css';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const mockIncidents = [
    { x: 25, y: 30, intensity: 3, label: 'Zone A' },
    { x: 60, y: 45, intensity: 5, label: 'Zone B' },
    { x: 40, y: 70, intensity: 2, label: 'Zone C' },
    { x: 75, y: 25, intensity: 4, label: 'Zone D' },
    { x: 50, y: 55, intensity: 1, label: 'Zone E' }
];

const getIntensityClass = (intensity) => {
    if (intensity >= 5) return 'marker--danger';
    if (intensity >= 3) return 'marker--warning';
    return 'marker--cyan';
};

const IncidentMap = () => {
    const [, setTrafficData] = useState({});

    useEffect(() => {
        fetch(`${API_URL}/api/v1/analytics/traffic`)
            .then(res => {
                if (!res.ok) throw new Error('Not available');
                return res.json();
            })
            .then(data => setTrafficData(data))
            .catch(() => { });
    }, []);

    return (
        <div className="incident-map glass">
            <div className="incident-map__header">
                <span className="incident-map__icon">🗺️</span>
                <h3 className="incident-map__title">Incident Map</h3>
            </div>

            <div className="incident-map__grid">
                {/* Grid overlay */}
                <div className="incident-map__grid-lines"></div>

                {/* Incident markers */}
                {mockIncidents.map((incident, index) => (
                    <div
                        key={index}
                        className="incident-map__marker-container"
                        style={{ left: `${incident.x}%`, top: `${incident.y}%` }}
                    >
                        <div
                            className={`incident-map__pulse ${getIntensityClass(incident.intensity)}`}
                            style={{
                                width: `${8 + incident.intensity * 4}px`,
                                height: `${8 + incident.intensity * 4}px`
                            }}
                        ></div>
                        <div
                            className={`incident-map__marker ${getIntensityClass(incident.intensity)}`}
                            style={{
                                width: `${8 + incident.intensity * 4}px`,
                                height: `${8 + incident.intensity * 4}px`
                            }}
                        ></div>
                        <div className="incident-map__tooltip glass">
                            <p className="incident-map__tooltip-label">{incident.label}</p>
                            <p className="incident-map__tooltip-count">Incidents: {incident.intensity}</p>
                        </div>
                    </div>
                ))}

                {/* Active Zones overlay */}
                <div className="incident-map__zones-overlay glass">
                    <span className="incident-map__zones-icon">📍</span>
                    <div>
                        <p className="incident-map__zones-label">Active Zones</p>
                        <p className="incident-map__zones-count">{mockIncidents.length}</p>
                    </div>
                </div>

                {/* Legend */}
                <div className="incident-map__legend glass">
                    <p className="incident-map__legend-title">Severity</p>
                    <div className="incident-map__legend-items">
                        <div className="incident-map__legend-item">
                            <div className="incident-map__legend-dot marker--cyan"></div>
                            <span>Low</span>
                        </div>
                        <div className="incident-map__legend-item">
                            <div className="incident-map__legend-dot marker--warning"></div>
                            <span>Medium</span>
                        </div>
                        <div className="incident-map__legend-item">
                            <div className="incident-map__legend-dot marker--danger"></div>
                            <span>High</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default IncidentMap;
