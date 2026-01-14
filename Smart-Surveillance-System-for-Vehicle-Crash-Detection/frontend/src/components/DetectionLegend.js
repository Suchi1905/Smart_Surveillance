import React, { useState } from 'react';
import './DetectionLegend.css';

/**
 * DetectionLegend Component
 * 
 * An overlay legend that explains what the colored boxes and indicators
 * in the video feed mean. Helps users understand the AI's visual output.
 */
const DetectionLegend = ({ isVisible = true }) => {
    const [isMinimized, setIsMinimized] = useState(false);

    const severityLevels = [
        { level: 'Severe', color: '#ef4444', desc: 'Critical - Emergency response needed' },
        { level: 'Moderate', color: '#f59e0b', desc: 'Medical attention required' },
        { level: 'Mild', color: '#eab308', desc: 'Minor incident detected' },
        { level: 'Tracking', color: '#22c55e', desc: 'Normal vehicle monitoring' },
    ];

    const speedIndicators = [
        { speed: '0-40 km/h', color: '#22c55e', label: 'Safe' },
        { speed: '40-80 km/h', color: '#eab308', label: 'Moderate' },
        { speed: '80+ km/h', color: '#ef4444', label: 'High' },
    ];

    const behaviorAlerts = [
        { icon: '↩️', label: 'Swerving' },
        { icon: '⛔', label: 'Wrong Way' },
        { icon: '🛑', label: 'Sudden Brake' },
        { icon: '💨', label: 'Speeding' },
    ];

    if (!isVisible) return null;

    return (
        <div className={`detection-legend ${isMinimized ? 'detection-legend--minimized' : ''}`}>
            <header
                className="detection-legend__header"
                onClick={() => setIsMinimized(!isMinimized)}
            >
                <span className="detection-legend__title">
                    {isMinimized ? '📊' : '📊 Detection Legend'}
                </span>
                <button className="detection-legend__toggle" aria-label="Toggle legend">
                    {isMinimized ? '◀' : '▶'}
                </button>
            </header>

            {!isMinimized && (
                <div className="detection-legend__content">
                    {/* Severity Section */}
                    <section className="legend-section">
                        <h4 className="legend-section__title">Crash Severity</h4>
                        <div className="legend-items">
                            {severityLevels.map((item) => (
                                <div key={item.level} className="legend-item">
                                    <span
                                        className="legend-item__color"
                                        style={{ backgroundColor: item.color }}
                                    ></span>
                                    <span className="legend-item__label">{item.level}</span>
                                </div>
                            ))}
                        </div>
                    </section>

                    {/* Speed Section */}
                    <section className="legend-section">
                        <h4 className="legend-section__title">Speed Indicators</h4>
                        <div className="legend-items legend-items--speed">
                            {speedIndicators.map((item) => (
                                <div key={item.label} className="legend-item legend-item--speed">
                                    <span
                                        className="legend-item__bar"
                                        style={{ backgroundColor: item.color }}
                                    ></span>
                                    <span className="legend-item__speed-label">{item.label}</span>
                                </div>
                            ))}
                        </div>
                    </section>

                    {/* Behavior Alerts Section */}
                    <section className="legend-section">
                        <h4 className="legend-section__title">Behavior Alerts</h4>
                        <div className="legend-items legend-items--behavior">
                            {behaviorAlerts.map((item) => (
                                <div key={item.label} className="legend-item legend-item--behavior">
                                    <span className="legend-item__icon">{item.icon}</span>
                                    <span className="legend-item__behavior-label">{item.label}</span>
                                </div>
                            ))}
                        </div>
                    </section>

                    {/* Visual Elements */}
                    <section className="legend-section">
                        <h4 className="legend-section__title">Visual Elements</h4>
                        <div className="legend-items legend-items--visual">
                            <div className="legend-item legend-item--visual">
                                <span className="legend-visual legend-visual--box"></span>
                                <span>Detection Box</span>
                            </div>
                            <div className="legend-item legend-item--visual">
                                <span className="legend-visual legend-visual--trail"></span>
                                <span>Trajectory Trail</span>
                            </div>
                            <div className="legend-item legend-item--visual">
                                <span className="legend-visual legend-visual--line"></span>
                                <span>Collision Risk</span>
                            </div>
                        </div>
                    </section>
                </div>
            )}
        </div>
    );
};

export default DetectionLegend;
