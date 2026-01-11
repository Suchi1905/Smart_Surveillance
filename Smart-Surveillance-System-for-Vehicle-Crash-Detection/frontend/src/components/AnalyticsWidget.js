import React, { useEffect, useState } from 'react';
import './AnalyticsWidget.css';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

/**
 * Analytics widget showing speed, behavior, and incident stats
 */
const AnalyticsWidget = () => {
    const [dashboardData, setDashboardData] = useState(null);
    const [speedStats, setSpeedStats] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    // Fetch analytics data
    useEffect(() => {
        const fetchAnalytics = async () => {
            try {
                const [dashboardRes, speedRes] = await Promise.all([
                    fetch(`${API_URL}/api/v1/analytics/dashboard`),
                    fetch(`${API_URL}/api/v1/analytics/speed?hours=24`)
                ]);

                if (dashboardRes.ok) {
                    const data = await dashboardRes.json();
                    setDashboardData(data);
                }

                if (speedRes.ok) {
                    const data = await speedRes.json();
                    setSpeedStats(data);
                }

                setLoading(false);
            } catch (err) {
                console.error('Failed to fetch analytics:', err);
                setError('Failed to load analytics');
                setLoading(false);
            }
        };

        fetchAnalytics();

        // Refresh every 30 seconds
        const interval = setInterval(fetchAnalytics, 30000);
        return () => clearInterval(interval);
    }, []);

    if (loading) {
        return (
            <div className="analytics-widget glass">
                <div className="analytics-widget__loading">
                    <span className="spinner"></span>
                    Loading analytics...
                </div>
            </div>
        );
    }

    if (error) {
        return (
            <div className="analytics-widget glass">
                <div className="analytics-widget__error">
                    ⚠️ {error}
                </div>
            </div>
        );
    }

    return (
        <div className="analytics-widget glass">
            <header className="analytics-widget__header">
                <h3 className="analytics-widget__title">📊 Analytics Dashboard</h3>
                <span className="analytics-widget__updated">
                    Updated: {dashboardData?.timestamp ? new Date(dashboardData.timestamp).toLocaleTimeString() : 'N/A'}
                </span>
            </header>

            <div className="analytics-widget__grid">
                {/* Summary Stats */}
                <div className="analytics-card">
                    <div className="analytics-card__icon">🚗</div>
                    <div className="analytics-card__content">
                        <span className="analytics-card__value">
                            {dashboardData?.summary?.total_vehicles_tracked || 0}
                        </span>
                        <span className="analytics-card__label">Vehicles Tracked</span>
                    </div>
                </div>

                <div className="analytics-card">
                    <div className="analytics-card__icon">⚡</div>
                    <div className="analytics-card__content">
                        <span className="analytics-card__value">
                            {dashboardData?.summary?.total_behavior_alerts || 0}
                        </span>
                        <span className="analytics-card__label">Behavior Alerts</span>
                    </div>
                </div>

                <div className="analytics-card">
                    <div className="analytics-card__icon">🚨</div>
                    <div className="analytics-card__content">
                        <span className="analytics-card__value">
                            {dashboardData?.summary?.total_incidents || 0}
                        </span>
                        <span className="analytics-card__label">Incidents</span>
                    </div>
                </div>

                <div className="analytics-card">
                    <div className="analytics-card__icon">📹</div>
                    <div className="analytics-card__content">
                        <span className="analytics-card__value">
                            {dashboardData?.summary?.active_cameras || 0}
                        </span>
                        <span className="analytics-card__label">Active Cameras</span>
                    </div>
                </div>
            </div>

            {/* Speed Statistics */}
            {speedStats && (
                <div className="analytics-widget__section">
                    <h4 className="analytics-widget__section-title">🏎️ Speed Statistics (24h)</h4>
                    <div className="speed-stats">
                        <div className="speed-stat">
                            <span className="speed-stat__label">Average</span>
                            <span className="speed-stat__value">{speedStats.average_speed} km/h</span>
                        </div>
                        <div className="speed-stat">
                            <span className="speed-stat__label">Max</span>
                            <span className="speed-stat__value speed-stat__value--max">{speedStats.max_speed} km/h</span>
                        </div>
                        <div className="speed-stat">
                            <span className="speed-stat__label">Speeding</span>
                            <span className="speed-stat__value speed-stat__value--warning">{speedStats.speeding_count}</span>
                        </div>
                    </div>

                    {/* Speed Distribution */}
                    {speedStats.speed_distribution && (
                        <div className="speed-distribution">
                            {Object.entries(speedStats.speed_distribution).map(([range, count]) => (
                                <div key={range} className="speed-bar">
                                    <span className="speed-bar__label">{range}</span>
                                    <div className="speed-bar__track">
                                        <div
                                            className="speed-bar__fill"
                                            style={{
                                                width: `${Math.min(100, (count / (speedStats.total_vehicles || 1)) * 100)}%`
                                            }}
                                        />
                                    </div>
                                    <span className="speed-bar__count">{count}</span>
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            )}

            {/* Recent Alerts */}
            {dashboardData?.recent_alerts?.length > 0 && (
                <div className="analytics-widget__section">
                    <h4 className="analytics-widget__section-title">⚡ Recent Behavior Alerts</h4>
                    <div className="recent-alerts">
                        {dashboardData.recent_alerts.slice(0, 3).map((alert, idx) => (
                            <div key={idx} className="recent-alert">
                                <span className="recent-alert__type">{alert.behavior_type || 'Alert'}</span>
                                <span className="recent-alert__severity">{alert.severity}</span>
                            </div>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
};

export default AnalyticsWidget;
