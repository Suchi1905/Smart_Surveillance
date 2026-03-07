import React, { useState, useEffect } from 'react';
import {
    BarChart, Bar, LineChart, Line, PieChart, Pie, Cell,
    XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend
} from 'recharts';
import './AnalyticsWidget.css';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
        return (
            <div className="chart-tooltip glass">
                <p className="chart-tooltip__label">{payload[0].name || payload[0].payload?.range || payload[0].payload?.hour}</p>
                <p className="chart-tooltip__value">{payload[0].value}</p>
            </div>
        );
    }
    return null;
};

const AnalyticsWidget = () => {
    const [activeTab, setActiveTab] = useState('speed');
    const [dashboardData, setDashboardData] = useState(null);
    const [speedData, setSpeedData] = useState(null);
    const [behaviorData, setBehaviorData] = useState(null);

    useEffect(() => {
        // Fetch dashboard summary
        fetch(`${API_URL}/api/v1/analytics/dashboard`)
            .then(res => res.ok ? res.json() : Promise.reject())
            .then(data => setDashboardData(data))
            .catch(() => { });

        // Fetch speed analytics
        fetch(`${API_URL}/api/v1/analytics/speed?hours=24`)
            .then(res => res.ok ? res.json() : Promise.reject())
            .then(data => setSpeedData(data))
            .catch(() => { });

        // Fetch behavior analytics
        fetch(`${API_URL}/api/v1/analytics/behavior`)
            .then(res => res.ok ? res.json() : Promise.reject())
            .then(data => setBehaviorData(data))
            .catch(() => { });
    }, []);

    // Mock data fallbacks
    const mockSpeedDistribution = [
        { range: '0-20', count: 45 },
        { range: '20-40', count: 120 },
        { range: '40-60', count: 85 },
        { range: '60-80', count: 40 },
        { range: '80+', count: 15 }
    ];

    const mockIncidentTrend = [
        { hour: '00:00', incidents: 2 },
        { hour: '04:00', incidents: 1 },
        { hour: '08:00', incidents: 5 },
        { hour: '12:00', incidents: 8 },
        { hour: '16:00', incidents: 6 },
        { hour: '20:00', incidents: 4 }
    ];

    const mockBehaviorTypes = [
        { name: 'Normal', value: 450, color: '#10b981' },
        { name: 'Aggressive', value: 25, color: '#f59e0b' },
        { name: 'Erratic', value: 15, color: '#ef4444' }
    ];

    const tabs = [
        { id: 'speed', label: '📈 Speed', icon: '📈' },
        { id: 'incidents', label: '⚠ Incidents', icon: '⚠' },
        { id: 'behavior', label: '🔄 Behavior', icon: '🔄' }
    ];

    const summary = dashboardData?.summary || {};

    return (
        <div className="analytics-widget glass">
            <div className="analytics-widget__header">
                <div className="analytics-widget__title-row">
                    <span className="analytics-widget__icon">📈</span>
                    <h3 className="analytics-widget__title">Analytics</h3>
                </div>
            </div>

            {/* Summary stats row */}
            <div className="analytics-widget__summary">
                <div className="analytics-summary-stat">
                    <span className="analytics-summary-stat__value">{summary.total_vehicles_tracked || 0}</span>
                    <span className="analytics-summary-stat__label">Vehicles</span>
                </div>
                <div className="analytics-summary-stat">
                    <span className="analytics-summary-stat__value">{summary.total_behavior_alerts || 0}</span>
                    <span className="analytics-summary-stat__label">Alerts</span>
                </div>
                <div className="analytics-summary-stat">
                    <span className="analytics-summary-stat__value">{summary.total_incidents || 0}</span>
                    <span className="analytics-summary-stat__label">Incidents</span>
                </div>
                <div className="analytics-summary-stat">
                    <span className="analytics-summary-stat__value">{summary.active_cameras || 0}</span>
                    <span className="analytics-summary-stat__label">Cameras</span>
                </div>
            </div>

            {/* Tabs */}
            <div className="analytics-widget__tabs">
                {tabs.map(tab => (
                    <button
                        key={tab.id}
                        className={`analytics-tab ${activeTab === tab.id ? 'analytics-tab--active' : ''}`}
                        onClick={() => setActiveTab(tab.id)}
                    >
                        {tab.label}
                    </button>
                ))}
            </div>

            {/* Tab Content */}
            <div className="analytics-widget__content">
                {activeTab === 'speed' && (
                    <div className="analytics-chart-content">
                        <div className="analytics-speed-stats">
                            <div className="analytics-speed-stat">
                                <span className="analytics-speed-stat__label">Avg Speed</span>
                                <span className="analytics-speed-stat__value">
                                    {speedData?.average_speed?.toFixed(1) || '45.3'} km/h
                                </span>
                            </div>
                            <div className="analytics-speed-stat">
                                <span className="analytics-speed-stat__label">Max Speed</span>
                                <span className="analytics-speed-stat__value analytics-speed-stat__value--warning">
                                    {speedData?.max_speed?.toFixed(1) || '92.5'} km/h
                                </span>
                            </div>
                            <div className="analytics-speed-stat">
                                <span className="analytics-speed-stat__label">Speeding</span>
                                <span className="analytics-speed-stat__value analytics-speed-stat__value--danger">
                                    {speedData?.speeding_count || 15}
                                </span>
                            </div>
                        </div>

                        <div className="analytics-chart-area">
                            <ResponsiveContainer width="100%" height="100%">
                                <BarChart data={speedData?.speed_distribution || mockSpeedDistribution}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                                    <XAxis dataKey="range" stroke="#94a3b8" tick={{ fill: '#94a3b8', fontSize: 11 }} />
                                    <YAxis stroke="#94a3b8" tick={{ fill: '#94a3b8', fontSize: 11 }} />
                                    <Tooltip content={<CustomTooltip />} />
                                    <defs>
                                        <linearGradient id="speedGradient" x1="0" y1="0" x2="0" y2="1">
                                            <stop offset="0%" stopColor="#22d3ee" stopOpacity={0.8} />
                                            <stop offset="100%" stopColor="#6366f1" stopOpacity={0.3} />
                                        </linearGradient>
                                    </defs>
                                    <Bar dataKey="count" fill="url(#speedGradient)" radius={[6, 6, 0, 0]} />
                                </BarChart>
                            </ResponsiveContainer>
                        </div>
                    </div>
                )}

                {activeTab === 'incidents' && (
                    <div className="analytics-chart-area analytics-chart-area--full">
                        <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={mockIncidentTrend}>
                                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                                <XAxis dataKey="hour" stroke="#94a3b8" tick={{ fill: '#94a3b8', fontSize: 11 }} />
                                <YAxis stroke="#94a3b8" tick={{ fill: '#94a3b8', fontSize: 11 }} />
                                <Tooltip content={<CustomTooltip />} />
                                <Line
                                    type="monotone"
                                    dataKey="incidents"
                                    stroke="#22d3ee"
                                    strokeWidth={3}
                                    dot={{ fill: '#22d3ee', r: 4 }}
                                    activeDot={{ r: 6, fill: '#22d3ee', strokeWidth: 2, stroke: '#0ea5e9' }}
                                />
                            </LineChart>
                        </ResponsiveContainer>
                    </div>
                )}

                {activeTab === 'behavior' && (
                    <div className="analytics-chart-area analytics-chart-area--full">
                        <ResponsiveContainer width="100%" height="100%">
                            <PieChart>
                                <Pie
                                    data={mockBehaviorTypes}
                                    cx="50%"
                                    cy="50%"
                                    innerRadius={55}
                                    outerRadius={90}
                                    paddingAngle={5}
                                    dataKey="value"
                                >
                                    {mockBehaviorTypes.map((entry, index) => (
                                        <Cell key={`cell-${index}`} fill={entry.color} />
                                    ))}
                                </Pie>
                                <Tooltip content={<CustomTooltip />} />
                                <Legend
                                    verticalAlign="bottom"
                                    height={36}
                                    formatter={(value) => <span style={{ color: '#94a3b8', fontSize: '12px' }}>{value}</span>}
                                />
                            </PieChart>
                        </ResponsiveContainer>
                    </div>
                )}
            </div>
        </div>
    );
};

export default AnalyticsWidget;
