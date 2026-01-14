import React, { useState, useEffect } from 'react';
import './LiveStatusPanel.css';

/**
 * LiveStatusPanel Component
 * 
 * Displays real-time metrics about the detection system's current status.
 * Shows active tracking count, FPS, alert count, and system health.
 */
const LiveStatusPanel = ({
    isStreaming = false,
    apiUrl = 'http://localhost:8000'
}) => {
    const [stats, setStats] = useState({
        activeVehicles: 0,
        fps: 0,
        alertsToday: 0,
        detectionConfidence: 0.6,
        uptime: '00:00:00',
        lastUpdate: null
    });

    const [systemMetrics, setSystemMetrics] = useState({
        cpuUsage: 45,
        memoryUsage: 62,
        modelStatus: 'ready'
    });

    // Simulated real-time updates when streaming
    useEffect(() => {
        if (!isStreaming) {
            setStats(prev => ({ ...prev, activeVehicles: 0, fps: 0 }));
            return;
        }

        // Simulate real-time data updates
        const interval = setInterval(() => {
            setStats(prev => ({
                ...prev,
                activeVehicles: Math.floor(Math.random() * 5) + 1,
                fps: Math.floor(25 + Math.random() * 10),
                lastUpdate: new Date().toLocaleTimeString()
            }));
        }, 1000);

        return () => clearInterval(interval);
    }, [isStreaming]);

    // Fetch real stats from API periodically
    useEffect(() => {
        const fetchStats = async () => {
            try {
                const response = await fetch(`${apiUrl}/api/v1/analytics/dashboard`);
                if (response.ok) {
                    const data = await response.json();
                    if (data.total_crashes !== undefined) {
                        setStats(prev => ({
                            ...prev,
                            alertsToday: data.total_crashes || 0
                        }));
                    }
                }
            } catch (error) {
                // API might not be available, use defaults
            }
        };

        fetchStats();
        const interval = setInterval(fetchStats, 30000);
        return () => clearInterval(interval);
    }, [apiUrl]);

    const statusItems = [
        {
            id: 'tracking',
            icon: '🚗',
            label: 'Active Tracking',
            value: isStreaming ? stats.activeVehicles : '-',
            unit: 'vehicles',
            status: isStreaming ? 'active' : 'idle'
        },
        {
            id: 'fps',
            icon: '⚡',
            label: 'Processing',
            value: isStreaming ? stats.fps : '-',
            unit: 'FPS',
            status: isStreaming && stats.fps >= 25 ? 'good' : isStreaming ? 'warning' : 'idle'
        },
        {
            id: 'alerts',
            icon: '🚨',
            label: 'Alerts Today',
            value: stats.alertsToday,
            unit: 'incidents',
            status: stats.alertsToday > 0 ? 'alert' : 'good'
        },
        {
            id: 'model',
            icon: '🤖',
            label: 'AI Model',
            value: systemMetrics.modelStatus === 'ready' ? 'Ready' : 'Loading',
            unit: '',
            status: systemMetrics.modelStatus === 'ready' ? 'good' : 'loading'
        }
    ];

    return (
        <div className="live-status-panel">
            <div className="live-status-panel__header">
                <div className="live-status-panel__indicator">
                    <span className={`status-dot ${isStreaming ? 'status-dot--active' : 'status-dot--idle'}`}></span>
                    <span className="live-status-panel__title">
                        {isStreaming ? 'Live Detection Active' : 'Detection Idle'}
                    </span>
                </div>
                {stats.lastUpdate && isStreaming && (
                    <span className="live-status-panel__timestamp">
                        Last update: {stats.lastUpdate}
                    </span>
                )}
            </div>

            <div className="live-status-panel__grid">
                {statusItems.map((item) => (
                    <div
                        key={item.id}
                        className={`status-item status-item--${item.status}`}
                    >
                        <span className="status-item__icon">{item.icon}</span>
                        <div className="status-item__content">
                            <span className="status-item__value">
                                {item.value}
                                {item.unit && <span className="status-item__unit">{item.unit}</span>}
                            </span>
                            <span className="status-item__label">{item.label}</span>
                        </div>
                    </div>
                ))}
            </div>

            {isStreaming && (
                <div className="live-status-panel__progress">
                    <div className="progress-bar">
                        <div className="progress-bar__label">
                            <span>Detection Confidence</span>
                            <span>{(stats.detectionConfidence * 100).toFixed(0)}%</span>
                        </div>
                        <div className="progress-bar__track">
                            <div
                                className="progress-bar__fill"
                                style={{ width: `${stats.detectionConfidence * 100}%` }}
                            ></div>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

export default LiveStatusPanel;
