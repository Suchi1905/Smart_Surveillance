import React, { useEffect, useState, useRef, useCallback } from 'react';
import './AlertsPanel.css';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
const WS_URL = API_URL.replace('http', 'ws');

/**
 * Real-time alerts panel with WebSocket connection
 */
const AlertsPanel = ({ maxAlerts = 10 }) => {
    const [alerts, setAlerts] = useState([]);
    const [connected, setConnected] = useState(false);
    const [wsStatus, setWsStatus] = useState('Disconnected');
    const wsRef = useRef(null);
    const reconnectTimeoutRef = useRef(null);

    // Connect to WebSocket
    const connectWebSocket = useCallback(() => {
        if (wsRef.current?.readyState === WebSocket.OPEN) return;

        try {
            const ws = new WebSocket(`${WS_URL}/ws/alerts`);
            wsRef.current = ws;

            ws.onopen = () => {
                setConnected(true);
                setWsStatus('Connected');
                console.log('WebSocket connected to alerts');
            };

            ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);

                    if (data.type === 'alert') {
                        const newAlert = {
                            id: Date.now(),
                            ...data.data,
                            timestamp: new Date().toLocaleTimeString(),
                            sentToTelegram: false
                        };

                        setAlerts(prev => [newAlert, ...prev].slice(0, maxAlerts));
                    } else if (data.type === 'notification_sent') {
                        // Update the alert to show it was sent to Telegram
                        const { track_id } = data.data;
                        setAlerts(prev => prev.map(alert =>
                            alert.track_id === track_id
                                ? { ...alert, sentToTelegram: true }
                                : alert
                        ));

                        // Optional: Show toast here if you implement a toast system
                        console.log('Telegram notification confirmed sent');
                    }
                } catch (err) {
                    console.error('Failed to parse WebSocket message:', err);
                }
            };

            ws.onclose = () => {
                setConnected(false);
                setWsStatus('Disconnected');
                // Reconnect after 3 seconds
                reconnectTimeoutRef.current = setTimeout(connectWebSocket, 3000);
            };

            ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                setWsStatus('Error');
            };
        } catch (error) {
            console.error('Failed to create WebSocket:', error);
            setWsStatus('Failed');
        }
    }, [maxAlerts]);

    // Initial connection
    useEffect(() => {
        connectWebSocket();

        return () => {
            if (reconnectTimeoutRef.current) {
                clearTimeout(reconnectTimeoutRef.current);
            }
            if (wsRef.current) {
                wsRef.current.close();
            }
        };
    }, [connectWebSocket]);

    // Get severity icon
    const getSeverityIcon = (severity) => {
        switch (severity?.toLowerCase()) {
            case 'critical': return '🚨';
            case 'severe': return '⚠️';
            case 'high': return '⚡';
            case 'warning': return '⚡';
            case 'moderate': return '📢';
            default: return 'ℹ️';
        }
    };

    // Get severity class
    const getSeverityClass = (severity) => {
        switch (severity?.toLowerCase()) {
            case 'critical': return 'alert--critical';
            case 'severe': return 'alert--severe';
            case 'high': return 'alert--high';
            case 'warning': return 'alert--warning';
            default: return 'alert--info';
        }
    };

    // Dismiss alert
    const dismissAlert = (id) => {
        setAlerts(prev => prev.filter(a => a.id !== id));
    };

    // Clear all alerts
    const clearAllAlerts = () => {
        setAlerts([]);
    };

    return (
        <div className="alerts-panel glass">
            <header className="alerts-panel__header">
                <h3 className="alerts-panel__title">
                    🔔 Live Alerts
                    <span className={`alerts-panel__status ${connected ? 'connected' : 'disconnected'}`}>
                        {wsStatus}
                    </span>
                </h3>
                {alerts.length > 0 && (
                    <button className="btn btn--ghost btn--sm" onClick={clearAllAlerts}>
                        Clear All
                    </button>
                )}
            </header>

            <div className="alerts-panel__list">
                {alerts.length === 0 ? (
                    <div className="alerts-panel__empty">
                        <span className="alerts-panel__empty-icon">✅</span>
                        <span>No active alerts</span>
                    </div>
                ) : (
                    alerts.map((alert) => (
                        <article
                            key={alert.id}
                            className={`alert-item ${getSeverityClass(alert.severity)}`}
                        >
                            <div className="alert-item__icon">
                                {getSeverityIcon(alert.severity)}
                            </div>
                            <div className="alert-item__content">
                                <div className="alert-item__header">
                                    <span className="alert-item__type">
                                        {alert.behavior_type || alert.type || 'Alert'}
                                    </span>
                                    <span className="alert-item__time">{alert.timestamp}</span>
                                </div>
                                <p className="alert-item__message">
                                    {alert.description || alert.message || 'Detection triggered'}
                                </p>
                                {alert.track_id && (
                                    <div className="alert-item__meta">
                                        <span className="alert-item__track">#{alert.track_id}</span>
                                        {alert.sentToTelegram && (
                                            <span className="alert-item__telegram" title="Notification sent to Telegram">
                                                ✈️ Sent
                                            </span>
                                        )}
                                    </div>
                                )}
                            </div>
                            <button
                                className="alert-item__dismiss"
                                onClick={() => dismissAlert(alert.id)}
                                aria-label="Dismiss"
                            >
                                ×
                            </button>
                        </article>
                    ))
                )}
            </div>

            <footer className="alerts-panel__footer">
                <span className="alerts-panel__count">
                    {alerts.length} alert{alerts.length !== 1 ? 's' : ''}
                </span>
            </footer>
        </div>
    );
};

export default AlertsPanel;
