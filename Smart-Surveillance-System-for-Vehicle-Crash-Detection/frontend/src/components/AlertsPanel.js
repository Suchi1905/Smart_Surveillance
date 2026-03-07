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
    const [soundEnabled, setSoundEnabled] = useState(true);
    const [toasts, setToasts] = useState([]);
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

                        // Toast notification for severe/critical alerts
                        const sev = (data.data.severity || '').toLowerCase();
                        if (sev === 'severe' || sev === 'critical') {
                            const toastId = Date.now() + Math.random();
                            setToasts(prev => [...prev, {
                                id: toastId,
                                severity: sev,
                                message: data.data.description || data.data.message || 'Alert triggered'
                            }]);
                            setTimeout(() => {
                                setToasts(prev => prev.filter(t => t.id !== toastId));
                            }, 5000);

                            // Sound notification
                            if (soundEnabled) {
                                try {
                                    const audio = new Audio('data:audio/wav;base64,UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbsGczIj2a0teleShJj8PV17dHMECKwtfTuk9FQH+7z9DDUVA=');
                                    audio.volume = 0.3;
                                    audio.play().catch(() => { });
                                } catch (e) { }
                            }
                        }
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
            {/* Toast Notifications */}
            {toasts.length > 0 && (
                <div className="toast-container">
                    {toasts.map(toast => (
                        <div key={toast.id} className={`toast toast--${toast.severity}`}>
                            <div className="toast__title">🚨 {toast.severity === 'critical' ? 'Critical' : 'Severe'} Alert</div>
                            <div className="toast__desc">{toast.message}</div>
                        </div>
                    ))}
                </div>
            )}

            <header className="alerts-panel__header">
                <h3 className="alerts-panel__title">
                    🔔 Live Alerts
                    <span className={`alerts-panel__status ${connected ? 'connected' : 'disconnected'}`}>
                        {wsStatus}
                    </span>
                </h3>
                <div className="alerts-panel__controls">
                    <button
                        className={`btn btn--ghost btn--sm alerts-panel__sound-btn ${soundEnabled ? '' : 'alerts-panel__sound-btn--muted'}`}
                        onClick={() => setSoundEnabled(!soundEnabled)}
                        title={soundEnabled ? 'Mute alerts' : 'Enable sound'}
                    >
                        {soundEnabled ? '🔊' : '🔇'}
                    </button>
                    {alerts.length > 0 && (
                        <button className="btn btn--ghost btn--sm" onClick={clearAllAlerts}>
                            Clear All
                        </button>
                    )}
                </div>
            </header>

            <div className="alerts-panel__list">
                {alerts.length === 0 ? (
                    <div className="alerts-panel__empty">
                        <span className="alerts-panel__empty-icon">✅</span>
                        <span>No active alerts</span>
                    </div>
                ) : (
                    alerts.map((alert, index) => (
                        <article
                            key={alert.id}
                            className={`alert-item ${getSeverityClass(alert.severity)} alert-item--animate`}
                            style={{ animationDelay: `${index * 0.05}s` }}
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
                                        <span className={`alert-item__telegram ${alert.sentToTelegram ? 'alert-item__telegram--sent' : 'alert-item__telegram--pending'}`}
                                            title={alert.sentToTelegram ? 'Sent to Telegram' : 'Not sent'}>
                                            {alert.sentToTelegram ? '✅ Sent' : '⏳ Pending'}
                                        </span>
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
