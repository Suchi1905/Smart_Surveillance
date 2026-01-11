import React, { useEffect, useState } from 'react';
import './SettingsView.css';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

/**
 * Settings View
 * System configuration and preferences
 */
const SettingsView = () => {
    const [settings, setSettings] = useState({
        confidence_threshold: 0.6,
        anonymization_enabled: true,
        alert_cooldown: 30,
        telegram_enabled: false,
        telegram_token: '',
        telegram_chat_id: ''
    });
    const [systemInfo, setSystemInfo] = useState(null);
    const [loading, setLoading] = useState(true);
    const [saving, setSaving] = useState(false);
    const [message, setMessage] = useState(null);

    // Fetch current settings
    useEffect(() => {
        const fetchSettings = async () => {
            try {
                const [configRes, statusRes] = await Promise.all([
                    fetch(`${API_URL}/api/v1/system/config`),
                    fetch(`${API_URL}/api/system/status`)
                ]);

                if (configRes.ok) {
                    const config = await configRes.json();
                    setSettings(prev => ({ ...prev, ...config }));
                }

                if (statusRes.ok) {
                    const status = await statusRes.json();
                    setSystemInfo(status);
                }
            } catch (err) {
                console.error('Failed to fetch settings:', err);
            } finally {
                setLoading(false);
            }
        };

        fetchSettings();
    }, []);

    // Handle input changes
    const handleChange = (e) => {
        const { name, value, type, checked } = e.target;
        setSettings(prev => ({
            ...prev,
            [name]: type === 'checkbox' ? checked :
                type === 'number' ? parseFloat(value) : value
        }));
    };

    // Save settings
    const handleSave = async () => {
        setSaving(true);
        setMessage(null);

        try {
            const response = await fetch(`${API_URL}/api/v1/system/config`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(settings)
            });

            if (response.ok) {
                setMessage({ type: 'success', text: 'Settings saved successfully!' });
            } else {
                setMessage({ type: 'error', text: 'Failed to save settings' });
            }
        } catch (err) {
            setMessage({ type: 'error', text: 'Unable to connect to server' });
        } finally {
            setSaving(false);
            setTimeout(() => setMessage(null), 3000);
        }
    };

    if (loading) {
        return (
            <div className="settings-view">
                <div className="settings-view__loading">
                    <span className="spinner"></span>
                    Loading settings...
                </div>
            </div>
        );
    }

    return (
        <div className="settings-view">
            <header className="settings-view__header">
                <h1 className="settings-view__title">⚙️ System Settings</h1>
                <p className="settings-view__subtitle">
                    Configure detection parameters and alert preferences
                </p>
            </header>

            {message && (
                <div className={`settings-view__message settings-view__message--${message.type}`}>
                    {message.text}
                </div>
            )}

            <div className="settings-view__grid">
                {/* Detection Settings */}
                <section className="settings-card glass">
                    <h2 className="settings-card__title">🎯 Detection Settings</h2>

                    <div className="settings-field">
                        <label className="settings-field__label">
                            Confidence Threshold
                        </label>
                        <div className="settings-field__input-row">
                            <input
                                type="range"
                                name="confidence_threshold"
                                min="0.1"
                                max="1.0"
                                step="0.05"
                                value={settings.confidence_threshold}
                                onChange={handleChange}
                            />
                            <span className="settings-field__value">
                                {settings.confidence_threshold.toFixed(2)}
                            </span>
                        </div>
                        <p className="settings-field__help">
                            Minimum confidence score for detection (higher = fewer false positives)
                        </p>
                    </div>

                    <div className="settings-field">
                        <label className="settings-field__label">
                            Alert Cooldown (seconds)
                        </label>
                        <input
                            type="number"
                            name="alert_cooldown"
                            min="10"
                            max="300"
                            value={settings.alert_cooldown}
                            onChange={handleChange}
                            className="settings-field__input"
                        />
                        <p className="settings-field__help">
                            Time between consecutive alerts to prevent spam
                        </p>
                    </div>
                </section>

                {/* Privacy Settings */}
                <section className="settings-card glass">
                    <h2 className="settings-card__title">🔒 Privacy & Anonymization</h2>

                    <div className="settings-field">
                        <label className="settings-field__checkbox">
                            <input
                                type="checkbox"
                                name="anonymization_enabled"
                                checked={settings.anonymization_enabled}
                                onChange={handleChange}
                            />
                            <span>Enable Face Anonymization</span>
                        </label>
                        <p className="settings-field__help">
                            Automatically blur faces in captured frames (GDPR compliance)
                        </p>
                    </div>
                </section>

                {/* Telegram Alerts */}
                <section className="settings-card glass">
                    <h2 className="settings-card__title">📱 Telegram Alerts</h2>

                    <div className="settings-field">
                        <label className="settings-field__checkbox">
                            <input
                                type="checkbox"
                                name="telegram_enabled"
                                checked={settings.telegram_enabled}
                                onChange={handleChange}
                            />
                            <span>Enable Telegram Notifications</span>
                        </label>
                    </div>

                    {settings.telegram_enabled && (
                        <>
                            <div className="settings-field">
                                <label className="settings-field__label">Bot Token</label>
                                <input
                                    type="password"
                                    name="telegram_token"
                                    value={settings.telegram_token}
                                    onChange={handleChange}
                                    placeholder="Enter your Telegram bot token"
                                    className="settings-field__input"
                                />
                            </div>

                            <div className="settings-field">
                                <label className="settings-field__label">Chat ID</label>
                                <input
                                    type="text"
                                    name="telegram_chat_id"
                                    value={settings.telegram_chat_id}
                                    onChange={handleChange}
                                    placeholder="Enter target chat ID"
                                    className="settings-field__input"
                                />
                            </div>
                        </>
                    )}
                </section>

                {/* System Info */}
                <section className="settings-card glass">
                    <h2 className="settings-card__title">ℹ️ System Information</h2>

                    {systemInfo ? (
                        <div className="system-info">
                            <div className="system-info__row">
                                <span className="system-info__label">Model Status</span>
                                <span className={`system-info__value system-info__value--${systemInfo.ml_service?.status || 'unknown'}`}>
                                    {systemInfo.ml_service?.status || 'Unknown'}
                                </span>
                            </div>
                            <div className="system-info__row">
                                <span className="system-info__label">Model Path</span>
                                <code className="system-info__code">
                                    {systemInfo.ml_service?.model_path || 'Not loaded'}
                                </code>
                            </div>
                            <div className="system-info__row">
                                <span className="system-info__label">Anonymization</span>
                                <span className="system-info__value">
                                    {systemInfo.anonymization ? '✅ Enabled' : '❌ Disabled'}
                                </span>
                            </div>
                            <div className="system-info__row">
                                <span className="system-info__label">API Version</span>
                                <span className="system-info__value">v2.1</span>
                            </div>
                        </div>
                    ) : (
                        <p className="settings-card__empty">Unable to load system information</p>
                    )}
                </section>
            </div>

            <div className="settings-view__actions">
                <button
                    className="btn btn--primary"
                    onClick={handleSave}
                    disabled={saving}
                >
                    {saving ? 'Saving...' : 'Save Settings'}
                </button>
            </div>
        </div>
    );
};

export default SettingsView;
