import React, { useEffect, useRef, useState } from "react";
import "./App.css";
import AlertsPanel from "./components/AlertsPanel";
import AnalyticsWidget from "./components/AnalyticsWidget";

const API_URL = process.env.REACT_APP_API_URL || "http://localhost:8000";

function App() {
  const [clock, setClock] = useState("--:--:--");
  const [conf, setConf] = useState(0.6);
  const [systemHealthy, setSystemHealthy] = useState(true);
  const [systemSubtitle, setSystemSubtitle] = useState(
    "Model loading status: checking..."
  );
  const [aiAccuracy, setAiAccuracy] = useState("--%");
  const [incidentsToday] = useState(0);
  const liveFeedRef = useRef(null);
  const [streamStatus, setStreamStatus] = useState("Stream idle");
  const [streaming, setStreaming] = useState(false);

  // Clock
  useEffect(() => {
    const updateClock = () => {
      const now = new Date();
      const parts = [
        now.getHours().toString().padStart(2, "0"),
        now.getMinutes().toString().padStart(2, "0"),
        now.getSeconds().toString().padStart(2, "0"),
      ];
      setClock(parts.join(":"));
    };
    updateClock();
    const id = setInterval(updateClock, 1000);
    return () => clearInterval(id);
  }, []);

  // System status
  useEffect(() => {
    const hydrateStatus = async () => {
      try {
        const [healthRes, statusRes] = await Promise.all([
          fetch(`${API_URL}/health`),
          fetch(`${API_URL}/api/system/status`),
        ]);

        const health = await healthRes.json();
        const sys = await statusRes.json();

        const healthy =
          health.status === "healthy" && health.model_loaded === true;
        setSystemHealthy(healthy);

        if (healthy) {
          const modelPath =
            sys.ml_service && sys.ml_service.model_path
              ? sys.ml_service.model_path
              : "loaded";
          setSystemSubtitle(
            `Model: ${modelPath} • Anonymization: ${sys.anonymization ? "enabled" : "disabled"
            }`
          );
          setAiAccuracy("94.3%");
        } else {
          setSystemSubtitle("Model not loaded or health check failed");
          setAiAccuracy("--%");
        }
      } catch (e) {
        setSystemHealthy(false);
        setSystemSubtitle("Unable to reach /health or /api/system/status");
        setAiAccuracy("--%");
      }
    };

    hydrateStatus();
    const id = setInterval(hydrateStatus, 30000);
    return () => clearInterval(id);
  }, []);

  // Initialize from query (?conf=...)
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const initialConf = parseFloat(params.get("conf") || "0.6");
    if (!Number.isNaN(initialConf)) {
      setConf(initialConf);
      startStream(initialConf);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const startStream = (confValue = conf) => {
    const img = liveFeedRef.current;
    if (!img) return;
    const url = `${API_URL}/video?conf=${confValue}`;
    img.src = url;
    img.style.opacity = 0;
    setStreamStatus("Connecting to stream...");
    setStreaming(true);

    img.onload = () => {
      img.style.opacity = 1;
      setStreamStatus("Live stream active");
    };

    img.onerror = () => {
      setStreamStatus("Error loading stream");
      setStreaming(false);
    };
  };

  const stopStream = () => {
    const img = liveFeedRef.current;
    if (img) {
      img.src = "";
    }
    setStreaming(false);
    setStreamStatus("Stream idle");
  };

  return (
    <div className="app-container">
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="sidebar__logo">
          <span className="sidebar__logo-mark" />
        </div>

        <nav className="sidebar__nav">
          <button
            className="sidebar__item sidebar__item--active"
            aria-label="Dashboard"
          >
            <span className="sidebar__icon">⬤</span>
          </button>
          <button className="sidebar__item" aria-label="Live Feeds">
            <span className="sidebar__icon">📹</span>
          </button>
          <button className="sidebar__item" aria-label="Incident Logs">
            <span className="sidebar__icon">⚠️</span>
          </button>
          <button className="sidebar__item" aria-label="Settings">
            <span className="sidebar__icon">⚙️</span>
          </button>
        </nav>

        <div className="sidebar__footer">
          <span className="sidebar__version">v2.1</span>
        </div>
      </aside>

      {/* Main layout */}
      <div className="layout">
        {/* Header */}
        <header className="header">
          <div className="header__left">
            <div
              className={
                "system-pill" + (systemHealthy ? "" : " system-pill--error")
              }
            >
              <span className="system-pill__dot">🟢</span>
              <span className="system-pill__label">
                {systemHealthy ? "System Healthy" : "System Warning"}
              </span>
            </div>

            <div className="header__meta">
              <span className="header__label">Smart Surveillance Dashboard</span>
              <span className="header__sub">{systemSubtitle}</span>
            </div>
          </div>

          <div className="header__right">
            <div className="header__clock">{clock}</div>

            <div className="header__search">
              <span className="header__search-icon">🔍</span>
              <input
                type="text"
                className="header__search-input"
                placeholder="Search cameras, incidents, zones..."
              />
            </div>
          </div>
        </header>

        {/* Main content */}
        <main className="main">
          {/* Analytics strip */}
          <section className="analytics">
            <article className="stat-card glass">
              <div className="stat-card__label">TOTAL CAMERAS</div>
              <div className="stat-card__value">4</div>
              <div className="stat-card__meta">All zones online</div>
            </article>

            <article className="stat-card glass">
              <div className="stat-card__label">UPTIME</div>
              <div className="stat-card__value">99.98%</div>
              <div className="stat-card__meta">Last restart: &lt; 24h</div>
            </article>

            <article className="stat-card glass">
              <div className="stat-card__label">INCIDENTS TODAY</div>
              <div className="stat-card__value">{incidentsToday}</div>
              <div className="stat-card__meta">Crash triage events</div>
            </article>

            <article className="stat-card glass">
              <div className="stat-card__label">AI ACCURACY</div>
              <div className="stat-card__value">{aiAccuracy}</div>
              <div className="stat-card__meta">Model performance (approx)</div>
            </article>
          </section>

          {/* Camera grid */}
          <section className="grid">
            {/* Primary live feed card */}
            <article className="camera-card glass camera-card--primary">
              <header className="camera-card__header">
                <div className="camera-card__title-group">
                  <h2 className="camera-card__title">Live Crash Detection Feed</h2>
                  <span className="camera-card__subtitle">
                    Edge anonymization • Severity triage • YOLO detection
                  </span>
                </div>

                <div className="camera-card__badges">
                  <span className="badge badge--live">LIVE</span>
                  <span className="badge">Accident</span>
                  <span className="badge">Vehicle</span>
                  <span className="badge">Human</span>
                </div>
              </header>

              <div className="camera-card__body">
                <div className="camera-card__controls">
                  <label className="field">
                    <span className="field__label">Confidence Threshold</span>
                    <div className="field__input-row">
                      <input
                        type="range"
                        min="0.1"
                        max="1.0"
                        step="0.1"
                        value={conf}
                        onChange={(e) => setConf(parseFloat(e.target.value))}
                      />
                      <span className="field__value">{conf.toFixed(1)}</span>
                    </div>
                  </label>

                  <div className="button-row">
                    <button
                      type="button"
                      className="btn btn--primary"
                      onClick={() => startStream(conf)}
                      disabled={streaming}
                    >
                      Start Detection
                    </button>
                    <button
                      type="button"
                      className="btn btn--ghost"
                      onClick={stopStream}
                      disabled={!streaming}
                    >
                      Stop
                    </button>
                  </div>

                  <div className="meta-row">
                    <span className="meta-row__item">
                      Stream URL:
                      <code>{`${API_URL}/video?conf=${conf.toFixed(1)}`}</code>
                    </span>
                    <span className="meta-row__item">{streamStatus}</span>
                  </div>
                </div>

                <div className="camera-card__feed">
                  <div className="camera-card__feed-frame">
                    <img
                      ref={liveFeedRef}
                      alt="Live crash detection stream"
                      className="live-feed"
                    />
                    <div className="camera-card__feed-overlay" />
                  </div>
                  <div className="camera-card__caption">
                    <span className="camera-card__caption-main">
                      Feed: North Intersection
                    </span>
                    <span className="camera-card__caption-sub">
                      Powered by /video endpoint &amp; real-time YOLO inference
                    </span>
                  </div>
                </div>
              </div>
            </article>
          </section>

          {/* New: Alerts and Analytics Section */}
          <section className="dashboard-extras">
            <div className="dashboard-extras__alerts">
              <AlertsPanel maxAlerts={10} />
            </div>
            <div className="dashboard-extras__analytics">
              <AnalyticsWidget />
            </div>
          </section>
        </main>
      </div>
    </div>
  );
}

export default App;
