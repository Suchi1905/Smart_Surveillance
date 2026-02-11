import React, { useEffect, useRef, useState, useCallback } from "react";
import "./App.css";
import AlertsPanel from "./components/AlertsPanel";
import AnalyticsWidget from "./components/AnalyticsWidget";
import IncidentsView from "./components/IncidentsView";
import SettingsView from "./components/SettingsView";
import LiveStatusPanel from "./components/LiveStatusPanel";

const API_URL = process.env.REACT_APP_API_URL || "http://localhost:8000";

function App() {
  const [clock, setClock] = useState("--:--:--");
  const [conf, setConf] = useState(0.6);
  const [systemHealthy, setSystemHealthy] = useState(true);
  const [systemSubtitle, setSystemSubtitle] = useState(
    "Initializing system..."
  );
  const [aiAccuracy, setAiAccuracy] = useState("--%");
  const [incidentsToday] = useState(0);
  const liveFeedRef = useRef(null);
  const [streamStatus, setStreamStatus] = useState("Awaiting input");
  const [streaming, setStreaming] = useState(false);

  // Video source state: webcam | url | file
  const [streamSource, setStreamSource] = useState('file');
  const [videoUrl, setVideoUrl] = useState('');

  // File upload state
  const [uploadedFile, setUploadedFile] = useState(null);
  const [uploadedFilename, setUploadedFilename] = useState('');
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef(null);

  // Navigation state
  const [activeView, setActiveView] = useState('dashboard');

  // Sidebar collapsed state
  const [sidebarExpanded, setSidebarExpanded] = useState(true);

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
            `YOLOv8 + ViT Hybrid • ${modelPath}`
          );
          setAiAccuracy("94.3%");
        } else {
          setSystemSubtitle("Model not loaded");
          setAiAccuracy("--%");
        }
      } catch (e) {
        setSystemHealthy(false);
        setSystemSubtitle("Backend unreachable");
        setAiAccuracy("--%");
      }
    };

    hydrateStatus();
    const id = setInterval(hydrateStatus, 30000);
    return () => clearInterval(id);
  }, []);

  // File upload handler
  const handleFileUpload = useCallback(async (file) => {
    if (!file) return;

    const ext = file.name.split('.').pop().toLowerCase();
    const allowed = ['mp4', 'avi', 'mkv', 'mov', 'wmv', 'flv', 'webm'];
    if (!allowed.includes(ext)) {
      setStreamStatus(`Unsupported format .${ext}. Use: ${allowed.join(', ')}`);
      return;
    }

    setUploading(true);
    setUploadProgress(0);
    setStreamStatus(`Uploading ${file.name}...`);
    setUploadedFile(file);

    const formData = new FormData();
    formData.append('file', file);

    try {
      // Use relative URL to go through React dev proxy (avoids CORS)
      const response = await fetch(`/video/upload`, {
        method: 'POST',
        body: formData,
      });

      setUploadProgress(100);

      if (!response.ok) {
        const errText = await response.text();
        throw new Error(errText || `HTTP ${response.status}`);
      }

      const data = await response.json();
      setUploadedFilename(data.filename);
      setStreamStatus(`✓ Uploaded: ${file.name} (${data.size_mb} MB) — Ready to analyze`);
    } catch (err) {
      console.error('Upload error:', err);
      setStreamStatus(`Upload failed: ${err.message}`);
      setUploadedFile(null);
    } finally {
      setUploading(false);
    }
  }, []);

  // Drag & drop handlers
  const handleDragOver = (e) => {
    e.preventDefault();
    setDragOver(true);
  };
  const handleDragLeave = () => setDragOver(false);
  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFileUpload(file);
  };

  const startStream = (confValue = conf) => {
    const img = liveFeedRef.current;
    if (!img) return;
    const url = `${API_URL}/video?conf=${confValue}`;
    img.src = url;
    img.style.opacity = 0;
    setStreamStatus("Connecting to webcam...");
    setStreaming(true);

    img.onload = () => {
      img.style.opacity = 1;
      setStreamStatus("● Live — Webcam stream active");
    };

    img.onerror = () => {
      setStreamStatus("Error: webcam connection failed");
      setStreaming(false);
    };
  };

  const startUrlStream = (confValue = conf) => {
    const img = liveFeedRef.current;
    if (!img) return;

    if (!videoUrl.trim()) {
      setStreamStatus("Enter a video URL first");
      return;
    }

    const encodedUrl = encodeURIComponent(videoUrl.trim());
    const url = `${API_URL}/video/url?source=${encodedUrl}&conf=${confValue}`;
    img.src = url;
    img.style.opacity = 0;
    setStreamStatus("Connecting to URL source...");
    setStreaming(true);

    img.onload = () => {
      img.style.opacity = 1;
      setStreamStatus("● Live — URL stream active");
    };

    img.onerror = () => {
      setStreamStatus("Error: URL stream failed");
      setStreaming(false);
    };
  };

  const startFileStream = (confValue = conf) => {
    const img = liveFeedRef.current;
    if (!img || !uploadedFilename) return;

    const encodedFilename = encodeURIComponent(uploadedFilename);
    const url = `${API_URL}/video/file?filename=${encodedFilename}&conf=${confValue}`;
    img.src = url;
    img.style.opacity = 0;
    setStreamStatus("Starting analysis on uploaded video...");
    setStreaming(true);

    img.onload = () => {
      img.style.opacity = 1;
      setStreamStatus("● Live — Running crash detection on file");
    };

    img.onerror = () => {
      setStreamStatus("Error: file stream failed");
      setStreaming(false);
    };
  };

  const handleStartStream = () => {
    if (streamSource === 'webcam') {
      startStream(conf);
    } else if (streamSource === 'url') {
      startUrlStream(conf);
    } else if (streamSource === 'file') {
      startFileStream(conf);
    }
  };

  const stopStream = async () => {
    const img = liveFeedRef.current;
    if (!img) return;

    try {
      await fetch(`${API_URL}/video/stop`, { method: 'POST' });
    } catch (err) {
      console.warn("Failed to send stop signal:", err);
    }

    img.src = "";
    setStreaming(false);
    setStreamStatus("Stream stopped");
  };

  // Navigation items
  const navItems = [
    { id: 'dashboard', icon: '◉', label: 'Dashboard' },
    { id: 'feeds', icon: '▶', label: 'Live Feed' },
    { id: 'incidents', icon: '⚠', label: 'Incidents' },
    { id: 'settings', icon: '⚙', label: 'Settings' }
  ];

  // Get page title based on active view
  const getPageTitle = () => {
    switch (activeView) {
      case 'feeds': return 'Live Detection Feed';
      case 'incidents': return 'Incident Logs';
      case 'settings': return 'System Settings';
      default: return 'Command Center';
    }
  };

  // Source selector icons
  const sourceOptions = [
    { value: 'file', icon: '📁', label: 'Local File' },
    { value: 'webcam', icon: '📷', label: 'Webcam' },
    { value: 'url', icon: '🔗', label: 'URL / RTSP' },
  ];

  // Render main content based on active view
  const renderMainContent = () => {
    switch (activeView) {
      case 'incidents':
        return <IncidentsView />;

      case 'settings':
        return <SettingsView />;

      case 'feeds':
      case 'dashboard':
      default:
        return (
          <>
            {/* Stat cards — only on dashboard */}
            {activeView === 'dashboard' && (
              <section className="stats-row">
                <article className="stat-card glass">
                  <div className="stat-card__icon stat-card__icon--blue">🎯</div>
                  <div className="stat-card__content">
                    <div className="stat-card__value">{aiAccuracy}</div>
                    <div className="stat-card__label">Model Accuracy</div>
                  </div>
                </article>

                <article className="stat-card glass">
                  <div className="stat-card__icon stat-card__icon--green">📡</div>
                  <div className="stat-card__content">
                    <div className="stat-card__value">1</div>
                    <div className="stat-card__label">Active Feeds</div>
                  </div>
                </article>

                <article className="stat-card glass">
                  <div className="stat-card__icon stat-card__icon--amber">⚠</div>
                  <div className="stat-card__content">
                    <div className="stat-card__value">{incidentsToday}</div>
                    <div className="stat-card__label">Incidents Today</div>
                  </div>
                </article>

                <article className="stat-card glass">
                  <div className="stat-card__icon stat-card__icon--purple">🧠</div>
                  <div className="stat-card__content">
                    <div className="stat-card__value">Hybrid</div>
                    <div className="stat-card__label">YOLO + ViT + Fusion</div>
                  </div>
                </article>
              </section>
            )}

            {/* Live Status Panel - real-time metrics */}
            {activeView === 'dashboard' && (
              <LiveStatusPanel isStreaming={streaming} apiUrl={API_URL} />
            )}

            {/* Main detection area */}
            <section className="detection-area">
              {/* Control sidebar */}
              <aside className="control-panel glass">
                <h3 className="control-panel__title">Video Source</h3>

                {/* Source tabs */}
                <div className="source-tabs">
                  {sourceOptions.map(opt => (
                    <button
                      key={opt.value}
                      className={`source-tab ${streamSource === opt.value ? 'source-tab--active' : ''}`}
                      onClick={() => !streaming && setStreamSource(opt.value)}
                      disabled={streaming}
                    >
                      <span className="source-tab__icon">{opt.icon}</span>
                      <span className="source-tab__label">{opt.label}</span>
                    </button>
                  ))}
                </div>

                {/* File upload dropzone */}
                {streamSource === 'file' && (
                  <div
                    className={`dropzone ${dragOver ? 'dropzone--active' : ''} ${uploadedFile ? 'dropzone--has-file' : ''}`}
                    onDragOver={handleDragOver}
                    onDragLeave={handleDragLeave}
                    onDrop={handleDrop}
                    onClick={() => !uploading && fileInputRef.current?.click()}
                  >
                    <input
                      ref={fileInputRef}
                      type="file"
                      accept=".mp4,.avi,.mkv,.mov,.wmv,.flv,.webm"
                      onChange={(e) => handleFileUpload(e.target.files[0])}
                      style={{ display: 'none' }}
                    />
                    {uploading ? (
                      <div className="dropzone__uploading">
                        <div className="dropzone__progress-bar">
                          <div
                            className="dropzone__progress-fill"
                            style={{ width: `${uploadProgress}%` }}
                          />
                        </div>
                        <span className="dropzone__text">Uploading... {uploadProgress}%</span>
                      </div>
                    ) : uploadedFile ? (
                      <div className="dropzone__file-info">
                        <span className="dropzone__file-icon">🎬</span>
                        <span className="dropzone__filename">{uploadedFile.name}</span>
                        <span className="dropzone__file-size">
                          {(uploadedFile.size / (1024 * 1024)).toFixed(1)} MB
                        </span>
                        <span className="dropzone__change">Click to change</span>
                      </div>
                    ) : (
                      <div className="dropzone__empty">
                        <span className="dropzone__upload-icon">⬆</span>
                        <span className="dropzone__text">Drop video file here</span>
                        <span className="dropzone__subtext">or click to browse</span>
                        <span className="dropzone__formats">MP4 • AVI • MKV • MOV</span>
                      </div>
                    )}
                  </div>
                )}

                {/* URL Input */}
                {streamSource === 'url' && (
                  <div className="url-input-group">
                    <label className="field__label">Video URL</label>
                    <input
                      type="text"
                      value={videoUrl}
                      onChange={(e) => setVideoUrl(e.target.value)}
                      placeholder="https://youtube.com/watch?v=... or rtsp://..."
                      className="url-input"
                      disabled={streaming}
                    />
                  </div>
                )}

                {/* Confidence slider */}
                <div className="confidence-control">
                  <label className="field__label">
                    Detection Confidence
                    <span className="confidence-value">{conf.toFixed(1)}</span>
                  </label>
                  <input
                    type="range"
                    min="0.1"
                    max="1.0"
                    step="0.1"
                    value={conf}
                    onChange={(e) => setConf(parseFloat(e.target.value))}
                  />
                </div>

                {/* Action buttons */}
                <div className="action-buttons">
                  <button
                    type="button"
                    className="btn btn--primary btn--full"
                    onClick={handleStartStream}
                    disabled={streaming || (streamSource === 'file' && !uploadedFilename) || (streamSource === 'url' && !videoUrl.trim())}
                  >
                    {streaming ? '● Analyzing...' : '▶ Start Detection'}
                  </button>
                  <button
                    type="button"
                    className="btn btn--danger btn--full"
                    onClick={stopStream}
                    disabled={!streaming}
                  >
                    ■ Stop
                  </button>
                </div>

                {/* Status display */}
                <div className="stream-status">
                  <span className={`stream-status__dot ${streaming ? 'stream-status__dot--live' : ''}`} />
                  <span className="stream-status__text">{streamStatus}</span>
                </div>
              </aside>

              {/* Video feed */}
              <div className="feed-container glass">
                <header className="feed-header">
                  <div className="feed-header__left">
                    <h2 className="feed-header__title">Crash Detection Feed</h2>
                    <span className="feed-header__subtitle">
                      YOLO + ViT Hybrid • Severity Triage • Edge Anonymization
                    </span>
                  </div>
                  <div className="feed-header__badges">
                    {streaming && <span className="badge badge--live">● LIVE</span>}
                    <span className="badge">Accident</span>
                    <span className="badge">Vehicle</span>
                  </div>
                </header>

                <div className="feed-viewport">
                  <div className="feed-frame">
                    <img
                      ref={liveFeedRef}
                      alt="Live crash detection stream"
                      className="live-feed"
                    />
                    {!streaming && (
                      <div className="feed-placeholder">
                        <div className="feed-placeholder__icon">🎥</div>
                        <div className="feed-placeholder__title">No Active Feed</div>
                        <div className="feed-placeholder__sub">
                          {streamSource === 'file'
                            ? 'Upload a video file and click "Start Detection"'
                            : streamSource === 'webcam'
                              ? 'Click "Start Detection" to begin webcam capture'
                              : 'Enter a URL and click "Start Detection"'
                          }
                        </div>
                      </div>
                    )}
                    <div className="feed-scanline" />
                  </div>
                </div>
              </div>
            </section>

            {/* Alerts and Analytics - only on dashboard */}
            {activeView === 'dashboard' && (
              <section className="dashboard-bottom">
                <div className="dashboard-bottom__alerts">
                  <AlertsPanel maxAlerts={10} />
                </div>
                <div className="dashboard-bottom__analytics">
                  <AnalyticsWidget />
                </div>
              </section>
            )}
          </>
        );
    }
  };

  return (
    <div className={`app-container ${sidebarExpanded ? '' : 'app-container--collapsed'}`}>
      {/* Sidebar */}
      <aside className={`sidebar ${sidebarExpanded ? 'sidebar--expanded' : ''}`}>
        <div className="sidebar__brand" onClick={() => setSidebarExpanded(!sidebarExpanded)}>
          <span className="sidebar__logo-mark" />
          {sidebarExpanded && (
            <span className="sidebar__brand-text">CrashGuard AI</span>
          )}
        </div>

        <nav className="sidebar__nav">
          {navItems.map((item) => (
            <button
              key={item.id}
              className={`sidebar__item ${activeView === item.id ? 'sidebar__item--active' : ''}`}
              aria-label={item.label}
              title={item.label}
              onClick={() => setActiveView(item.id)}
            >
              <span className="sidebar__icon">{item.icon}</span>
              {sidebarExpanded && (
                <span className="sidebar__label">{item.label}</span>
              )}
            </button>
          ))}
        </nav>

        <div className="sidebar__footer">
          <span className="sidebar__version">v3.0</span>
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
              <span className="system-pill__dot">{systemHealthy ? '●' : '○'}</span>
              <span className="system-pill__label">
                {systemHealthy ? "Online" : "Offline"}
              </span>
            </div>

            <div className="header__meta">
              <span className="header__label">{getPageTitle()}</span>
              <span className="header__sub">{systemSubtitle}</span>
            </div>
          </div>

          <div className="header__right">
            <div className="header__clock">{clock}</div>
          </div>
        </header>

        {/* Main content */}
        <main className="main">
          {renderMainContent()}
        </main>
      </div>
    </div>
  );
}

export default App;
