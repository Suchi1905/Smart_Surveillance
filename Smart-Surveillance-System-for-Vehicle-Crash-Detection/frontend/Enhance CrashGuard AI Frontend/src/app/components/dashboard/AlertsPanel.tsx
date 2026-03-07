import { useState, useEffect } from 'react';
import { Bell, BellOff, Volume2, VolumeX, CheckCircle, XCircle, Clock } from 'lucide-react';
import { toast } from 'sonner';
import { motion, AnimatePresence } from 'motion/react';
import { Button } from '../ui/button';

interface Alert {
  id: string;
  severity: 'Severe' | 'Moderate' | 'Mild';
  confidence: number;
  track_id: number;
  description: string;
  timestamp: Date;
  telegram_sent?: boolean;
}

export default function AlertsPanel() {
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [soundEnabled, setSoundEnabled] = useState(true);
  const [ws, setWs] = useState<WebSocket | null>(null);

  useEffect(() => {
    // Connect to WebSocket for real-time alerts
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const websocket = new WebSocket(`${protocol}//${window.location.host}/ws/alerts`);

    websocket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === 'alert' && data.data) {
        const newAlert: Alert = {
          id: `${Date.now()}-${Math.random()}`,
          severity: data.data.severity,
          confidence: data.data.confidence,
          track_id: data.data.track_id,
          description: data.data.description,
          timestamp: new Date(),
          telegram_sent: data.data.telegram_sent
        };

        setAlerts(prev => [newAlert, ...prev].slice(0, 50));

        // Show toast notification for critical alerts
        if (newAlert.severity === 'Severe') {
          toast.error(`🚨 Critical: ${newAlert.description}`, {
            description: `Confidence: ${(newAlert.confidence * 100).toFixed(1)}%`,
            duration: 5000
          });

          // Play alert sound
          if (soundEnabled) {
            const audio = new Audio('data:audio/wav;base64,UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+DyvmwhBTGH0PLPeCkGI3vJ8NuJOgkVYrjn6KFRFAw=' );
            audio.play().catch(() => {});
          }
        } else if (newAlert.severity === 'Moderate') {
          toast.warning(`⚠️ Alert: ${newAlert.description}`, {
            description: `Confidence: ${(newAlert.confidence * 100).toFixed(1)}%`
          });
        }
      }
    };

    websocket.onerror = () => {
      // Silently handle WebSocket errors when backend is not available
    };

    setWs(websocket);

    return () => {
      websocket.close();
    };
  }, [soundEnabled]);

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'Severe': return 'danger';
      case 'Moderate': return 'warning';
      case 'Mild': return 'brand-cyan';
      default: return 'text-3';
    }
  };

  const getSeverityBg = (severity: string) => {
    switch (severity) {
      case 'Severe': return 'bg-danger/10 border-danger/30';
      case 'Moderate': return 'bg-warning/10 border-warning/30';
      case 'Mild': return 'bg-brand-cyan/10 border-brand-cyan/30';
      default: return 'bg-text-3/10 border-text-3/30';
    }
  };

  return (
    <div className="glass rounded-xl p-6 flex flex-col h-[500px]">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <span className="text-2xl">🔔</span>
          <h2 className="text-xl font-bold text-text-1">Live Alerts</h2>
          <div className="w-2 h-2 rounded-full bg-danger pulse-live"></div>
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setSoundEnabled(!soundEnabled)}
            className="text-text-3 hover:text-text-1"
          >
            {soundEnabled ? <Volume2 className="w-4 h-4" /> : <VolumeX className="w-4 h-4" />}
          </Button>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto space-y-3 scrollbar-thin">
        <AnimatePresence initial={false}>
          {alerts.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full text-text-3">
              <Bell className="w-12 h-12 mb-2 opacity-30" />
              <p className="text-sm">No alerts yet</p>
              <p className="text-xs mt-1">System is monitoring...</p>
            </div>
          ) : (
            alerts.map((alert) => (
              <motion.div
                key={alert.id}
                initial={{ opacity: 0, x: 50 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -50 }}
                className={`p-4 rounded-lg border ${getSeverityBg(alert.severity)} relative overflow-hidden`}
              >
                {/* Severity indicator bar */}
                <div 
                  className={`absolute left-0 top-0 bottom-0 w-1 bg-${getSeverityColor(alert.severity)}`}
                ></div>

                <div className="ml-3">
                  <div className="flex items-start justify-between mb-2">
                    <div className="flex items-center gap-2">
                      <span className={`px-2 py-0.5 text-xs font-bold rounded text-${getSeverityColor(alert.severity)} bg-${getSeverityColor(alert.severity)}/20`}>
                        {alert.severity}
                      </span>
                      <span className="text-xs text-text-3 font-mono">
                        Track #{alert.track_id}
                      </span>
                    </div>
                    {alert.telegram_sent !== undefined && (
                      <div className="flex items-center gap-1">
                        {alert.telegram_sent ? (
                          <CheckCircle className="w-3 h-3 text-success" />
                        ) : (
                          <XCircle className="w-3 h-3 text-danger" />
                        )}
                      </div>
                    )}
                  </div>

                  <p className="text-sm text-text-1 mb-2">{alert.description}</p>

                  <div className="flex items-center justify-between text-xs">
                    <span className="text-text-3 font-mono">
                      Confidence: <span className="text-brand-cyan font-bold">
                        {(alert.confidence * 100).toFixed(1)}%
                      </span>
                    </span>
                    <span className="text-text-3 font-mono flex items-center gap-1">
                      <Clock className="w-3 h-3" />
                      {alert.timestamp.toLocaleTimeString()}
                    </span>
                  </div>
                </div>
              </motion.div>
            ))
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}