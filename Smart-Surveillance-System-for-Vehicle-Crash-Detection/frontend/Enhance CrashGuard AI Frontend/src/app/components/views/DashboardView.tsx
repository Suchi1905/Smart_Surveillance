import { useState, useEffect } from 'react';
import { SystemConfig } from '../../App';
import StatCards from '../dashboard/StatCards';
import LiveStatusPanel from '../dashboard/LiveStatusPanel';
import DetectionArea from '../dashboard/DetectionArea';
import AlertsPanel from '../dashboard/AlertsPanel';
import AnalyticsWidget from '../dashboard/AnalyticsWidget';
import IncidentMap from '../dashboard/IncidentMap';
import SeverityDistribution from '../dashboard/SeverityDistribution';
import IncidentTimeline from '../dashboard/IncidentTimeline';

interface DashboardViewProps {
  config: SystemConfig;
}

export default function DashboardView({ config }: DashboardViewProps) {
  const [systemStatus, setSystemStatus] = useState({
    model_accuracy: 94.3,
    active_feeds: 1,
    incidents_today: 0,
    ai_model: 'YOLOv8 + ViT Hybrid'
  });

  const [dashboardData, setDashboardData] = useState({
    total_vehicles_tracked: 0,
    total_behavior_alerts: 0,
    total_incidents: 0,
    active_cameras: 0
  });

  useEffect(() => {
    // Fetch dashboard analytics
    fetch('/api/v1/analytics/dashboard')
      .then(res => {
        if (!res.ok || !res.headers.get('content-type')?.includes('application/json')) {
          throw new Error('Backend not available');
        }
        return res.json();
      })
      .then(data => {
        if (data.summary) {
          setDashboardData(data.summary);
        }
      })
      .catch(() => {
        // Silently use mock data when backend is not available
      });

    // Fetch incidents for today count
    fetch('/api/v1/crashes')
      .then(res => {
        if (!res.ok || !res.headers.get('content-type')?.includes('application/json')) {
          throw new Error('Backend not available');
        }
        return res.json();
      })
      .then(data => {
        const today = new Date().toDateString();
        const todayIncidents = data.crashes?.filter((crash: any) => 
          new Date(crash.timestamp).toDateString() === today
        ).length || 0;
        setSystemStatus(prev => ({ ...prev, incidents_today: todayIncidents }));
      })
      .catch(() => {
        // Silently use mock data when backend is not available
      });
  }, []);

  return (
    <div className="min-h-screen p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-text-1">Command Center</h1>
          <p className="text-text-2 mt-1">Real-time crash detection & monitoring</p>
        </div>
        <div className="flex items-center gap-2 glass px-4 py-2 rounded-lg">
          <div className="w-2 h-2 rounded-full bg-success animate-pulse"></div>
          <span className="text-sm text-text-2 font-mono">{new Date().toLocaleTimeString()}</span>
        </div>
      </div>

      {/* Stat Cards */}
      <StatCards stats={systemStatus} />

      {/* Live Status Panel */}
      <LiveStatusPanel dashboardData={dashboardData} />

      {/* Enhanced Analytics Row */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <IncidentMap />
        <SeverityDistribution />
        <IncidentTimeline />
      </div>

      {/* Detection Area */}
      <DetectionArea config={config} />

      {/* Bottom Panels */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <AlertsPanel />
        <AnalyticsWidget />
      </div>
    </div>
  );
}