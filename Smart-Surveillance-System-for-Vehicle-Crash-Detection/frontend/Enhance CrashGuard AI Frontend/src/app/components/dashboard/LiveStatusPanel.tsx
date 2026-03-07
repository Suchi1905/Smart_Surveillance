import { Activity, Zap, AlertCircle, CheckCircle } from 'lucide-react';
import { Progress } from '../ui/progress';
import { motion } from 'motion/react';

interface LiveStatusPanelProps {
  dashboardData: {
    total_vehicles_tracked: number;
    total_behavior_alerts: number;
    total_incidents: number;
    active_cameras: number;
  };
}

export default function LiveStatusPanel({ dashboardData }: LiveStatusPanelProps) {
  const metrics = [
    {
      label: 'Active Tracking',
      value: `${dashboardData.total_vehicles_tracked} vehicles`,
      icon: Activity,
      color: 'brand-cyan',
      emoji: '🚗'
    },
    {
      label: 'Processing FPS',
      value: '28.5 fps',
      icon: Zap,
      color: 'success',
      emoji: '⚡'
    },
    {
      label: 'Alerts Today',
      value: `${dashboardData.total_behavior_alerts}`,
      icon: AlertCircle,
      color: 'warning',
      emoji: '🚨'
    },
    {
      label: 'AI Model Status',
      value: 'Operational',
      icon: CheckCircle,
      color: 'success',
      emoji: '✅'
    }
  ];

  return (
    <div className="glass rounded-xl p-6">
      <div className="flex items-center gap-2 mb-6">
        <span className="text-2xl">📊</span>
        <h2 className="text-xl font-bold text-text-1">Live Status</h2>
        <div className="ml-auto flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-danger pulse-live"></div>
          <span className="text-xs text-text-2 font-mono uppercase tracking-wide">LIVE</span>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        {metrics.map((metric, index) => {
          const Icon = metric.icon;
          return (
            <motion.div
              key={metric.label}
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: index * 0.05 }}
              className="bg-bg-elevated/50 rounded-lg p-4 border border-border-subtle"
            >
              <div className="flex items-center gap-2 mb-2">
                <span className="text-lg">{metric.emoji}</span>
                <p className="text-xs text-text-3 uppercase tracking-wide">{metric.label}</p>
              </div>
              <p className="text-lg font-bold text-text-1 font-mono">{metric.value}</p>
            </motion.div>
          );
        })}
      </div>

      <div className="space-y-3">
        <div className="flex items-center justify-between text-sm">
          <span className="text-text-2">Detection Confidence</span>
          <span className="text-text-1 font-mono font-bold">94.3%</span>
        </div>
        <Progress value={94.3} className="h-2" />
      </div>
    </div>
  );
}
