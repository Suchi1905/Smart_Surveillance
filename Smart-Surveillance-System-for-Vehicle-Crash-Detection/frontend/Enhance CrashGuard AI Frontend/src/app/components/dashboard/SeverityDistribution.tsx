import { useState, useEffect } from 'react';
import { PieChart, Pie, Cell, ResponsiveContainer, Legend, Tooltip } from 'recharts';

interface CrashStats {
  severity_distribution?: {
    Severe: number;
    Moderate: number;
    Mild: number;
  };
}

export default function SeverityDistribution() {
  const [crashStats, setCrashStats] = useState<CrashStats>({});

  useEffect(() => {
    fetch('/api/v1/crashes/stats/summary')
      .then(res => {
        if (!res.ok || !res.headers.get('content-type')?.includes('application/json')) {
          throw new Error('Backend not available');
        }
        return res.json();
      })
      .then(data => setCrashStats(data))
      .catch(() => {
        // Silently use mock data when backend is not available
      });
  }, []);

  // Use API data or fallback to mock data
  const severityData = crashStats.severity_distribution
    ? [
        { name: 'Severe', value: crashStats.severity_distribution.Severe, color: '#ef4444' },
        { name: 'Moderate', value: crashStats.severity_distribution.Moderate, color: '#f59e0b' },
        { name: 'Mild', value: crashStats.severity_distribution.Mild, color: '#22d3ee' }
      ]
    : [
        { name: 'Severe', value: 3, color: '#ef4444' },
        { name: 'Moderate', value: 8, color: '#f59e0b' },
        { name: 'Mild', value: 12, color: '#22d3ee' }
      ];

  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="glass rounded-lg p-3 border border-border-subtle">
          <p className="text-xs text-text-2">{payload[0].name}</p>
          <p className="text-sm font-bold text-brand-cyan">{payload[0].value}</p>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="glass rounded-xl p-6 h-full">
      <div className="flex items-center gap-2 mb-4">
        <span className="text-2xl">📊</span>
        <h2 className="text-lg font-bold text-text-1">Severity Distribution</h2>
      </div>

      <div className="h-[280px]">
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie
              data={severityData}
              cx="50%"
              cy="50%"
              innerRadius={50}
              outerRadius={80}
              paddingAngle={5}
              dataKey="value"
            >
              {severityData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.color} />
              ))}
            </Pie>
            <Tooltip content={<CustomTooltip />} />
            <Legend
              verticalAlign="bottom"
              height={36}
              formatter={(value) => <span className="text-text-2 text-sm">{value}</span>}
            />
          </PieChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
