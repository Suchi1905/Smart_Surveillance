import { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { TrendingUp, TrendingDown } from 'lucide-react';

export default function IncidentTimeline() {
  const [incidents, setIncidents] = useState<any[]>([]);

  const processTimelineData = (crashes: any[]) => {
    // Group crashes by hour
    const hourlyMap = new Map();
    crashes.forEach((crash: any) => {
      const hour = new Date(crash.timestamp).getHours();
      hourlyMap.set(hour, (hourlyMap.get(hour) || 0) + 1);
    });

    // Create timeline data for last 24 hours
    const timeline = [];
    const now = new Date();
    for (let i = 23; i >= 0; i--) {
      const hour = (now.getHours() - i + 24) % 24;
      timeline.push({
        time: `${hour.toString().padStart(2, '0')}:00`,
        count: hourlyMap.get(hour) || 0
      });
    }
    return timeline;
  };

  useEffect(() => {
    fetch('/api/v1/crashes')
      .then(res => {
        if (!res.ok || !res.headers.get('content-type')?.includes('application/json')) {
          throw new Error('Backend not available');
        }
        return res.json();
      })
      .then(data => {
        if (data.crashes) {
          // Group by hour for last 24 hours
          const hourlyData = processTimelineData(data.crashes);
          setIncidents(hourlyData);
        }
      })
      .catch(() => {
        // Silently use mock data when backend is not available
      });
  }, []);

  // Mock timeline data
  const mockTimeline = [
    { time: '00:00', count: 1 },
    { time: '02:00', count: 0 },
    { time: '04:00', count: 2 },
    { time: '06:00', count: 1 },
    { time: '08:00', count: 3 },
    { time: '10:00', count: 2 },
    { time: '12:00', count: 5 },
    { time: '14:00', count: 3 },
    { time: '16:00', count: 4 },
    { time: '18:00', count: 6 },
    { time: '20:00', count: 2 },
    { time: '22:00', count: 1 }
  ];

  const timelineData = incidents.length > 0 ? incidents : mockTimeline;
  const totalIncidents = timelineData.reduce((sum, item) => sum + item.count, 0);
  
  // Calculate trend
  const firstHalf = timelineData.slice(0, Math.floor(timelineData.length / 2)).reduce((sum, item) => sum + item.count, 0);
  const secondHalf = timelineData.slice(Math.floor(timelineData.length / 2)).reduce((sum, item) => sum + item.count, 0);
  const trend = secondHalf > firstHalf ? 'up' : 'down';

  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="glass rounded-lg p-3 border border-border-subtle">
          <p className="text-xs text-text-2">{payload[0].payload.time}</p>
          <p className="text-sm font-bold text-brand-cyan">{payload[0].value} incidents</p>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="glass rounded-xl p-6 h-full">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <span className="text-2xl">📈</span>
          <h2 className="text-lg font-bold text-text-1">24h Timeline</h2>
        </div>
        <div className="flex items-center gap-2">
          {trend === 'up' ? (
            <TrendingUp className="w-4 h-4 text-danger" />
          ) : (
            <TrendingDown className="w-4 h-4 text-success" />
          )}
          <span className="text-xs font-mono text-text-3">{totalIncidents} total</span>
        </div>
      </div>

      <div className="h-[280px]">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={timelineData}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
            <XAxis
              dataKey="time"
              stroke="#94a3b8"
              tick={{ fill: '#94a3b8', fontSize: 10 }}
              interval="preserveStartEnd"
            />
            <YAxis
              stroke="#94a3b8"
              tick={{ fill: '#94a3b8', fontSize: 10 }}
            />
            <Tooltip content={<CustomTooltip />} />
            <Line
              type="monotone"
              dataKey="count"
              stroke="#22d3ee"
              strokeWidth={2}
              dot={{ fill: '#22d3ee', r: 3 }}
              activeDot={{ r: 5, fill: '#22d3ee', strokeWidth: 2, stroke: '#0ea5e9' }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
