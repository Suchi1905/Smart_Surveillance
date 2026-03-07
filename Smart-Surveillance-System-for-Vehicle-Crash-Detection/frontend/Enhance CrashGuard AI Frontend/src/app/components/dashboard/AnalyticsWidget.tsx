import { useState, useEffect } from 'react';
import { BarChart, Bar, LineChart, Line, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { TrendingUp, Activity, AlertCircle } from 'lucide-react';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '../ui/tabs';

interface SpeedData {
  average_speed: number;
  max_speed: number;
  speeding_count: number;
  speed_distribution: { range: string; count: number }[];
  total_vehicles: number;
}

export default function AnalyticsWidget() {
  const [speedData, setSpeedData] = useState<SpeedData | null>(null);
  const [behaviorData, setBehaviorData] = useState<any>(null);

  useEffect(() => {
    // Fetch speed analytics
    fetch('/api/v1/analytics/speed?hours=24')
      .then(res => {
        if (!res.ok || !res.headers.get('content-type')?.includes('application/json')) {
          throw new Error('Backend not available');
        }
        return res.json();
      })
      .then(data => setSpeedData(data))
      .catch(() => {
        // Silently use mock data when backend is not available
      });

    // Fetch behavior analytics
    fetch('/api/v1/analytics/behavior')
      .then(res => {
        if (!res.ok || !res.headers.get('content-type')?.includes('application/json')) {
          throw new Error('Backend not available');
        }
        return res.json();
      })
      .then(data => setBehaviorData(data))
      .catch(() => {
        // Silently use mock data when backend is not available
      });
  }, []);

  // Mock data for demonstration
  const mockSpeedDistribution = [
    { range: '0-20', count: 45 },
    { range: '20-40', count: 120 },
    { range: '40-60', count: 85 },
    { range: '60-80', count: 40 },
    { range: '80+', count: 15 }
  ];

  const mockIncidentTrend = [
    { hour: '00:00', incidents: 2 },
    { hour: '04:00', incidents: 1 },
    { hour: '08:00', incidents: 5 },
    { hour: '12:00', incidents: 8 },
    { hour: '16:00', incidents: 6 },
    { hour: '20:00', incidents: 4 }
  ];

  const mockBehaviorTypes = [
    { name: 'Normal', value: 450, color: '#10b981' },
    { name: 'Aggressive', value: 25, color: '#f59e0b' },
    { name: 'Erratic', value: 15, color: '#ef4444' }
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
    <div className="glass rounded-xl p-6 flex flex-col h-[500px]">
      <div className="flex items-center gap-2 mb-4">
        <span className="text-2xl">📈</span>
        <h2 className="text-xl font-bold text-text-1">Analytics</h2>
      </div>

      <Tabs defaultValue="speed" className="flex-1 flex flex-col">
        <TabsList className="grid grid-cols-3 w-full">
          <TabsTrigger value="speed" className="text-xs">
            <TrendingUp className="w-3 h-3 mr-1" />
            Speed
          </TabsTrigger>
          <TabsTrigger value="incidents" className="text-xs">
            <AlertCircle className="w-3 h-3 mr-1" />
            Incidents
          </TabsTrigger>
          <TabsTrigger value="behavior" className="text-xs">
            <Activity className="w-3 h-3 mr-1" />
            Behavior
          </TabsTrigger>
        </TabsList>

        <TabsContent value="speed" className="flex-1 mt-4">
          <div className="space-y-4">
            <div className="grid grid-cols-3 gap-3">
              <div className="bg-bg-elevated/50 rounded-lg p-3 border border-border-subtle">
                <p className="text-xs text-text-3 mb-1">Avg Speed</p>
                <p className="text-lg font-bold text-text-1 font-mono">
                  {speedData?.average_speed.toFixed(1) || '45.3'} km/h
                </p>
              </div>
              <div className="bg-bg-elevated/50 rounded-lg p-3 border border-border-subtle">
                <p className="text-xs text-text-3 mb-1">Max Speed</p>
                <p className="text-lg font-bold text-warning font-mono">
                  {speedData?.max_speed.toFixed(1) || '92.5'} km/h
                </p>
              </div>
              <div className="bg-bg-elevated/50 rounded-lg p-3 border border-border-subtle">
                <p className="text-xs text-text-3 mb-1">Speeding</p>
                <p className="text-lg font-bold text-danger font-mono">
                  {speedData?.speeding_count || 15}
                </p>
              </div>
            </div>

            <div className="h-[250px]">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={speedData?.speed_distribution || mockSpeedDistribution}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                  <XAxis 
                    dataKey="range" 
                    stroke="#94a3b8" 
                    tick={{ fill: '#94a3b8', fontSize: 12 }}
                  />
                  <YAxis 
                    stroke="#94a3b8"
                    tick={{ fill: '#94a3b8', fontSize: 12 }}
                  />
                  <Tooltip content={<CustomTooltip />} />
                  <Bar dataKey="count" fill="url(#speedGradient)" radius={[8, 8, 0, 0]} />
                  <defs>
                    <linearGradient id="speedGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#22d3ee" stopOpacity={0.8} />
                      <stop offset="100%" stopColor="#6366f1" stopOpacity={0.3} />
                    </linearGradient>
                  </defs>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </TabsContent>

        <TabsContent value="incidents" className="flex-1 mt-4">
          <div className="h-[320px]">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={mockIncidentTrend}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                <XAxis 
                  dataKey="hour" 
                  stroke="#94a3b8"
                  tick={{ fill: '#94a3b8', fontSize: 12 }}
                />
                <YAxis 
                  stroke="#94a3b8"
                  tick={{ fill: '#94a3b8', fontSize: 12 }}
                />
                <Tooltip content={<CustomTooltip />} />
                <Line 
                  type="monotone" 
                  dataKey="incidents" 
                  stroke="#22d3ee" 
                  strokeWidth={3}
                  dot={{ fill: '#22d3ee', r: 4 }}
                  activeDot={{ r: 6, fill: '#22d3ee', strokeWidth: 2, stroke: '#0ea5e9' }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </TabsContent>

        <TabsContent value="behavior" className="flex-1 mt-4">
          <div className="h-[320px] flex items-center justify-center">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={mockBehaviorTypes}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={100}
                  paddingAngle={5}
                  dataKey="value"
                >
                  {mockBehaviorTypes.map((entry, index) => (
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
        </TabsContent>
      </Tabs>
    </div>
  );
}