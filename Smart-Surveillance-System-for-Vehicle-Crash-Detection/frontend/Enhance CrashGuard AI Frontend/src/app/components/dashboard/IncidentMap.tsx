import { useState, useEffect } from 'react';
import { MapPin } from 'lucide-react';

interface TrafficData {
  locations?: Array<{
    lat: number;
    lng: number;
    incidents: number;
  }>;
}

export default function IncidentMap() {
  const [trafficData, setTrafficData] = useState<TrafficData>({});

  useEffect(() => {
    fetch('/api/v1/analytics/traffic')
      .then(res => {
        if (!res.ok || !res.headers.get('content-type')?.includes('application/json')) {
          throw new Error('Backend not available');
        }
        return res.json();
      })
      .then(data => setTrafficData(data))
      .catch(() => {
        // Silently use mock data when backend is not available
      });
  }, []);

  // Mock incident heatmap data
  const mockIncidents = [
    { x: 25, y: 30, intensity: 3, label: 'Zone A' },
    { x: 60, y: 45, intensity: 5, label: 'Zone B' },
    { x: 40, y: 70, intensity: 2, label: 'Zone C' },
    { x: 75, y: 25, intensity: 4, label: 'Zone D' },
    { x: 50, y: 55, intensity: 1, label: 'Zone E' }
  ];

  const getIntensityColor = (intensity: number) => {
    if (intensity >= 5) return 'bg-danger';
    if (intensity >= 3) return 'bg-warning';
    return 'bg-brand-cyan';
  };

  const getIntensitySize = (intensity: number) => {
    return 8 + intensity * 4;
  };

  return (
    <div className="glass rounded-xl p-6 h-full">
      <div className="flex items-center gap-2 mb-4">
        <span className="text-2xl">🗺️</span>
        <h2 className="text-lg font-bold text-text-1">Incident Map</h2>
      </div>

      <div className="relative w-full h-[280px] bg-bg-elevated/30 rounded-lg border border-border-subtle overflow-hidden">
        {/* Grid overlay */}
        <div className="absolute inset-0 opacity-20" style={{
          backgroundImage: 'linear-gradient(rgba(255,255,255,0.05) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.05) 1px, transparent 1px)',
          backgroundSize: '20px 20px'
        }}></div>

        {/* Incident markers */}
        {mockIncidents.map((incident, index) => (
          <div
            key={index}
            className="absolute group cursor-pointer"
            style={{
              left: `${incident.x}%`,
              top: `${incident.y}%`,
              transform: 'translate(-50%, -50%)'
            }}
          >
            {/* Pulse effect */}
            <div
              className={`absolute ${getIntensityColor(incident.intensity)} rounded-full opacity-30 animate-ping`}
              style={{
                width: `${getIntensitySize(incident.intensity)}px`,
                height: `${getIntensitySize(incident.intensity)}px`,
                left: '50%',
                top: '50%',
                transform: 'translate(-50%, -50%)'
              }}
            ></div>

            {/* Main marker */}
            <div
              className={`${getIntensityColor(incident.intensity)} rounded-full opacity-80 transition-all duration-200 group-hover:scale-125`}
              style={{
                width: `${getIntensitySize(incident.intensity)}px`,
                height: `${getIntensitySize(incident.intensity)}px`
              }}
            ></div>

            {/* Tooltip */}
            <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none whitespace-nowrap">
              <div className="glass rounded px-2 py-1 border border-border-subtle">
                <p className="text-xs text-text-1 font-medium">{incident.label}</p>
                <p className="text-xs text-text-3">Incidents: {incident.intensity}</p>
              </div>
            </div>
          </div>
        ))}

        {/* Legend */}
        <div className="absolute bottom-4 right-4 glass rounded-lg p-3 border border-border-subtle">
          <p className="text-xs text-text-3 mb-2">Severity</p>
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 bg-brand-cyan rounded-full"></div>
              <span className="text-xs text-text-2">Low</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 bg-warning rounded-full"></div>
              <span className="text-xs text-text-2">Medium</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 bg-danger rounded-full"></div>
              <span className="text-xs text-text-2">High</span>
            </div>
          </div>
        </div>

        {/* Stats overlay */}
        <div className="absolute top-4 left-4 glass rounded-lg p-3 border border-border-subtle">
          <div className="flex items-center gap-2">
            <MapPin className="w-4 h-4 text-brand-cyan" />
            <div>
              <p className="text-xs text-text-3">Active Zones</p>
              <p className="text-lg font-bold text-text-1 font-mono">{mockIncidents.length}</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
