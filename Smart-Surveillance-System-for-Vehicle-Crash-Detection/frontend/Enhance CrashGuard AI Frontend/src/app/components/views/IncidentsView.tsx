import { useState, useEffect } from 'react';
import { Download, Filter, X, Calendar, MapPin } from 'lucide-react';
import { Button } from '../ui/button';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../ui/select';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '../ui/dialog';
import { motion, AnimatePresence } from 'motion/react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

interface Incident {
  id: string;
  severity: 'Severe' | 'Moderate' | 'Mild' | 'Insufficient Data';
  confidence: number;
  severity_index: number;
  timestamp: string;
  location?: string;
  description?: string;
}

export default function IncidentsView() {
  const [incidents, setIncidents] = useState<Incident[]>([]);
  const [severityFilter, setSeverityFilter] = useState<string>('all');
  const [selectedIncident, setSelectedIncident] = useState<Incident | null>(null);
  const [modalOpen, setModalOpen] = useState(false);

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
          setIncidents(data.crashes);
        }
      })
      .catch(() => {
        // Silently use mock data when backend is not available
      });
  }, []);

  const filteredIncidents = severityFilter === 'all'
    ? incidents
    : incidents.filter(i => i.severity === severityFilter);

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

  const handleExport = () => {
    const csv = [
      ['ID', 'Severity', 'Confidence', 'Severity Index', 'Timestamp', 'Location', 'Description'],
      ...filteredIncidents.map(i => [
        i.id,
        i.severity,
        i.confidence,
        i.severity_index,
        i.timestamp,
        i.location || '',
        i.description || ''
      ])
    ].map(row => row.join(',')).join('\n');

    const blob = new Blob([csv], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `incidents-${new Date().toISOString()}.csv`;
    a.click();
  };

  const handleDelete = async (id: string) => {
    try {
      await fetch(`/api/v1/crashes/${id}`, { method: 'DELETE' });
      setIncidents(prev => prev.filter(i => i.id !== id));
      setModalOpen(false);
    } catch (error) {
      console.error('Failed to delete incident:', error);
    }
  };

  // Mock severity timeline for selected incident
  const mockSeverityTimeline = [
    { time: '0s', severity: 2.1 },
    { time: '2s', severity: 3.5 },
    { time: '4s', severity: 5.2 },
    { time: '6s', severity: 7.8 },
    { time: '8s', severity: 8.9 }
  ];

  return (
    <div className="min-h-screen p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-text-1">Incident Log</h1>
          <p className="text-text-2 mt-1">Historical crash detection records</p>
        </div>
      </div>

      {/* Severity Timeline Chart */}
      <div className="glass rounded-xl p-6">
        <div className="flex items-center gap-2 mb-4">
          <span className="text-xl">📈</span>
          <h2 className="text-lg font-bold text-text-1">Severity Timeline</h2>
        </div>
        <div className="h-[200px]">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={mockSeverityTimeline}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
              <XAxis 
                dataKey="time" 
                stroke="#94a3b8"
                tick={{ fill: '#94a3b8', fontSize: 12 }}
              />
              <YAxis 
                stroke="#94a3b8"
                tick={{ fill: '#94a3b8', fontSize: 12 }}
                label={{ value: 'Severity Index', angle: -90, position: 'insideLeft', fill: '#94a3b8' }}
              />
              <Tooltip 
                contentStyle={{ 
                  background: 'rgba(17, 24, 39, 0.9)', 
                  border: '1px solid rgba(255,255,255,0.1)',
                  borderRadius: '8px'
                }}
              />
              <Line 
                type="monotone" 
                dataKey="severity" 
                stroke="#ef4444" 
                strokeWidth={3}
                dot={{ fill: '#ef4444', r: 4 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Filters and Actions */}
      <div className="glass rounded-xl p-4 flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <Filter className="w-4 h-4 text-text-3" />
            <span className="text-sm text-text-2">Filter:</span>
          </div>
          <Select value={severityFilter} onValueChange={setSeverityFilter}>
            <SelectTrigger className="w-[180px]">
              <SelectValue placeholder="All severities" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Severities</SelectItem>
              <SelectItem value="Severe">🔴 Severe</SelectItem>
              <SelectItem value="Moderate">🟠 Moderate</SelectItem>
              <SelectItem value="Mild">🔵 Mild</SelectItem>
            </SelectContent>
          </Select>
          <div className="px-3 py-1 bg-brand-cyan/10 border border-brand-cyan/30 rounded-lg">
            <span className="text-sm font-mono text-brand-cyan font-bold">
              {filteredIncidents.length} incidents
            </span>
          </div>
        </div>

        <Button onClick={handleExport} variant="outline" className="gap-2">
          <Download className="w-4 h-4" />
          Export CSV
        </Button>
      </div>

      {/* Incidents List */}
      <div className="space-y-3">
        <AnimatePresence>
          {filteredIncidents.length === 0 ? (
            <div className="glass rounded-xl p-12 text-center">
              <div className="text-6xl mb-4">📋</div>
              <p className="text-text-2">No incidents found</p>
              <p className="text-text-3 text-sm mt-2">Adjust filters or wait for new detections</p>
            </div>
          ) : (
            filteredIncidents.map((incident, index) => (
              <motion.div
                key={incident.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                transition={{ delay: index * 0.05 }}
                onClick={() => {
                  setSelectedIncident(incident);
                  setModalOpen(true);
                }}
                className={`glass rounded-xl p-5 border ${getSeverityBg(incident.severity)} hover:border-brand-cyan/50 transition-all cursor-pointer relative overflow-hidden`}
              >
                {/* Severity bar */}
                <div 
                  className={`absolute left-0 top-0 bottom-0 w-1 bg-${getSeverityColor(incident.severity)}`}
                ></div>

                <div className="ml-4 grid grid-cols-1 md:grid-cols-[1fr_auto_auto_auto] gap-4 items-center">
                  <div>
                    <div className="flex items-center gap-3 mb-2">
                      <span className={`px-3 py-1 text-xs font-bold rounded-lg text-${getSeverityColor(incident.severity)} bg-${getSeverityColor(incident.severity)}/20`}>
                        {incident.severity}
                      </span>
                      <span className="text-sm text-text-3 font-mono">ID: {incident.id}</span>
                    </div>
                    {incident.description && (
                      <p className="text-sm text-text-1">{incident.description}</p>
                    )}
                    {incident.location && (
                      <div className="flex items-center gap-1 mt-2 text-text-3">
                        <MapPin className="w-3 h-3" />
                        <span className="text-xs">{incident.location}</span>
                      </div>
                    )}
                  </div>

                  <div className="text-center">
                    <p className="text-xs text-text-3 mb-1">Confidence</p>
                    <p className="text-lg font-bold text-brand-cyan font-mono">
                      {(incident.confidence * 100).toFixed(1)}%
                    </p>
                  </div>

                  <div className="text-center">
                    <p className="text-xs text-text-3 mb-1">Severity Index</p>
                    <p className="text-lg font-bold text-warning font-mono">
                      {incident.severity_index.toFixed(1)}
                    </p>
                  </div>

                  <div className="text-right">
                    <div className="flex items-center gap-1 text-text-3 mb-1">
                      <Calendar className="w-3 h-3" />
                      <p className="text-xs font-mono">
                        {new Date(incident.timestamp).toLocaleDateString()}
                      </p>
                    </div>
                    <p className="text-xs font-mono text-text-3">
                      {new Date(incident.timestamp).toLocaleTimeString()}
                    </p>
                  </div>
                </div>
              </motion.div>
            ))
          )}
        </AnimatePresence>
      </div>

      {/* Detail Modal */}
      <Dialog open={modalOpen} onOpenChange={setModalOpen}>
        <DialogContent className="glass border border-border-subtle max-w-2xl">
          {selectedIncident && (
            <>
              <DialogHeader>
                <DialogTitle className="flex items-center gap-3">
                  <span className="text-2xl">🚨</span>
                  <span>Incident Details</span>
                  <span className={`px-3 py-1 text-xs font-bold rounded-lg text-${getSeverityColor(selectedIncident.severity)} bg-${getSeverityColor(selectedIncident.severity)}/20`}>
                    {selectedIncident.severity}
                  </span>
                </DialogTitle>
              </DialogHeader>

              <div className="space-y-4 mt-4">
                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-bg-elevated/50 rounded-lg p-4 border border-border-subtle">
                    <p className="text-xs text-text-3 mb-1">Incident ID</p>
                    <p className="text-sm font-mono text-text-1">{selectedIncident.id}</p>
                  </div>
                  <div className="bg-bg-elevated/50 rounded-lg p-4 border border-border-subtle">
                    <p className="text-xs text-text-3 mb-1">Timestamp</p>
                    <p className="text-sm font-mono text-text-1">
                      {new Date(selectedIncident.timestamp).toLocaleString()}
                    </p>
                  </div>
                  <div className="bg-bg-elevated/50 rounded-lg p-4 border border-border-subtle">
                    <p className="text-xs text-text-3 mb-1">Confidence</p>
                    <p className="text-lg font-bold text-brand-cyan font-mono">
                      {(selectedIncident.confidence * 100).toFixed(1)}%
                    </p>
                  </div>
                  <div className="bg-bg-elevated/50 rounded-lg p-4 border border-border-subtle">
                    <p className="text-xs text-text-3 mb-1">Severity Index</p>
                    <p className="text-lg font-bold text-warning font-mono">
                      {selectedIncident.severity_index.toFixed(1)}
                    </p>
                  </div>
                </div>

                {selectedIncident.location && (
                  <div className="bg-bg-elevated/50 rounded-lg p-4 border border-border-subtle">
                    <p className="text-xs text-text-3 mb-2">Location</p>
                    <div className="flex items-center gap-2">
                      <MapPin className="w-4 h-4 text-brand-cyan" />
                      <p className="text-sm text-text-1">{selectedIncident.location}</p>
                    </div>
                  </div>
                )}

                {selectedIncident.description && (
                  <div className="bg-bg-elevated/50 rounded-lg p-4 border border-border-subtle">
                    <p className="text-xs text-text-3 mb-2">Description</p>
                    <p className="text-sm text-text-1">{selectedIncident.description}</p>
                  </div>
                )}

                <div className="flex items-center gap-3">
                  <Button 
                    variant="destructive" 
                    onClick={() => handleDelete(selectedIncident.id)}
                    className="flex-1"
                  >
                    Delete Incident
                  </Button>
                  <Button 
                    variant="outline" 
                    onClick={() => setModalOpen(false)}
                    className="flex-1"
                  >
                    Close
                  </Button>
                </div>
              </div>
            </>
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
}