import { useState, useEffect } from 'react';
import { Toaster } from './components/ui/sonner';
import Sidebar from './components/Sidebar';
import DashboardView from './components/views/DashboardView';
import LiveFeedsView from './components/views/LiveFeedsView';
import IncidentsView from './components/views/IncidentsView';
import SettingsView from './components/views/SettingsView';

export type View = 'dashboard' | 'feeds' | 'incidents' | 'settings';

export interface SystemConfig {
  confidence_threshold: number;
  anonymization_enabled: boolean;
  alert_cooldown: number;
  telegram_enabled: boolean;
  telegram_token: string;
  telegram_chat_id: string;
}

export default function App() {
  const [activeView, setActiveView] = useState<View>('dashboard');
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [config, setConfig] = useState<SystemConfig>({
    confidence_threshold: 0.6,
    anonymization_enabled: true,
    alert_cooldown: 30,
    telegram_enabled: false,
    telegram_token: '',
    telegram_chat_id: ''
  });

  // Load config from API on mount
  useEffect(() => {
    fetch('/api/v1/system/config')
      .then(res => {
        if (!res.ok || !res.headers.get('content-type')?.includes('application/json')) {
          throw new Error('Backend not available');
        }
        return res.json();
      })
      .then(data => setConfig(data))
      .catch(() => {
        // Silently use default config when backend is not available
      });
  }, []);

  const renderView = () => {
    switch (activeView) {
      case 'dashboard':
        return <DashboardView config={config} />;
      case 'feeds':
        return <LiveFeedsView config={config} />;
      case 'incidents':
        return <IncidentsView />;
      case 'settings':
        return <SettingsView config={config} setConfig={setConfig} />;
      default:
        return <DashboardView config={config} />;
    }
  };

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-bg-base">
      <Sidebar
        activeView={activeView}
        onViewChange={setActiveView}
        collapsed={sidebarCollapsed}
        onToggleCollapse={() => setSidebarCollapsed(!sidebarCollapsed)}
      />
      <main className="flex-1 overflow-auto">
        {renderView()}
      </main>
      <Toaster position="top-right" theme="dark" />
    </div>
  );
}