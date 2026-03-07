import { LayoutDashboard, Video, AlertTriangle, Settings, ChevronLeft, ChevronRight } from 'lucide-react';
import { View } from '../App';

interface SidebarProps {
  activeView: View;
  onViewChange: (view: View) => void;
  collapsed: boolean;
  onToggleCollapse: () => void;
}

const navItems = [
  { id: 'dashboard' as View, label: 'Dashboard', icon: LayoutDashboard, emoji: '🎯' },
  { id: 'feeds' as View, label: 'Live Feeds', icon: Video, emoji: '📡' },
  { id: 'incidents' as View, label: 'Incidents', icon: AlertTriangle, emoji: '⚠️' },
  { id: 'settings' as View, label: 'Settings', icon: Settings, emoji: '⚙️' }
];

export default function Sidebar({ activeView, onViewChange, collapsed, onToggleCollapse }: SidebarProps) {
  return (
    <aside 
      className={`glass flex flex-col border-r border-border-subtle transition-all duration-300 ${
        collapsed ? 'w-[72px]' : 'w-[220px]'
      }`}
      style={{ background: 'var(--bg-surface)' }}
    >
      {/* Header */}
      <div className="flex items-center justify-between p-6 border-b border-border-subtle">
        {!collapsed && (
          <div className="flex items-center gap-2">
            <span className="text-2xl">🚨</span>
            <div>
              <h1 className="text-sm font-bold text-text-1">CrashGuard</h1>
              <p className="text-xs text-text-3 font-mono">AI Monitor</p>
            </div>
          </div>
        )}
        {collapsed && <span className="text-2xl mx-auto">🚨</span>}
      </div>

      {/* Navigation */}
      <nav className="flex-1 p-4 space-y-2">
        {navItems.map(item => {
          const Icon = item.icon;
          const isActive = activeView === item.id;
          
          return (
            <button
              key={item.id}
              onClick={() => onViewChange(item.id)}
              className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg transition-all ${
                isActive
                  ? 'bg-brand-cyan/10 text-brand-cyan border border-brand-cyan/30 glow-cyan-sm'
                  : 'text-text-2 hover:text-text-1 hover:bg-bg-elevated'
              }`}
            >
              <span className="text-xl">{item.emoji}</span>
              {!collapsed && (
                <span className="text-sm font-medium">{item.label}</span>
              )}
            </button>
          );
        })}
      </nav>

      {/* Footer */}
      <div className="p-4 border-t border-border-subtle">
        <button
          onClick={onToggleCollapse}
          className="w-full flex items-center justify-center gap-2 px-4 py-2 rounded-lg text-text-3 hover:text-text-1 hover:bg-bg-elevated transition-colors"
        >
          {collapsed ? <ChevronRight className="w-4 h-4" /> : <ChevronLeft className="w-4 h-4" />}
          {!collapsed && <span className="text-xs">Collapse</span>}
        </button>
        {!collapsed && (
          <div className="mt-3 text-center">
            <p className="text-xs text-text-3 font-mono">v2.1.0</p>
          </div>
        )}
      </div>
    </aside>
  );
}
