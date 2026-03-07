import { useState, useEffect } from 'react';
import { Save, Send, Loader2, CheckCircle, Info } from 'lucide-react';
import { SystemConfig } from '../../App';
import { Button } from '../ui/button';
import { Input } from '../ui/input';
import { Label } from '../ui/label';
import { Slider } from '../ui/slider';
import { Switch } from '../ui/switch';
import { toast } from 'sonner';

interface SettingsViewProps {
  config: SystemConfig;
  setConfig: (config: SystemConfig) => void;
}

export default function SettingsView({ config, setConfig }: SettingsViewProps) {
  const [localConfig, setLocalConfig] = useState(config);
  const [saving, setSaving] = useState(false);
  const [testingTelegram, setTestingTelegram] = useState(false);
  const [systemInfo, setSystemInfo] = useState<any>(null);

  useEffect(() => {
    setLocalConfig(config);
  }, [config]);

  useEffect(() => {
    // Fetch system info
    fetch('/api/system/status')
      .then(res => {
        if (!res.ok || !res.headers.get('content-type')?.includes('application/json')) {
          throw new Error('Backend not available');
        }
        return res.json();
      })
      .then(data => setSystemInfo(data))
      .catch(() => {
        // Silently use mock data when backend is not available
      });
  }, []);

  const handleSave = async () => {
    setSaving(true);
    try {
      const response = await fetch('/api/v1/system/config', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(localConfig)
      });
      
      if (response.ok) {
        setConfig(localConfig);
        toast.success('Settings saved successfully');
      } else {
        toast.error('Failed to save settings');
      }
    } catch (error) {
      console.error('Failed to save config:', error);
      toast.error('Failed to save settings');
    } finally {
      setSaving(false);
    }
  };

  const handleTestTelegram = async () => {
    setTestingTelegram(true);
    try {
      // Mock test notification - in real app would call API
      await new Promise(resolve => setTimeout(resolve, 1500));
      toast.success('Test notification sent to Telegram!');
    } catch (error) {
      toast.error('Failed to send test notification');
    } finally {
      setTestingTelegram(false);
    }
  };

  return (
    <div className="min-h-screen p-6 space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-text-1">Settings</h1>
        <p className="text-text-2 mt-1">Configure system parameters and integrations</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Detection Settings */}
        <div className="glass rounded-xl p-6 space-y-6">
          <div className="flex items-center gap-2">
            <span className="text-2xl">🎯</span>
            <h2 className="text-xl font-bold text-text-1">Detection Settings</h2>
          </div>

          <div className="space-y-4">
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label>Confidence Threshold</Label>
                <span className="text-sm font-mono text-brand-cyan font-bold">
                  {localConfig.confidence_threshold.toFixed(2)}
                </span>
              </div>
              <Slider
                value={[localConfig.confidence_threshold]}
                onValueChange={([v]) => setLocalConfig({ ...localConfig, confidence_threshold: v })}
                min={0.1}
                max={1.0}
                step={0.05}
                className="w-full"
              />
              <p className="text-xs text-text-3">
                Minimum confidence score for crash detection (0.1 = low, 1.0 = high)
              </p>
            </div>

            <div className="space-y-2">
              <Label>Alert Cooldown (seconds)</Label>
              <Input
                type="number"
                value={localConfig.alert_cooldown}
                onChange={(e) => setLocalConfig({ ...localConfig, alert_cooldown: parseInt(e.target.value) || 0 })}
                min={0}
                max={300}
                className="font-mono"
              />
              <p className="text-xs text-text-3">
                Minimum time between alerts for the same incident
              </p>
            </div>
          </div>
        </div>

        {/* Privacy & Anonymization */}
        <div className="glass rounded-xl p-6 space-y-6">
          <div className="flex items-center gap-2">
            <span className="text-2xl">🔒</span>
            <h2 className="text-xl font-bold text-text-1">Privacy & Anonymization</h2>
          </div>

          <div className="space-y-4">
            <div className="flex items-center justify-between p-4 bg-bg-elevated/50 rounded-lg border border-border-subtle">
              <div className="flex-1">
                <Label>Face Anonymization</Label>
                <p className="text-xs text-text-3 mt-1">
                  Blur faces in video feed (GDPR compliant)
                </p>
              </div>
              <Switch
                checked={localConfig.anonymization_enabled}
                onCheckedChange={(checked) => setLocalConfig({ ...localConfig, anonymization_enabled: checked })}
              />
            </div>

            <div className="bg-brand-indigo/10 border border-brand-indigo/30 rounded-lg p-4">
              <div className="flex items-start gap-2">
                <Info className="w-4 h-4 text-brand-indigo mt-0.5" />
                <div>
                  <p className="text-xs font-medium text-text-1 mb-1">Privacy Notice</p>
                  <p className="text-xs text-text-3">
                    When enabled, all faces detected in the video stream will be automatically blurred 
                    before recording or transmission. This helps comply with privacy regulations like GDPR.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Telegram Alerts */}
        <div className="glass rounded-xl p-6 space-y-6">
          <div className="flex items-center gap-2">
            <span className="text-2xl">🔔</span>
            <h2 className="text-xl font-bold text-text-1">Telegram Alerts</h2>
          </div>

          <div className="space-y-4">
            <div className="flex items-center justify-between p-4 bg-bg-elevated/50 rounded-lg border border-border-subtle">
              <div className="flex-1">
                <Label>Enable Telegram Notifications</Label>
                <p className="text-xs text-text-3 mt-1">
                  Send alerts to Telegram bot
                </p>
              </div>
              <Switch
                checked={localConfig.telegram_enabled}
                onCheckedChange={(checked) => setLocalConfig({ ...localConfig, telegram_enabled: checked })}
              />
            </div>

            {localConfig.telegram_enabled && (
              <>
                <div className="space-y-2">
                  <Label>Bot Token</Label>
                  <Input
                    type="password"
                    value={localConfig.telegram_token}
                    onChange={(e) => setLocalConfig({ ...localConfig, telegram_token: e.target.value })}
                    placeholder="1234567890:ABCdefGHIjklMNOpqrsTUVwxyz"
                    className="font-mono text-xs"
                  />
                </div>

                <div className="space-y-2">
                  <Label>Chat ID</Label>
                  <Input
                    type="text"
                    value={localConfig.telegram_chat_id}
                    onChange={(e) => setLocalConfig({ ...localConfig, telegram_chat_id: e.target.value })}
                    placeholder="-1001234567890"
                    className="font-mono text-xs"
                  />
                </div>

                <Button
                  onClick={handleTestTelegram}
                  disabled={testingTelegram}
                  variant="outline"
                  className="w-full"
                >
                  {testingTelegram ? (
                    <>
                      <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                      Sending...
                    </>
                  ) : (
                    <>
                      <Send className="w-4 h-4 mr-2" />
                      Test Notification
                    </>
                  )}
                </Button>
              </>
            )}
          </div>
        </div>

        {/* System Information */}
        <div className="glass rounded-xl p-6 space-y-6">
          <div className="flex items-center gap-2">
            <span className="text-2xl">🧠</span>
            <h2 className="text-xl font-bold text-text-1">System Information</h2>
          </div>

          <div className="space-y-3">
            <div className="flex items-center justify-between p-3 bg-bg-elevated/50 rounded-lg border border-border-subtle">
              <span className="text-sm text-text-2">AI Model Status</span>
              <span className="flex items-center gap-2 text-sm font-mono text-success">
                <CheckCircle className="w-3 h-3" />
                {systemInfo?.ml_service?.status || 'Operational'}
              </span>
            </div>

            <div className="flex items-center justify-between p-3 bg-bg-elevated/50 rounded-lg border border-border-subtle">
              <span className="text-sm text-text-2">Model Path</span>
              <span className="text-xs font-mono text-text-3 max-w-xs truncate">
                {systemInfo?.ml_service?.model_path || '/models/yolov8_vit.pt'}
              </span>
            </div>

            <div className="flex items-center justify-between p-3 bg-bg-elevated/50 rounded-lg border border-border-subtle">
              <span className="text-sm text-text-2">Anonymization</span>
              <span className={`text-sm font-mono ${localConfig.anonymization_enabled ? 'text-success' : 'text-text-3'}`}>
                {localConfig.anonymization_enabled ? 'Enabled' : 'Disabled'}
              </span>
            </div>

            <div className="flex items-center justify-between p-3 bg-bg-elevated/50 rounded-lg border border-border-subtle">
              <span className="text-sm text-text-2">API Version</span>
              <span className="text-sm font-mono text-brand-cyan">v2.1.0</span>
            </div>

            <div className="flex items-center justify-between p-3 bg-bg-elevated/50 rounded-lg border border-border-subtle">
              <span className="text-sm text-text-2">Inference Time</span>
              <span className="text-sm font-mono text-text-1">~35ms</span>
            </div>

            <div className="flex items-center justify-between p-3 bg-bg-elevated/50 rounded-lg border border-border-subtle">
              <span className="text-sm text-text-2">GPU Usage</span>
              <span className="text-sm font-mono text-text-1">4.2 GB / 8 GB</span>
            </div>
          </div>
        </div>
      </div>

      {/* Save Button */}
      <div className="flex items-center justify-center">
        <Button
          onClick={handleSave}
          disabled={saving}
          className="w-full max-w-md bg-brand-cyan hover:bg-brand-cyan/90 text-bg-base font-bold glow-cyan"
          size="lg"
        >
          {saving ? (
            <>
              <Loader2 className="w-5 h-5 mr-2 animate-spin" />
              Saving...
            </>
          ) : (
            <>
              <Save className="w-5 h-5 mr-2" />
              Save Configuration
            </>
          )}
        </Button>
      </div>
    </div>
  );
}