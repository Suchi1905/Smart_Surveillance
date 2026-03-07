import { useState } from 'react';
import { Upload, Webcam as WebcamIcon, Link as LinkIcon, Play, Square, Settings as SettingsIcon, Eye, Tag, Percent } from 'lucide-react';
import { SystemConfig } from '../../App';
import { Slider } from '../ui/slider';
import { Button } from '../ui/button';
import { Input } from '../ui/input';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '../ui/tabs';
import { Switch } from '../ui/switch';
import { Label } from '../ui/label';

interface DetectionAreaProps {
  config: SystemConfig;
}

export default function DetectionArea({ config }: DetectionAreaProps) {
  const [sourceType, setSourceType] = useState<'file' | 'webcam' | 'url'>('webcam');
  const [isStreaming, setIsStreaming] = useState(false);
  const [confidence, setConfidence] = useState(config.confidence_threshold);
  const [streamUrl, setStreamUrl] = useState('');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  
  // Overlay controls
  const [showBoundingBoxes, setShowBoundingBoxes] = useState(true);
  const [showLabels, setShowLabels] = useState(true);
  const [showConfidence, setShowConfidence] = useState(true);
  const [detectionCount, setDetectionCount] = useState(0);

  const handleStartStream = async () => {
    try {
      let endpoint = '';
      if (sourceType === 'webcam') {
        endpoint = `/video?source=webcam&conf=${confidence}`;
      } else if (sourceType === 'file' && selectedFile) {
        const formData = new FormData();
        formData.append('file', selectedFile);
        await fetch('/video/upload', {
          method: 'POST',
          body: formData
        });
        endpoint = `/video/file?filename=${selectedFile.name}&conf=${confidence}`;
      } else if (sourceType === 'url' && streamUrl) {
        endpoint = `/video/url?url=${encodeURIComponent(streamUrl)}&conf=${confidence}`;
      }
      
      setIsStreaming(true);
    } catch (error) {
      console.error('Failed to start stream:', error);
    }
  };

  const handleStopStream = async () => {
    try {
      await fetch('/video/stop', { method: 'POST' });
      setIsStreaming(false);
    } catch (error) {
      console.error('Failed to stop stream:', error);
    }
  };

  const getStreamUrl = () => {
    if (!isStreaming) return '';
    if (sourceType === 'webcam') {
      return `/video?source=webcam&conf=${confidence}`;
    } else if (sourceType === 'file' && selectedFile) {
      return `/video/file?filename=${selectedFile.name}&conf=${confidence}`;
    } else if (sourceType === 'url' && streamUrl) {
      return `/video/url?url=${encodeURIComponent(streamUrl)}&conf=${confidence}`;
    }
    return '';
  };

  return (
    <div className="glass rounded-xl p-6">
      <div className="flex items-center gap-2 mb-6">
        <span className="text-2xl">📹</span>
        <h2 className="text-xl font-bold text-text-1">Detection Feed</h2>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-[320px_1fr] gap-6">
        {/* Control Panel */}
        <div className="space-y-6">
          {/* Source Tabs */}
          <Tabs value={sourceType} onValueChange={(v) => setSourceType(v as any)}>
            <TabsList className="grid grid-cols-3 w-full">
              <TabsTrigger value="file" className="text-xs">
                <Upload className="w-3 h-3 mr-1" />
                File
              </TabsTrigger>
              <TabsTrigger value="webcam" className="text-xs">
                <WebcamIcon className="w-3 h-3 mr-1" />
                Webcam
              </TabsTrigger>
              <TabsTrigger value="url" className="text-xs">
                <LinkIcon className="w-3 h-3 mr-1" />
                URL
              </TabsTrigger>
            </TabsList>

            <TabsContent value="file" className="mt-4">
              <div className="border-2 border-dashed border-border-muted rounded-lg p-6 text-center bg-bg-elevated/30 hover:border-brand-cyan/50 transition-colors cursor-pointer">
                <input
                  type="file"
                  accept="video/*"
                  onChange={(e) => setSelectedFile(e.target.files?.[0] || null)}
                  className="hidden"
                  id="video-upload"
                />
                <label htmlFor="video-upload" className="cursor-pointer">
                  <Upload className="w-8 h-8 mx-auto mb-2 text-text-3" />
                  <p className="text-sm text-text-2">Drop video file or click</p>
                  {selectedFile && (
                    <p className="text-xs text-brand-cyan mt-2 font-mono">{selectedFile.name}</p>
                  )}
                </label>
              </div>
            </TabsContent>

            <TabsContent value="webcam" className="mt-4">
              <div className="bg-bg-elevated/30 rounded-lg p-4 text-center border border-border-subtle">
                <WebcamIcon className="w-12 h-12 mx-auto mb-2 text-brand-cyan" />
                <p className="text-sm text-text-2">Default camera ready</p>
              </div>
            </TabsContent>

            <TabsContent value="url" className="mt-4 space-y-3">
              <Input
                placeholder="YouTube URL / RTSP stream"
                value={streamUrl}
                onChange={(e) => setStreamUrl(e.target.value)}
                className="font-mono text-xs"
              />
              <p className="text-xs text-text-3">Supports YouTube, RTSP, HTTP streams</p>
            </TabsContent>
          </Tabs>

          {/* Confidence Slider */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <Label className="text-sm text-text-2">Confidence Threshold</Label>
              <span className="text-sm font-mono text-brand-cyan font-bold">{confidence.toFixed(2)}</span>
            </div>
            <Slider
              value={[confidence]}
              onValueChange={([v]) => setConfidence(v)}
              min={0.1}
              max={1.0}
              step={0.05}
              className="w-full"
            />
          </div>

          {/* Overlay Controls */}
          <div className="space-y-3 border-t border-border-subtle pt-4">
            <div className="flex items-center gap-2 mb-2">
              <SettingsIcon className="w-4 h-4 text-text-3" />
              <span className="text-sm font-medium text-text-2">Display Overlays</span>
            </div>
            
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Eye className="w-3 h-3 text-text-3" />
                  <Label className="text-xs">Bounding Boxes</Label>
                </div>
                <Switch checked={showBoundingBoxes} onCheckedChange={setShowBoundingBoxes} />
              </div>
              
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Tag className="w-3 h-3 text-text-3" />
                  <Label className="text-xs">Labels</Label>
                </div>
                <Switch checked={showLabels} onCheckedChange={setShowLabels} />
              </div>
              
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Percent className="w-3 h-3 text-text-3" />
                  <Label className="text-xs">Confidence Scores</Label>
                </div>
                <Switch checked={showConfidence} onCheckedChange={setShowConfidence} />
              </div>
            </div>
          </div>

          {/* Control Buttons */}
          <div className="space-y-2">
            {!isStreaming ? (
              <Button
                onClick={handleStartStream}
                className="w-full bg-brand-cyan hover:bg-brand-cyan/90 text-bg-base font-bold glow-cyan"
              >
                <Play className="w-4 h-4 mr-2" />
                Start Detection
              </Button>
            ) : (
              <Button
                onClick={handleStopStream}
                variant="destructive"
                className="w-full"
              >
                <Square className="w-4 h-4 mr-2" />
                Stop Stream
              </Button>
            )}
          </div>

          {/* Status Indicator */}
          <div className={`rounded-lg p-3 text-center border ${
            isStreaming
              ? 'bg-success/10 border-success/30 text-success'
              : 'bg-text-3/10 border-text-3/30 text-text-3'
          }`}>
            <div className="flex items-center justify-center gap-2">
              <div className={`w-2 h-2 rounded-full ${isStreaming ? 'bg-success animate-pulse' : 'bg-text-3'}`}></div>
              <span className="text-xs font-mono uppercase tracking-wide">
                {isStreaming ? 'Streaming' : 'Idle'}
              </span>
            </div>
          </div>
        </div>

        {/* Video Feed */}
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <h3 className="text-lg font-bold text-text-1">Live Feed</h3>
              {isStreaming && (
                <span className="px-2 py-1 bg-danger text-xs font-bold text-white rounded pulse-live">
                  LIVE
                </span>
              )}
            </div>
            {isStreaming && (
              <div className="text-sm text-text-2 font-mono">
                Detections: <span className="text-brand-cyan font-bold">{detectionCount}</span>
              </div>
            )}
          </div>

          <div className="relative bg-bg-elevated rounded-xl overflow-hidden border border-border-subtle aspect-video scanlines">
            {isStreaming ? (
              <img
                src={getStreamUrl()}
                alt="Live detection feed"
                className="w-full h-full object-contain"
              />
            ) : (
              <div className="absolute inset-0 flex flex-col items-center justify-center">
                <div className="text-6xl mb-4">📹</div>
                <p className="text-text-2">No active feed</p>
                <p className="text-text-3 text-sm mt-2">Select a source and start detection</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
