import { SystemConfig } from '../../App';
import DetectionArea from '../dashboard/DetectionArea';

interface LiveFeedsViewProps {
  config: SystemConfig;
}

export default function LiveFeedsView({ config }: LiveFeedsViewProps) {
  return (
    <div className="min-h-screen p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-text-1">Live Feeds</h1>
          <p className="text-text-2 mt-1">Real-time video stream monitoring</p>
        </div>
        <div className="flex items-center gap-2 glass px-4 py-2 rounded-lg">
          <div className="w-2 h-2 rounded-full bg-danger pulse-live"></div>
          <span className="text-sm text-text-2 font-mono uppercase tracking-wide">LIVE</span>
        </div>
      </div>

      <DetectionArea config={config} />
    </div>
  );
}
