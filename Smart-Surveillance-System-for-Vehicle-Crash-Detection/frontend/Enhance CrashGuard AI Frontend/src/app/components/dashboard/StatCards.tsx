import { TrendingUp, Video, AlertTriangle, Brain } from 'lucide-react';
import { motion } from 'motion/react';
import { useEffect, useState } from 'react';

interface StatsProps {
  stats: {
    model_accuracy: number;
    active_feeds: number;
    incidents_today: number;
    ai_model: string;
  };
}

function AnimatedCounter({ value, suffix = '' }: { value: number; suffix?: string }) {
  const [count, setCount] = useState(0);

  useEffect(() => {
    const duration = 1000;
    const steps = 50;
    const stepValue = value / steps;
    let current = 0;

    const timer = setInterval(() => {
      current += stepValue;
      if (current >= value) {
        setCount(value);
        clearInterval(timer);
      } else {
        setCount(current);
      }
    }, duration / steps);

    return () => clearInterval(timer);
  }, [value]);

  return (
    <span>
      {suffix === '%' ? count.toFixed(1) : Math.floor(count)}
      {suffix}
    </span>
  );
}

export default function StatCards({ stats }: StatsProps) {
  const cards = [
    {
      label: 'Model Accuracy',
      value: stats.model_accuracy,
      suffix: '%',
      icon: TrendingUp,
      color: 'success',
      emoji: '🎯'
    },
    {
      label: 'Active Feeds',
      value: stats.active_feeds,
      suffix: '',
      icon: Video,
      color: 'brand-cyan',
      emoji: '📡'
    },
    {
      label: 'Incidents Today',
      value: stats.incidents_today,
      suffix: '',
      icon: AlertTriangle,
      color: 'warning',
      emoji: '⚠️'
    },
    {
      label: 'AI Model',
      value: stats.ai_model,
      icon: Brain,
      color: 'brand-indigo',
      emoji: '🧠',
      isText: true
    }
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
      {cards.map((card, index) => {
        const Icon = card.icon;
        return (
          <motion.div
            key={card.label}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
            className="glass rounded-xl p-6 hover:border-brand-cyan/30 transition-all cursor-pointer group"
          >
            <div className="flex items-start justify-between mb-4">
              <div className="text-3xl">{card.emoji}</div>
              <Icon className={`w-5 h-5 text-${card.color} opacity-70 group-hover:opacity-100 transition-opacity`} />
            </div>
            <p className="text-text-3 text-sm mb-2">{card.label}</p>
            <p className="text-2xl font-bold text-text-1">
              {card.isText ? (
                <span className="text-lg">{card.value}</span>
              ) : (
                <AnimatedCounter value={card.value as number} suffix={card.suffix} />
              )}
            </p>
          </motion.div>
        );
      })}
    </div>
  );
}
