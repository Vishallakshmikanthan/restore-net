import React, { useEffect, useState } from 'react';

// A simple hook to animate numbers
const useAnimatedValue = (targetValue, duration = 1000) => {
  const [value, setValue] = useState(0);

  useEffect(() => {
    let startTimestamp = null;
    let startValue = value;

    const step = (timestamp) => {
      if (!startTimestamp) startTimestamp = timestamp;
      const progress = Math.min((timestamp - startTimestamp) / duration, 1);
      
      // easeOutExpo
      const easeProgress = progress === 1 ? 1 : 1 - Math.pow(2, -10 * progress);
      
      setValue(startValue + (targetValue - startValue) * easeProgress);

      if (progress < 1) {
        window.requestAnimationFrame(step);
      }
    };

    window.requestAnimationFrame(step);
  }, [targetValue, duration]);

  return value;
};

const AnimatedMetric = ({ value, decimals = 2 }) => {
  const animated = useAnimatedValue(value);
  return <span>{animated.toFixed(decimals)}</span>;
};

export default function MetricsBar({ metrics }) {
  return (
    <div className="grid grid-cols-4 gap-stack-md shrink-0">
      <div className="bg-surface-container border border-outline-variant p-3 flex flex-col justify-between rounded">
        <span className="text-label-xs font-label-xs uppercase text-on-surface-variant">PSNR (dB)</span>
        <span className="text-metric-lg font-metric-lg text-on-surface text-right mt-2">
          <AnimatedMetric value={metrics.psnr} decimals={2} />
        </span>
      </div>
      <div className="bg-surface-container border border-outline-variant p-3 flex flex-col justify-between rounded">
        <span className="text-label-xs font-label-xs uppercase text-on-surface-variant">SSIM</span>
        <span className="text-metric-lg font-metric-lg text-on-surface text-right mt-2">
          <AnimatedMetric value={metrics.ssim} decimals={3} />
        </span>
      </div>
      <div className="bg-surface-container border border-outline-variant p-3 flex flex-col justify-between rounded">
        <span className="text-label-xs font-label-xs uppercase text-on-surface-variant">LPIPS</span>
        <span className="text-metric-lg font-metric-lg text-primary text-right mt-2">
          <AnimatedMetric value={metrics.lpips} decimals={3} />
        </span>
      </div>
      <div className="bg-surface-container border border-outline-variant p-3 flex flex-col justify-between rounded">
        <span className="text-label-xs font-label-xs uppercase text-on-surface-variant">Latency (ms)</span>
        <span className="text-metric-lg font-metric-lg text-primary-container text-right mt-2">
          <AnimatedMetric value={metrics.latency} decimals={1} />
        </span>
      </div>
    </div>
  );
}
