import React, { useEffect, useState } from 'react';

const useAnimatedValue = (targetValue, duration = 800) => {
  const [value, setValue] = useState(0);

  useEffect(() => {
    if (targetValue === 0) {
      setValue(0);
      return;
    }

    let startTimestamp = null;
    const startValue = value;

    const step = (timestamp) => {
      if (!startTimestamp) startTimestamp = timestamp;
      const progress = Math.min((timestamp - startTimestamp) / duration, 1);
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

const AnimatedMetric = ({ value, decimals = 2, active = false }) => {
  const animated = useAnimatedValue(active ? value : 0);
  if (!active) return <span>--</span>;
  return <span>{animated.toFixed(decimals)}</span>;
};

export default function MetricsBar({ metrics, hasResults = false }) {
  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-gutter shrink-0">
      {/* PSNR */}
      <div className="bg-layer-top border border-border-subtle rounded-clinical p-3.5 flex flex-col justify-between">
        <span className="text-[11px] font-mono text-on-surface-variant font-medium uppercase mb-1">
          PSNR (dB)
        </span>
        <span className={`text-[28px] font-mono font-bold transition-all ${hasResults ? 'text-on-surface' : 'text-on-surface/20'}`}>
          <AnimatedMetric value={metrics.psnr} decimals={2} active={hasResults} />
        </span>
      </div>

      {/* SSIM */}
      <div className="bg-layer-top border border-border-subtle rounded-clinical p-3.5 flex flex-col justify-between">
        <span className="text-[11px] font-mono text-on-surface-variant font-medium uppercase mb-1">
          SSIM
        </span>
        <span className={`text-[28px] font-mono font-bold transition-all ${hasResults ? 'text-on-surface' : 'text-on-surface/20'}`}>
          <AnimatedMetric value={metrics.ssim} decimals={3} active={hasResults} />
        </span>
      </div>

      {/* LPIPS */}
      <div className="bg-layer-top border border-border-subtle rounded-clinical p-3.5 flex flex-col justify-between">
        <span className="text-[11px] font-mono text-on-surface-variant font-medium uppercase mb-1">
          LPIPS
        </span>
        <span className={`text-[28px] font-mono font-bold transition-all ${hasResults ? 'text-on-surface' : 'text-on-surface/20'}`}>
          <AnimatedMetric value={metrics.lpips} decimals={3} active={hasResults} />
        </span>
      </div>

      {/* E2E RUNTIME */}
      <div className="bg-layer-top border border-border-subtle border-l-2 border-l-accent-cyan rounded-clinical p-3.5 flex flex-col justify-between shadow-[0_0_15px_rgba(0,229,255,0.05)]">
        <span className="text-[11px] font-mono text-accent-cyan font-bold uppercase mb-1">
          E2E RUNTIME
        </span>
        <div className="flex items-baseline gap-1">
          <span className={`text-[28px] font-mono font-bold transition-all ${hasResults ? 'text-accent-cyan' : 'text-accent-cyan/20'}`}>
            <AnimatedMetric value={metrics.latency} decimals={1} active={hasResults} />
          </span>
          <span className={`text-[12px] font-mono ${hasResults ? 'text-accent-cyan/80' : 'text-accent-cyan/20'}`}>
            ms
          </span>
        </div>
      </div>
    </div>
  );
}
