import React, { useEffect, useState } from 'react';

export default function WaveformTrace({ isProcessing }) {
  const [wavePath, setWavePath] = useState('');

  useEffect(() => {
    if (isProcessing) {
      const interval = setInterval(() => {
        let d = 'M0,30 ';
        for (let i = 20; i <= 1000; i += 20) {
          const y = 30 + (Math.random() - 0.5) * 45;
          d += `L${i},${Math.max(5, Math.min(55, y.toFixed(1)))} `;
        }
        setWavePath(d);
      }, 100);
      return () => clearInterval(interval);
    } else {
      setWavePath('M0,30 L100,30 L120,10 L140,50 L160,30 L300,30 L320,15 L340,45 L360,30 L600,30 L620,12 L640,48 L660,30 L1000,30');
    }
  }, [isProcessing]);

  return (
    <div className="h-[56px] w-full border border-border-subtle rounded-clinical bg-layer-top flex items-center overflow-hidden relative shrink-0">
      <div className="absolute left-4 top-1/2 -translate-y-1/2 text-[10px] font-mono font-bold text-on-surface-variant z-10 bg-layer-top/90 px-2 py-0.5 rounded border border-border-subtle/50 flex items-center gap-1.5">
        <span className={`w-1.5 h-1.5 rounded-full ${isProcessing ? 'bg-accent-cyan animate-ping' : 'bg-outline'}`}></span>
        SYS_TRACE // {isProcessing ? 'STREAM_ACTIVE' : 'IDLE'}
      </div>
      <svg className="w-full h-full" preserveAspectRatio="none" viewBox="0 0 1000 60">
        <path
          className={`waveform transition-all duration-300 ${isProcessing ? 'opacity-90 stroke-accent-cyan' : 'opacity-40 stroke-outline'}`}
          d={wavePath}
          fill="none"
          strokeWidth={isProcessing ? 2 : 1.5}
        />
      </svg>
    </div>
  );
}
