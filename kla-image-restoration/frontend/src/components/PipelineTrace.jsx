import React from 'react';

export default function PipelineTrace({ isComplete, runtimeMs = 38 }) {
  return (
    <div className="bg-layer-top border border-border-subtle rounded-clinical p-3.5 mt-auto shrink-0">
      <div className="flex justify-between items-center mb-2">
        <span className="text-[11px] font-mono uppercase text-on-surface-variant font-medium">
          PIPELINE EXECUTION PROFILE
        </span>
        <span className="text-[11px] font-mono text-accent-cyan font-semibold">
          TOTAL: {isComplete ? `${runtimeMs.toFixed(1)}ms` : '--'}
        </span>
      </div>

      {/* Stacked Execution Bar */}
      <div className={`w-full h-3.5 bg-layer-base rounded overflow-hidden flex transition-all duration-700 ${isComplete ? 'opacity-100' : 'opacity-25'}`}>
        <div className="h-full bg-outline-variant transition-all duration-700" style={{ width: isComplete ? '10%' : '0%' }} title="Disk I/O"></div>
        <div className="h-full bg-primary-fixed-dim/60 transition-all duration-700" style={{ width: isComplete ? '15%' : '0%' }} title="Preprocess"></div>
        <div className="h-full bg-primary-fixed-dim/40 transition-all duration-700" style={{ width: isComplete ? '5%' : '0%' }} title="CPU->GPU Memory Transfer"></div>
        <div className="h-full bg-accent-cyan transition-all duration-700 shadow-[0_0_8px_rgba(0,229,255,0.5)]" style={{ width: isComplete ? '50%' : '0%' }} title="Neural Inference"></div>
        <div className="h-full bg-primary-fixed-dim/40 transition-all duration-700" style={{ width: isComplete ? '5%' : '0%' }} title="GPU->CPU Memory Transfer"></div>
        <div className="h-full bg-primary-fixed-dim/80 transition-all duration-700" style={{ width: isComplete ? '15%' : '0%' }} title="Postprocess & Normalization"></div>
      </div>

      {/* Legend */}
      <div className="flex gap-4 mt-2.5 flex-wrap">
        <div className="flex items-center gap-1.5">
          <div className="w-2 h-2 bg-outline-variant rounded-[1px]"></div>
          <span className="text-[10px] font-mono text-on-surface-variant">I/O (10%)</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-2 h-2 bg-primary-fixed-dim/60 rounded-[1px]"></div>
          <span className="text-[10px] font-mono text-on-surface-variant">PRE (15%)</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-2 h-2 bg-primary-fixed-dim/40 rounded-[1px]"></div>
          <span className="text-[10px] font-mono text-on-surface-variant">MEM_TX (10%)</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-2 h-2 bg-accent-cyan rounded-[1px]"></div>
          <span className="text-[10px] font-mono text-accent-cyan font-bold">INFER (50%)</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-2 h-2 bg-primary-fixed-dim/80 rounded-[1px]"></div>
          <span className="text-[10px] font-mono text-on-surface-variant">POST (15%)</span>
        </div>
      </div>
    </div>
  );
}
