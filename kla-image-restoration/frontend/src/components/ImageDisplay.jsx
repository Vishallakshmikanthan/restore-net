import React, { useEffect, useRef, useState, useCallback } from 'react';
import { Image as ImageIcon, Sparkles, Layers, Sliders } from 'lucide-react';

const renderGrayscale = (canvas, data) => {
  if (!canvas || !data) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;
  const imgData = ctx.createImageData(w, h);
  const dataLen = data.length;

  // Compute min/max for proper auto-scaling to 0..255
  let min = Infinity, max = -Infinity;
  for (let i = 0; i < dataLen; i++) {
    if (data[i] < min) min = data[i];
    if (data[i] > max) max = data[i];
  }
  const range = max > min ? max - min : 1;

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const srcIdx = y * w + x;
      const rawVal = srcIdx < dataLen ? data[srcIdx] : 0;
      const normalized = Math.max(0, Math.min(1, (rawVal - min) / range));
      const byteVal = Math.floor(normalized * 255);
      const dstIdx = (y * w + x) * 4;
      imgData.data[dstIdx] = byteVal;
      imgData.data[dstIdx + 1] = byteVal;
      imgData.data[dstIdx + 2] = byteVal;
      imgData.data[dstIdx + 3] = 255;
    }
  }
  ctx.putImageData(imgData, 0, 0);
};

const renderDiffMap = (canvas, degradedData, restoredData) => {
  if (!canvas || !degradedData || !restoredData) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;
  const imgData = ctx.createImageData(w, h);
  const n = Math.min(degradedData.length, restoredData.length, w * h);

  for (let i = 0; i < n; i++) {
    const diff = Math.abs(degradedData[i] - restoredData[i]);
    const idx = i * 4;
    // Inferno-style colormap for semiconductor defect visualizer
    const scaled = Math.min(1, diff * 3.0);
    if (scaled < 0.33) {
      imgData.data[idx] = Math.floor(scaled * 3 * 60);
      imgData.data[idx + 1] = 0;
      imgData.data[idx + 2] = Math.floor(scaled * 3 * 220);
    } else if (scaled < 0.66) {
      const t = (scaled - 0.33) * 3;
      imgData.data[idx] = Math.floor(60 + t * 195);
      imgData.data[idx + 1] = Math.floor(t * 120);
      imgData.data[idx + 2] = Math.floor(220 * (1 - t));
    } else {
      const t = (scaled - 0.66) * 3;
      imgData.data[idx] = 255;
      imgData.data[idx + 1] = Math.floor(120 + t * 135);
      imgData.data[idx + 2] = Math.floor(t * 100);
    }
    imgData.data[idx + 3] = 255;
  }
  ctx.putImageData(imgData, 0, 0);
};

export default function ImageDisplay({ inputData, restoredData, isProcessing }) {
  const canvasInput = useRef(null);
  const canvasOutput = useRef(null);
  const canvasDiff = useRef(null);
  const sliderContainer = useRef(null);
  const [divider, setDivider] = useState(50);
  const [dragging, setDragging] = useState(false);
  const [resolution, setResolution] = useState(128);

  useEffect(() => {
    if (inputData) {
      const size = Math.floor(Math.sqrt(inputData.length)) || 128;
      setResolution(size);
      if (canvasInput.current) {
        canvasInput.current.width = size;
        canvasInput.current.height = size;
        renderGrayscale(canvasInput.current, inputData);
      }
    }
  }, [inputData]);

  useEffect(() => {
    if (restoredData && inputData) {
      const size = Math.floor(Math.sqrt(restoredData.length)) || 128;
      if (canvasOutput.current) {
        canvasOutput.current.width = size;
        canvasOutput.current.height = size;
        renderGrayscale(canvasOutput.current, restoredData);
      }
      if (canvasDiff.current) {
        canvasDiff.current.width = size;
        canvasDiff.current.height = size;
        renderDiffMap(canvasDiff.current, inputData, restoredData);
      }
    }
  }, [restoredData, inputData]);

  const updateDivider = useCallback((clientX) => {
    if (!sliderContainer.current) return;
    const rect = sliderContainer.current.getBoundingClientRect();
    const x = Math.max(0, Math.min(clientX - rect.left, rect.width));
    setDivider((x / rect.width) * 100);
  }, []);

  const onMouseDown = (e) => {
    e.preventDefault();
    setDragging(true);
    updateDivider(e.clientX);
  };

  useEffect(() => {
    if (!dragging) return;
    const onMove = (e) => updateDivider(e.clientX);
    const onUp = () => setDragging(false);
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
    return () => {
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
    };
  }, [dragging, updateDivider]);

  const hasData = !!inputData;
  const hasRestored = !!restoredData;

  return (
    <div className={`grid grid-cols-1 md:grid-cols-3 gap-gutter min-h-[320px] flex-1 ${isProcessing ? 'processing' : ''}`}>
      {/* Panel 1: Input // NoisyLR */}
      <div className="bg-layer-top border border-border-subtle rounded-clinical flex flex-col relative overflow-hidden group">
        <div className="p-2.5 border-b border-border-subtle flex justify-between items-center bg-layer-mid z-10">
          <div className="flex items-center gap-1.5">
            <span className="text-[11px] font-mono font-bold text-on-surface">INPUT // NOISYLR</span>
          </div>
          <span className="text-[10px] font-mono text-on-surface-variant flex items-center gap-1">
            <ImageIcon size={12} />
            {hasData ? `${resolution}x${resolution}` : 'NOISY'}
          </span>
        </div>
        <div className="flex-1 relative bg-[#000000] flex items-center justify-center p-3">
          <canvas
            ref={canvasInput}
            width={128}
            height={128}
            className={`w-full max-w-[280px] aspect-square object-contain pixelated border border-border-subtle/40 transition-opacity duration-300 ${
              hasData ? 'opacity-100' : 'opacity-0'
            }`}
          />
          <div className="scanline"></div>
          {!hasData && (
            <div className="absolute inset-0 flex items-center justify-center text-on-surface-variant/40 text-[13px] font-mono font-bold">
              NO DATA LOADED
            </div>
          )}
        </div>
      </div>

      {/* Panel 2: Restored // Output (with comparison divider) */}
      <div className="bg-layer-top border border-border-subtle rounded-clinical flex flex-col relative overflow-hidden">
        <div className="p-2.5 border-b border-border-subtle flex justify-between items-center bg-layer-mid z-10">
          <div className="flex items-center gap-1.5">
            <Sparkles size={13} className="text-accent-cyan" />
            <span className="text-[11px] font-mono font-bold text-accent-cyan">RESTORED // OUTPUT</span>
          </div>
          <span className="text-[10px] font-mono text-accent-cyan/80">
            {hasRestored ? (hasData ? 'DRAG SLIDER TO COMPARE' : 'SIGMA_OUT') : 'SIGMA_OUT'}
          </span>
        </div>
        <div
          ref={sliderContainer}
          onMouseDown={hasRestored && hasData ? onMouseDown : undefined}
          className={`flex-1 relative bg-[#000000] flex items-center justify-center p-3 select-none ${
            hasRestored && hasData ? (dragging ? 'cursor-col-resize' : 'cursor-ew-resize') : ''
          }`}
        >
          <canvas
            ref={canvasOutput}
            width={128}
            height={128}
            className={`w-full max-w-[280px] aspect-square object-contain pixelated border border-border-subtle/40 transition-opacity duration-500 ${
              hasRestored ? 'opacity-100' : 'opacity-0'
            }`}
          />
          <div className="scanline"></div>

          {/* Interactive Split-Comparison Overlay */}
          {hasRestored && hasData && (
            <>
              {/* Overlay Input Image clipped */}
              <div
                className="absolute inset-3 max-w-[280px] aspect-square mx-auto pointer-events-none overflow-hidden"
                style={{ clipPath: `inset(0 ${100 - divider}% 0 0)` }}
              >
                <canvas
                  ref={(node) => {
                    if (node && inputData) {
                      node.width = resolution;
                      node.height = resolution;
                      renderGrayscale(node, inputData);
                    }
                  }}
                  className="w-full h-full object-contain pixelated border border-primary/60"
                />
              </div>

              {/* Divider Handle */}
              <div
                className="absolute top-3 bottom-3 w-[2px] bg-accent-cyan pointer-events-none shadow-[0_0_8px_#00e5ff]"
                style={{ left: `calc(50% - 140px + (280px * ${divider / 100}))` }}
              />
              <div
                className="absolute top-1/2 -translate-y-1/2 w-6 h-6 rounded-full bg-accent-cyan text-layer-base flex items-center justify-center font-mono font-bold text-[10px] shadow-[0_0_10px_#00e5ff] pointer-events-none"
                style={{ left: `calc(50% - 140px + (280px * ${divider / 100}) - 12px)` }}
              >
                ↔
              </div>
            </>
          )}

          {!hasRestored && !isProcessing && (
            <div className="absolute inset-0 flex items-center justify-center text-on-surface-variant/40 text-[13px] font-mono font-bold">
              AWAITING INFERENCE
            </div>
          )}
        </div>
      </div>

      {/* Panel 3: Residual // Δ Map */}
      <div className="bg-layer-top border border-border-subtle rounded-clinical flex flex-col relative overflow-hidden">
        <div className="p-2.5 border-b border-border-subtle flex justify-between items-center bg-layer-mid z-10">
          <div className="flex items-center gap-1.5">
            <span className="text-[11px] font-mono font-bold text-on-surface">RESIDUAL // Δ MAP</span>
          </div>
          <span className="text-[10px] font-mono text-on-surface-variant flex items-center gap-1">
            <Layers size={12} />
            L1 HEATMAP
          </span>
        </div>
        <div className="flex-1 relative bg-[#000000] flex items-center justify-center p-3">
          <canvas
            ref={canvasDiff}
            width={128}
            height={128}
            className={`w-full max-w-[280px] aspect-square object-contain pixelated border border-border-subtle/40 transition-opacity duration-500 ${
              hasRestored ? 'opacity-100' : 'opacity-0'
            }`}
          />
          <div className="scanline"></div>
          {!hasRestored && !isProcessing && (
            <div className="absolute inset-0 flex items-center justify-center text-on-surface-variant/40 text-[13px] font-mono font-bold">
              AWAITING INFERENCE
            </div>
          )}
        </div>
      </div>
    </div>
  );
}