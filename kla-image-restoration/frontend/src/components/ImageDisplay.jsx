import React, { useEffect, useRef, useState, useCallback } from 'react';

const renderGrayscale = (canvas, data) => {
  if (!canvas || !data) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;
  const imgData = ctx.createImageData(w, h);

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const srcIdx = y * w + x;
      const val = Math.floor(Math.max(0, Math.min(1, data[srcIdx])) * 255);
      const dstIdx = srcIdx * 4;
      imgData.data[dstIdx] = val;
      imgData.data[dstIdx + 1] = val;
      imgData.data[dstIdx + 2] = val;
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
  const n = Math.min(degradedData.length, restoredData.length);

  for (let i = 0; i < n; i++) {
    const diff = Math.abs(degradedData[i] - restoredData[i]);
    const idx = i * 4;
    imgData.data[idx] = Math.floor(diff * 400);
    imgData.data[idx + 1] = Math.floor(diff * 100);
    imgData.data[idx + 2] = Math.floor(255 - diff * 255);
    imgData.data[idx + 3] = 255;
  }
  ctx.putImageData(imgData, 0, 0);
};

export default function ImageDisplay({ inputData, restoredData, isProcessing }) {
  const canvasInput = useRef(null);
  const canvasDiff = useRef(null);
  const canvasRestored = useRef(null);
  const sliderContainer = useRef(null);
  const [divider, setDivider] = useState(50); // % from left
  const [dragging, setDragging] = useState(false);

  useEffect(() => {
    if (inputData && canvasInput.current) {
      renderGrayscale(canvasInput.current, inputData);
    }
  }, [inputData]);

  useEffect(() => {
    if (restoredData && inputData && canvasRestored.current && canvasDiff.current) {
      // Restored canvas may have different dimensions (×2). Resize to match the higher of the two.
      const w = Math.max(restoredData.length, inputData.length) > 0
        ? Math.round(Math.sqrt(Math.max(restoredData.length, inputData.length)))
        : 128;
      if (canvasRestored.current.width !== w) {
        canvasRestored.current.width = w;
        canvasRestored.current.height = w;
      }
      renderGrayscale(canvasRestored.current, restoredData);
      renderDiffMap(canvasDiff.current, inputData, restoredData);
    } else if (!restoredData) {
      if (canvasRestored.current) {
        const ctx = canvasRestored.current.getContext('2d');
        ctx.clearRect(0, 0, canvasRestored.current.width, canvasRestored.current.height);
      }
      if (canvasDiff.current) {
        const ctx = canvasDiff.current.getContext('2d');
        ctx.clearRect(0, 0, canvasDiff.current.width, canvasDiff.current.height);
      }
    }
  }, [restoredData, inputData]);

  const updateDividerFromEvent = useCallback((clientX) => {
    if (!sliderContainer.current) return;
    const rect = sliderContainer.current.getBoundingClientRect();
    const x = Math.max(0, Math.min(clientX - rect.left, rect.width));
    setDivider((x / rect.width) * 100);
  }, []);

  const onMouseDown = (e) => {
    e.preventDefault();
    setDragging(true);
    updateDividerFromEvent(e.clientX);
  };

  useEffect(() => {
    if (!dragging) return;
    const onMove = (e) => updateDividerFromEvent(e.clientX);
    const onUp = () => setDragging(false);
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
    return () => {
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
    };
  }, [dragging, updateDividerFromEvent]);

  const hasBoth = !!(inputData && restoredData);

  return (
    <div className="grid grid-cols-3 gap-stack-md flex-1 min-h-0">
      {/* Input Panel */}
      <div className="flex flex-col bg-surface-container border border-outline-variant rounded">
        <div className="px-2 py-1 border-b border-outline-variant bg-surface-container-high flex justify-between items-center">
          <span className="text-label-xs font-label-xs uppercase text-on-surface-variant">Input (Degraded)</span>
          <span className="text-[10px] font-mono text-outline-variant">FP32</span>
        </div>
        <div className="flex-1 p-2 flex items-center justify-center relative overflow-hidden bg-[#050810]">
          <canvas ref={canvasInput} width={128} height={128} className="w-full h-full object-contain max-h-[300px]" />
        </div>
      </div>

      {/* Diff Panel */}
      <div className="flex flex-col bg-surface-container border border-outline-variant rounded">
        <div className="px-2 py-1 border-b border-outline-variant bg-surface-container-high flex justify-between items-center">
          <span className="text-label-xs font-label-xs uppercase text-on-surface-variant">Difference Map</span>
          <span className="text-[10px] font-mono text-outline-variant">L1_LOSS</span>
        </div>
        <div className="flex-1 p-2 flex items-center justify-center relative overflow-hidden bg-[#050810]">
          <canvas ref={canvasDiff} width={128} height={128} className={`w-full h-full object-contain max-h-[300px] transition-opacity duration-500 ${restoredData ? 'opacity-100' : 'opacity-0'}`} />
        </div>
      </div>

      {/* Restored Panel — with comparison slider overlay */}
      <div className="flex flex-col bg-surface-container border border-outline-variant rounded">
        <div className="px-2 py-1 border-b border-outline-variant bg-surface-container-high flex justify-between items-center">
          <span className="text-label-xs font-label-xs uppercase text-primary">Restored</span>
          <span className="text-[10px] font-mono text-primary-container">
            {hasBoth ? 'DRAG TO COMPARE' : 'SIGMA_OUT'}
          </span>
        </div>
        <div
          ref={sliderContainer}
          className="flex-1 p-2 flex items-center justify-center relative overflow-hidden bg-[#050810] select-none"
          onMouseDown={hasBoth ? onMouseDown : undefined}
          style={{ cursor: hasBoth ? (dragging ? 'col-resize' : 'ew-resize') : 'default' }}
        >
          {isProcessing && <div className="scan-line"></div>}

          {/* Lower layer: Restored (full width) */}
          <canvas
            ref={canvasRestored}
            width={128}
            height={128}
            className={`absolute inset-2 w-[calc(100%-1rem)] h-[calc(100%-1rem)] object-contain max-h-[300px] transition-opacity duration-500 ${restoredData ? 'opacity-100' : 'opacity-0'}`}
          />

          {/* Upper layer: Input, clipped from the right of the divider */}
          {hasBoth && (
            <div
              className="absolute inset-2 max-h-[300px] overflow-hidden"
              style={{ clipPath: `inset(0 ${100 - divider}% 0 0)` }}
            >
              <canvas
                ref={canvasInput}
                width={128}
                height={128}
                className="w-full h-full object-contain"
              />
            </div>
          )}

          {/* Vertical divider line + handle */}
          {hasBoth && (
            <>
              <div
                className="absolute top-0 bottom-0 w-[2px] bg-primary pointer-events-none"
                style={{ left: `calc(${divider}% )` }}
              />
              <div
                className="absolute top-1/2 -translate-y-1/2 w-8 h-8 rounded-full bg-primary text-[#050810] flex items-center justify-center font-bold text-sm shadow-lg pointer-events-none"
                style={{ left: `calc(${divider}% - 1rem)` }}
              >
                ‖
              </div>
            </>
          )}

          {!restoredData && !isProcessing && (
            <div className="text-on-surface-variant text-label-xs uppercase">Run inference to see results</div>
          )}
        </div>
      </div>
    </div>
  );
}