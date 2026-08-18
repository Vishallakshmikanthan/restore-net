import React, { useEffect, useRef } from 'react';

const renderGrayscale = (canvas, data) => {
  if (!canvas || !data) return;
  const ctx = canvas.getContext('2d');
  const imgData = ctx.createImageData(128, 128);
  
  for(let i=0; i<data.length; i++) {
      const val = Math.floor(Math.max(0, Math.min(1, data[i])) * 255);
      const idx = i * 4;
      imgData.data[idx] = val;     // R
      imgData.data[idx+1] = val;   // G
      imgData.data[idx+2] = val;   // B
      imgData.data[idx+3] = 255;   // A
  }
  ctx.putImageData(imgData, 0, 0);
};

const renderDiffMap = (canvas, degradedData, restoredData) => {
  if (!canvas || !degradedData || !restoredData) return;
  const ctx = canvas.getContext('2d');
  const imgData = ctx.createImageData(128, 128);
  
  for(let i=0; i<degradedData.length; i++) {
      const diff = Math.abs(degradedData[i] - restoredData[i]);
      const idx = i * 4;
      imgData.data[idx] = Math.floor(diff * 400);       // R
      imgData.data[idx+1] = Math.floor(diff * 100);     // G
      imgData.data[idx+2] = Math.floor(255 - diff*255); // B
      imgData.data[idx+3] = 255;                        // A
  }
  ctx.putImageData(imgData, 0, 0);
};

export default function ImageDisplay({ inputData, restoredData, isProcessing }) {
  const canvasInput = useRef(null);
  const canvasDiff = useRef(null);
  const canvasRestored = useRef(null);

  useEffect(() => {
    if (inputData && canvasInput.current) {
      renderGrayscale(canvasInput.current, inputData);
    }
  }, [inputData]);

  useEffect(() => {
    if (restoredData && inputData && canvasRestored.current && canvasDiff.current) {
      renderGrayscale(canvasRestored.current, restoredData);
      renderDiffMap(canvasDiff.current, inputData, restoredData);
    } else if (!restoredData) {
      // clear output canvases
      if(canvasRestored.current) {
          const ctx = canvasRestored.current.getContext('2d');
          ctx.clearRect(0, 0, 128, 128);
      }
      if(canvasDiff.current) {
          const ctx = canvasDiff.current.getContext('2d');
          ctx.clearRect(0, 0, 128, 128);
      }
    }
  }, [restoredData, inputData]);

  return (
    <div className="grid grid-cols-3 gap-stack-md flex-1 min-h-0">
      {/* Input Panel */}
      <div className="flex flex-col bg-surface-container border border-outline-variant rounded">
        <div className="px-2 py-1 border-b border-outline-variant bg-surface-container-high flex justify-between items-center">
          <span className="text-label-xs font-label-xs uppercase text-on-surface-variant">Input (Degraded)</span>
          <span className="text-[10px] font-mono text-outline-variant">FP32:128x128</span>
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
      
      {/* Restored Panel */}
      <div className="flex flex-col bg-surface-container border border-outline-variant rounded">
        <div className="px-2 py-1 border-b border-outline-variant bg-surface-container-high flex justify-between items-center">
          <span className="text-label-xs font-label-xs uppercase text-primary">Restored</span>
          <span className="text-[10px] font-mono text-primary-container">SIGMA_OUT</span>
        </div>
        <div className="flex-1 p-2 flex items-center justify-center relative overflow-hidden bg-[#050810]">
          {isProcessing && <div className="scan-line"></div>}
          <canvas ref={canvasRestored} width={128} height={128} className={`w-full h-full object-contain max-h-[300px] transition-opacity duration-500 ${restoredData ? 'opacity-100' : 'opacity-0'}`} />
        </div>
      </div>
    </div>
  );
}
