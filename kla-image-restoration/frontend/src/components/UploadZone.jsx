import React, { useCallback, useRef, useState } from 'react';
import { Upload, FileCheck, Layers, Cpu, ChevronDown, ChevronUp, RefreshCw, Zap } from 'lucide-react';
import { generateWaferData } from '../utils/sampleData';

const readNpyHeaderLen = (buffer) => {
  const dataView = new DataView(buffer);
  let headerLen = 128;
  try {
    const majorVersion = dataView.getUint8(6);
    if (majorVersion === 1) {
      headerLen = 10 + dataView.getUint16(8, true);
    } else if (majorVersion === 2) {
      headerLen = 12 + dataView.getUint32(8, true);
    }
  } catch (e) {}
  return headerLen;
};

const processNpyFile = async (file) => {
  if (!file.name.endsWith('.npy')) {
    alert("Please upload a .npy array file");
    return null;
  }
  const buffer = await file.arrayBuffer();
  const headerLen = readNpyHeaderLen(buffer);
  const floatArray = new Float32Array(buffer, headerLen);
  const size = Math.floor(Math.sqrt(floatArray.length));

  let min = Infinity, max = -Infinity, sum = 0;
  for (let i = 0; i < floatArray.length; i++) {
    const v = floatArray[i];
    if (v < min) min = v;
    if (v > max) max = v;
    sum += v;
  }
  const mean = sum / (floatArray.length || 1);

  return {
    floatArray,
    info: {
      shape: `(${size}, ${size})`,
      size,
      dtype: 'float32',
      min,
      max,
      mean,
      originalFile: file
    }
  };
};

export default function UploadZone({
  onUpload,
  onGtUpload,
  fileInfo,
  gtFile,
  currentState,
  onRunInference,
  modelConfig,
  setModelConfig
}) {
  const [isDragActive, setIsDragActive] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const fileInputRef = useRef(null);
  const gtInputRef = useRef(null);

  const handleFile = async (file) => {
    const result = await processNpyFile(file);
    if (result && onUpload) onUpload(result.floatArray, result.info);
  };

  const handleGtFile = async (file) => {
    if (!file) return;
    const result = await processNpyFile(file);
    if (result && onGtUpload) onGtUpload(result.floatArray, result.info);
  };

  const onDrop = useCallback((e) => {
    e.preventDefault();
    setIsDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFile(e.dataTransfer.files[0]);
    }
  }, []);

  const handleLoadSample = () => {
    const sample = generateWaferData(true, 128);
    if (onUpload) onUpload(sample.floatArray, sample.info);
  };

  const getStatusColor = () => {
    switch (currentState) {
      case 'PROCESSING':
        return { dot: 'bg-[#FFC107] animate-ping', text: 'text-[#FFC107]', label: 'INFERENCE RUNNING' };
      case 'COMPLETE':
        return { dot: 'bg-success', text: 'text-success', label: 'COMPLETE' };
      case 'ERROR':
        return { dot: 'bg-alert', text: 'text-alert', label: 'ERROR' };
      default:
        return fileInfo
          ? { dot: 'bg-accent-cyan', text: 'text-accent-cyan', label: 'DATA LOADED' }
          : { dot: 'bg-alert', text: 'text-on-surface-variant', label: 'AWAITING INPUT' };
    }
  };

  const status = getStatusColor();

  return (
    <div className="flex flex-col gap-4">
      {/* Upload Zone */}
      <div
        className={`h-[150px] border border-dashed rounded-clinical flex flex-col items-center justify-center cursor-pointer transition-all relative group bg-layer-top ${
          isDragActive
            ? 'border-accent-cyan bg-accent-cyan/10'
            : fileInfo
            ? 'border-accent-cyan/60 hover:border-accent-cyan'
            : 'border-border-subtle hover:border-accent-cyan/50'
        }`}
        onDragOver={(e) => { e.preventDefault(); setIsDragActive(true); }}
        onDragLeave={() => setIsDragActive(false)}
        onDrop={onDrop}
        onClick={() => fileInputRef.current && fileInputRef.current.click()}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".npy"
          className="hidden"
          onChange={(e) => e.target.files && e.target.files[0] && handleFile(e.target.files[0])}
        />
        {fileInfo ? (
          <div className="flex flex-col items-center text-center p-3">
            <FileCheck className="text-accent-cyan mb-1.5 w-7 h-7" />
            <span className="text-[13px] font-mono font-bold text-on-surface truncate max-w-[280px]">
              {fileInfo.originalFile.name}
            </span>
            <span className="text-[11px] font-mono text-accent-cyan mt-0.5">Click or drop to replace array</span>
          </div>
        ) : (
          <div className="flex flex-col items-center text-center p-3">
            <Upload className="text-on-surface-variant group-hover:text-accent-cyan transition-colors mb-2 w-7 h-7" />
            <span className="text-[13px] font-mono font-medium text-on-surface">Drag &amp; Drop .npy array</span>
            <span className="text-[11px] text-on-surface-variant mt-0.5">or click to browse local files</span>
          </div>
        )}
      </div>

      {/* Quick Demo Wafer Loader Button */}
      <div className="flex justify-between items-center px-1">
        <button
          type="button"
          onClick={handleLoadSample}
          className="text-[11px] font-mono text-accent-cyan hover:underline flex items-center gap-1 cursor-pointer"
        >
          <RefreshCw size={12} />
          Load Synthetic Wafer (.npy)
        </button>
        {fileInfo?.isSynthetic && (
          <span className="text-[10px] font-mono text-on-surface-variant/80 uppercase">DEMO WAFER ACTIVE</span>
        )}
      </div>

      {/* Array Metadata Card */}
      <div className="bg-layer-top border border-border-subtle rounded-clinical p-3.5">
        <div className="text-[10px] font-mono font-bold text-on-surface-variant border-b border-border-subtle pb-2 mb-2.5 tracking-wider uppercase">
          ARRAY METADATA
        </div>
        <div className="grid grid-cols-2 gap-y-2.5 gap-x-4">
          <div className="flex flex-col">
            <span className="text-[10px] font-mono text-on-surface-variant uppercase">SHAPE</span>
            <span className="text-[13px] font-mono font-semibold text-on-surface">
              {fileInfo ? fileInfo.shape : '--'}
            </span>
          </div>
          <div className="flex flex-col">
            <span className="text-[10px] font-mono text-on-surface-variant uppercase">DTYPE</span>
            <span className="text-[13px] font-mono font-semibold text-on-surface">
              {fileInfo ? fileInfo.dtype : '--'}
            </span>
          </div>
          <div className="flex flex-col">
            <span className="text-[10px] font-mono text-on-surface-variant uppercase">MIN</span>
            <span className="text-[13px] font-mono text-on-surface">
              {fileInfo ? fileInfo.min.toFixed(3) : '--'}
            </span>
          </div>
          <div className="flex flex-col">
            <span className="text-[10px] font-mono text-on-surface-variant uppercase">MAX</span>
            <span className="text-[13px] font-mono text-on-surface">
              {fileInfo ? fileInfo.max.toFixed(3) : '--'}
            </span>
          </div>
          <div className="flex flex-col">
            <span className="text-[10px] font-mono text-on-surface-variant uppercase">MEAN</span>
            <span className="text-[13px] font-mono text-on-surface">
              {fileInfo ? fileInfo.mean.toFixed(3) : '--'}
            </span>
          </div>
          <div className="flex flex-col">
            <span className="text-[10px] font-mono text-on-surface-variant uppercase">STATUS</span>
            <div className="flex items-center gap-1.5 mt-0.5">
              <div className={`w-1.5 h-1.5 rounded-[1px] ${status.dot}`}></div>
              <span className={`text-[11px] font-mono font-medium ${status.text}`}>
                {status.label}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Ground Truth Upload (Optional for Real PSNR/SSIM evaluation) */}
      <div className="bg-layer-top border border-border-subtle rounded-clinical p-3 flex items-center justify-between">
        <div className="flex flex-col pr-2">
          <span className="text-[10px] font-mono uppercase text-on-surface-variant font-bold">
            Ground Truth .npy
          </span>
          <span className="text-[11px] font-mono text-on-surface truncate max-w-[180px]">
            {gtFile ? gtFile.name : 'Optional for real metrics'}
          </span>
        </div>
        <button
          type="button"
          onClick={() => gtInputRef.current && gtInputRef.current.click()}
          className="px-2.5 py-1 bg-surface-container border border-border-subtle hover:border-accent-cyan text-on-surface rounded text-[11px] font-mono font-semibold transition-colors shrink-0"
        >
          {gtFile ? 'Change GT' : '+ Add GT'}
        </button>
        <input
          ref={gtInputRef}
          type="file"
          accept=".npy"
          className="hidden"
          onChange={(e) => e.target.files && e.target.files[0] && handleGtFile(e.target.files[0])}
        />
      </div>

      {/* Run Inference Button */}
      <button
        onClick={onRunInference}
        disabled={!fileInfo || currentState === 'PROCESSING'}
        className="bg-accent-cyan text-layer-base font-mono text-[13px] font-bold uppercase py-3 rounded-clinical flex items-center justify-center gap-2 hover:bg-primary-fixed-dim transition-all shadow-[0_0_12px_rgba(0,229,255,0.3)] hover:shadow-[0_0_20px_rgba(0,229,255,0.6)] disabled:opacity-40 disabled:cursor-not-allowed disabled:shadow-none w-full cursor-pointer"
      >
        {currentState === 'PROCESSING' ? (
          <>
            <RefreshCw size={16} className="animate-spin text-layer-base" />
            <span>Processing Array...</span>
          </>
        ) : (
          <>
            <Zap size={16} className="fill-current text-layer-base" />
            <span>{gtFile ? 'Evaluate (Real Metrics)' : 'Run Inference'}</span>
          </>
        )}
      </button>

      {/* Advanced Config Accordion */}
      <div className="border border-border-subtle rounded-clinical overflow-hidden">
        <button
          type="button"
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="w-full bg-layer-top p-2.5 flex justify-between items-center text-[11px] font-mono text-on-surface border-b border-border-subtle hover:bg-layer-hover transition-colors cursor-pointer"
        >
          <span className="font-semibold">ADVANCED CONFIG</span>
          {showAdvanced ? <ChevronUp size={15} /> : <ChevronDown size={15} />}
        </button>
        {showAdvanced && (
          <div className="p-3 bg-layer-base flex flex-col gap-3">
            <div className="flex flex-col gap-1">
              <div className="flex justify-between text-[10px] font-mono text-on-surface-variant">
                <span>DENOISING STRENGTH</span>
                <span className="text-accent-cyan font-bold">{modelConfig.strength}%</span>
              </div>
              <input
                type="range"
                min="0"
                max="100"
                value={modelConfig.strength}
                onChange={(e) => setModelConfig({ ...modelConfig, strength: Number(e.target.value) })}
                className="w-full accent-accent-cyan cursor-pointer"
              />
            </div>
            <div className="flex flex-col gap-1">
              <label className="text-[10px] font-mono text-on-surface-variant uppercase">
                MODEL CHECKPOINT
              </label>
              <div className="bg-layer-top border border-border-subtle text-[12px] font-mono text-on-surface rounded p-2 flex items-center justify-between">
                <span>best_model.pt</span>
                <span className="text-[10px] text-success">LOADED</span>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}