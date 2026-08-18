import React, { useState, useEffect } from 'react';
import { Upload, Zap, Settings, HelpCircle, Activity } from 'lucide-react';
import UploadZone from './components/UploadZone';
import ImageDisplay from './components/ImageDisplay';
import MetricsBar from './components/MetricsBar';

const STATE = {
  IDLE: 'IDLE',
  PROCESSING: 'PROCESSING',
  COMPLETE: 'COMPLETE',
  ERROR: 'ERROR'
};

function App() {
  const [currentState, setCurrentState] = useState(STATE.IDLE);
  const [fileData, setFileData] = useState(null); // Float32Array
  const [fileInfo, setFileInfo] = useState(null); // { shape, min, max }
  const [restoredData, setRestoredData] = useState(null); // Float32Array
  const [metrics, setMetrics] = useState({ psnr: 0, ssim: 0, lpips: 0, latency: 0 });

  const handleFileUpload = (data, info) => {
    setFileData(data);
    setFileInfo(info);
    setCurrentState(STATE.IDLE);
    setRestoredData(null);
  };

  const handleRestore = async () => {
    if (!fileData || currentState === STATE.PROCESSING) return;
    
    setCurrentState(STATE.PROCESSING);
    
    // Convert Float32Array to Blob (simulating npy or passing binary)
    // Actually, we'll send the raw bytes to the API. 
    // For now, wait 2s to simulate processing if backend is not wired yet.
    // In Phase 4 we will replace this with real API call.
    setTimeout(() => {
      // Dummy restored data
      const dummy = new Float32Array(fileData.length);
      for(let i=0; i<fileData.length; i++) dummy[i] = Math.min(1.0, Math.max(0, fileData[i] + 0.1));
      setRestoredData(dummy);
      setMetrics({ psnr: 27.43, ssim: 0.812, lpips: 0.134, latency: 38 });
      setCurrentState(STATE.COMPLETE);
    }, 2000);
  };

  return (
    <div className="bg-background text-on-surface font-body-md h-screen overflow-hidden flex flex-col">
      {/* TopNavBar */}
      <nav className="h-[48px] w-full fixed top-0 left-0 border-b border-outline-variant flex justify-between items-center px-gutter z-50 bg-surface-container">
        <div className="flex items-center gap-2 text-headline-md font-headline-md font-bold tracking-tight text-primary">
          <Activity size={24} />
          RestoreNet
        </div>
        <div className="flex items-center gap-stack-md">
          <div className="px-unit py-[2px] bg-surface-variant rounded border border-outline-variant text-label-xs font-label-xs text-on-surface-variant flex items-center gap-1">
            <span 
              className={`w-[6px] h-[6px] rounded-full block`} 
              style={{ backgroundColor: currentState === STATE.PROCESSING ? '#c3f5ff' : currentState === STATE.COMPLETE ? '#00C896' : '#849396' }}
            ></span>
            <span>{currentState}</span>
          </div>
          <div className="flex items-center gap-2 text-on-surface-variant">
            <Settings size={20} className="cursor-pointer hover:text-on-surface transition-colors" />
            <HelpCircle size={20} className="cursor-pointer hover:text-on-surface transition-colors" />
          </div>
        </div>
      </nav>

      <div className="flex h-screen pt-[48px] w-full">
        {/* SideNavBar */}
        <aside className="fixed left-0 top-[48px] bottom-0 w-[380px] border-r border-outline-variant bg-surface-container-low flex flex-col h-full py-stack-lg gap-stack-md z-40">
          <div className="px-gutter mb-4 flex-1">
            <h2 className="text-label-md font-label-md font-bold text-on-surface mb-4 uppercase">Control Panel</h2>
            <UploadZone onUpload={handleFileUpload} />
            
            {fileInfo && (
              <div className="mt-4 p-3 bg-surface border border-outline-variant rounded">
                <div className="text-label-xs text-on-surface-variant uppercase mb-2">Image Info</div>
                <div className="text-body-sm flex justify-between"><span>Shape:</span> <span>{fileInfo.shape}x{fileInfo.shape}</span></div>
                <div className="text-body-sm flex justify-between"><span>Range:</span> <span>[{fileInfo.min.toFixed(2)}, {fileInfo.max.toFixed(2)}]</span></div>
              </div>
            )}
          </div>
          
          <div className="px-gutter mt-auto flex flex-col gap-4">
            <button 
              onClick={handleRestore}
              disabled={!fileData || currentState === STATE.PROCESSING}
              className="w-full bg-primary-container text-[#050810] font-label-md font-bold uppercase py-2 rounded transition-colors hover:bg-primary disabled:opacity-50 disabled:cursor-not-allowed flex justify-center items-center gap-2"
            >
              {currentState === STATE.PROCESSING ? 'Processing...' : 'Run Inference'}
              {!currentState.PROCESSING && <Zap size={16} />}
            </button>
          </div>
        </aside>

        {/* Main Content Area */}
        <main className={`ml-[380px] flex-1 flex flex-col h-full overflow-y-auto bg-background p-gutter gap-stack-lg relative ${currentState === STATE.PROCESSING ? 'processing' : ''}`}>
          <div className="h-[60px] flex items-center justify-between border-b border-outline-variant pb-2 shrink-0">
            <div className="flex flex-col">
              <span className="text-label-xs font-label-xs text-on-surface-variant uppercase tracking-widest">KLA SEMICON India Hackathon 2026</span>
              <span className="text-body-md font-body-md text-on-surface">Signal Recovery. Fidelity First.</span>
            </div>
          </div>

          {/* 3-Panel Image Viewer */}
          <ImageDisplay 
            inputData={fileData} 
            restoredData={restoredData} 
            isProcessing={currentState === STATE.PROCESSING} 
          />

          {/* Metrics Row */}
          <MetricsBar metrics={metrics} />
        </main>
      </div>
    </div>
  );
}

export default App;
