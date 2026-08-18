import React, { useCallback, useState } from 'react';
import { Upload } from 'lucide-react';

export default function UploadZone({ onUpload }) {
  const [isDragActive, setIsDragActive] = useState(false);

  const processFile = async (file) => {
    if (!file.name.endsWith('.npy')) {
      alert("Please upload a .npy file");
      return;
    }
    
    const buffer = await file.arrayBuffer();
    // A standard .npy file has a header. 
    // In a real robust implementation we'd parse the .npy header.
    // For this prototype, we'll strip the first 128 bytes (approximate npy header length)
    // or just pass the file to the backend directly.
    // Actually, to render preview on the frontend, we need the raw float32 array.
    // Let's do a simple heuristic: the data is at the end of the file.
    
    const dataView = new DataView(buffer);
    // Find the dictionary string in header to know header length
    // usually starts with \x93NUMPY
    let headerLen = 128; 
    
    // Read the dictionary string to find exact length (simplification)
    try {
        const majorVersion = dataView.getUint8(6);
        if (majorVersion === 1) {
            headerLen = 10 + dataView.getUint16(8, true);
        } else if (majorVersion === 2) {
            headerLen = 12 + dataView.getUint32(8, true);
        }
    } catch(e) {}

    const floatArray = new Float32Array(buffer, headerLen);
    
    const shape = Math.floor(Math.sqrt(floatArray.length));
    let min = Infinity, max = -Infinity;
    for(let i=0; i<floatArray.length; i++) {
        if(floatArray[i] < min) min = floatArray[i];
        if(floatArray[i] > max) max = floatArray[i];
    }
    
    onUpload(floatArray, { shape, min, max, originalFile: file });
  };

  const onDrop = useCallback((e) => {
    e.preventDefault();
    setIsDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      processFile(e.dataTransfer.files[0]);
    }
  }, [onUpload]);

  const onFileChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      processFile(e.target.files[0]);
    }
  };

  return (
    <div 
      className={`border-2 border-dashed rounded-lg p-6 flex flex-col items-center justify-center text-center cursor-pointer transition-colors ${isDragActive ? 'border-primary bg-primary/10' : 'border-outline-variant hover:border-primary'}`}
      onDragOver={(e) => { e.preventDefault(); setIsDragActive(true); }}
      onDragLeave={() => setIsDragActive(false)}
      onDrop={onDrop}
      onClick={() => document.getElementById('file-upload').click()}
    >
      <input 
        id="file-upload" 
        type="file" 
        accept=".npy" 
        className="hidden" 
        onChange={onFileChange}
      />
      <Upload size={32} className="text-on-surface-variant mb-2" />
      <div className="text-body-md text-on-surface font-medium">Drop .npy file here or click to upload</div>
      <div className="text-body-sm text-on-surface-variant mt-1">Accepts NumPy .npy float32 arrays</div>
    </div>
  );
}
