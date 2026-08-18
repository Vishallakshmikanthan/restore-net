import React, { useCallback, useRef, useState } from 'react';
import { Upload } from 'lucide-react';

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
        alert("Please upload a .npy file");
        return null;
    }
    const buffer = await file.arrayBuffer();
    const headerLen = readNpyHeaderLen(buffer);
    const floatArray = new Float32Array(buffer, headerLen);
    const shape = Math.floor(Math.sqrt(floatArray.length));
    let min = Infinity, max = -Infinity;
    for (let i = 0; i < floatArray.length; i++) {
        if (floatArray[i] < min) min = floatArray[i];
        if (floatArray[i] > max) max = floatArray[i];
    }
    return { floatArray, info: { shape, min, max, originalFile: file } };
};

export default function UploadZone({ onUpload, onGtUpload, gtFile }) {
    const [isDragActive, setIsDragActive] = useState(false);
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
    }, [onUpload]);

    const onFileChange = (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    };

    const onGtFileChange = (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleGtFile(e.target.files[0]);
        }
    };

    return (
        <div className="flex flex-col gap-2">
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
                <div className="text-body-md text-on-surface font-medium">Drop NoisyLR .npy here</div>
                <div className="text-body-sm text-on-surface-variant mt-1">NumPy float32 array</div>
            </div>

            <div className="border border-outline-variant rounded-lg p-3 flex items-center justify-between bg-surface-container-low">
                <div className="flex flex-col">
                    <span className="text-label-xs uppercase text-on-surface-variant font-bold">Ground Truth (optional)</span>
                    <span className="text-[11px] text-on-surface-variant">
                        {gtFile ? `${gtFile.name} loaded` : 'For real PSNR/SSIM metrics'}
                    </span>
                </div>
                <button
                    onClick={() => gtInputRef.current && gtInputRef.current.click()}
                    className="px-3 py-1 bg-primary-container text-[#050810] rounded text-label-xs font-bold uppercase hover:bg-primary transition-colors"
                >
                    {gtFile ? 'Replace GT' : 'Upload GT'}
                </button>
                <input
                    ref={gtInputRef}
                    type="file"
                    accept=".npy"
                    className="hidden"
                    onChange={onGtFileChange}
                />
            </div>
        </div>
    );
}