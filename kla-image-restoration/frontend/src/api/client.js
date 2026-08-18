const API_BASE = 'http://localhost:8000/api';

export const restoreImage = async (originalFile) => {
    const formData = new FormData();
    // Reconstruct a File from the buffer if we need to or just use the original File
    formData.append('file', originalFile);

    const response = await fetch(`${API_BASE}/restore`, {
        method: 'POST',
        body: formData,
    });

    if (!response.ok) {
        throw new Error(`Failed to restore: ${response.statusText}`);
    }

    // Read headers for metrics
    const latency = parseFloat(response.headers.get('X-Latency-Ms') || '0');
    const psnr = parseFloat(response.headers.get('X-PSNR') || '0');
    const ssim = parseFloat(response.headers.get('X-SSIM') || '0');
    const lpips = parseFloat(response.headers.get('X-LPIPS') || '0');

    // Read body as array buffer
    const arrayBuffer = await response.arrayBuffer();
    
    // The FastAPI backend writes the output numpy array. 
    // We can parse the float32 data. Since it's a simple np.save, 
    // it will have the numpy header. We can skip the header just like in UploadZone.
    const dataView = new DataView(arrayBuffer);
    let headerLen = 128;
    try {
        const majorVersion = dataView.getUint8(6);
        if (majorVersion === 1) {
            headerLen = 10 + dataView.getUint16(8, true);
        } else if (majorVersion === 2) {
            headerLen = 12 + dataView.getUint32(8, true);
        }
    } catch(e) {}
    
    const floatArray = new Float32Array(arrayBuffer, headerLen);

    return {
        restoredData: floatArray,
        metrics: { psnr, ssim, lpips, latency }
    };
};
