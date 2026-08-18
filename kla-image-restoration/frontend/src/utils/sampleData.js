// Generates synthetic semiconductor wafer pattern data (.npy structure) for demo testing
export function generateWaferData(isDegraded = false, size = 128) {
  const floatArray = new Float32Array(size * size);
  const cx = size / 2;
  const cy = size / 2;

  let min = Infinity;
  let max = -Infinity;
  let sum = 0;

  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const idx = y * size + x;

      // Base wafer concentric structure + semiconductor circuit traces
      const dist = Math.sqrt(Math.pow(x - cx, 2) + Math.pow(y - cy, 2));
      let val = Math.sin(dist / 6) * 0.25 + 0.5;

      // Circuit grid lines
      if ((x % 16 < 2) || (y % 16 < 2)) {
        val += 0.2;
      }
      if ((x > 30 && x < 50 && y > 30 && y < 50) || (x > 78 && x < 98 && y > 78 && y < 98)) {
        val += 0.15;
      }

      // Add noise & degradation if degraded
      if (isDegraded) {
        val += (Math.random() - 0.5) * 0.75;
        // Occasional sensor line dropout
        if (y % 32 === 0 && Math.random() > 0.5) {
          val *= 0.1;
        }
      }

      // Clamp to typical range
      val = Math.max(-0.5, Math.min(1.5, val));
      floatArray[idx] = val;

      if (val < min) min = val;
      if (val > max) max = val;
      sum += val;
    }
  }

  const mean = sum / floatArray.length;

  // Create a pseudo .npy File object
  const header = new Uint8Array([
    0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 0x01, 0x00,
    0x40, 0x00,
    // Header string "{'descr': '<f4', 'fortran_order': False, 'shape': (128, 128), }"
  ]);
  const blob = new Blob([header, floatArray], { type: 'application/octet-stream' });
  const file = new File([blob], isDegraded ? 'sample_noisy_lr.npy' : 'sample_ground_truth.npy');

  return {
    floatArray,
    info: {
      shape: `(${size}, ${size})`,
      size,
      dtype: 'float32',
      min,
      max,
      mean,
      originalFile: file,
      isSynthetic: true
    }
  };
}

export function generateRestoredData(inputArray, size = 128) {
  const restoredArray = new Float32Array(inputArray.length);
  const cx = size / 2;
  const cy = size / 2;

  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const idx = y * size + x;
      const dist = Math.sqrt(Math.pow(x - cx, 2) + Math.pow(y - cy, 2));
      let val = Math.sin(dist / 6) * 0.25 + 0.5;

      if ((x % 16 < 2) || (y % 16 < 2)) val += 0.2;
      if ((x > 30 && x < 50 && y > 30 && y < 50) || (x > 78 && x < 98 && y > 78 && y < 98)) val += 0.15;

      // Small residual artifact
      val += (Math.random() - 0.5) * 0.05;
      restoredArray[idx] = Math.max(0, Math.min(1, val));
    }
  }

  return restoredArray;
}
