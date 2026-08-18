"""
Synthetic degradation augmentation module for KLA Image Restoration.
"""

from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union
import numpy as np
import scipy.ndimage
import torch
from torch.utils.data import Dataset


class SyntheticDegradationAugmentor:
    """
    Applies synthetic physical degradations (Gaussian noise, speckle noise, downsampling)
    to clean ground truth images to simulate sensor noise and low resolution.
    """

    def __init__(
        self,
        noise_std_range: Tuple[float, float] = (0.01, 0.05),
        speckle_range: Tuple[float, float] = (0.5, 1.5),
        downsample_factors: Sequence[int] = (2, 3, 4),
    ) -> None:
        self.noise_std_range = tuple(noise_std_range)
        self.speckle_range = tuple(speckle_range)
        self.downsample_factors = tuple(downsample_factors)

    def apply_speckle(self, img: np.ndarray, strength: float) -> np.ndarray:
        """
        Multiplicative speckle noise: img * uniform(1/strength, strength).
        Preserves out-of-range values without clipping.
        """
        if strength <= 0:
            return img.copy()
        low = min(1.0 / strength, float(strength))
        high = max(1.0 / strength, float(strength))
        noise = np.random.uniform(low, high, size=img.shape)
        return (img * noise).astype(np.float32)

    def apply_gaussian(self, img: np.ndarray, std: float) -> np.ndarray:
        """
        Additive Gaussian noise: img + normal(0, std).
        Preserves out-of-range values without clipping.
        """
        noise = np.random.normal(0.0, std, size=img.shape)
        return (img + noise).astype(np.float32)

    def apply_downsample(self, img: np.ndarray, factor: int) -> np.ndarray:
        """
        Downsamples image by 1/factor scale using bilinear interpolation (order=1).
        """
        if factor == 1:
            return img.copy()
        # Bilinear interpolation using scipy zoom
        downscaled = scipy.ndimage.zoom(img, zoom=1.0 / factor, order=1)
        return downscaled.astype(np.float32)

    def generate_synthetic_pair(self, gt: np.ndarray) -> np.ndarray:
        """
        Applies all three degradations (Gaussian, speckle, downsampling) in a RANDOM order.

        Args:
            gt: Clean ground truth image array.

        Returns:
            Degraded synthetic NoisyLR image array (float32).
        """
        img = gt.astype(np.float32).copy()

        # Randomize degradation parameters
        std = float(np.random.uniform(self.noise_std_range[0], self.noise_std_range[1]))
        strength = float(np.random.uniform(self.speckle_range[0], self.speckle_range[1]))
        factor = int(np.random.choice(self.downsample_factors))

        # Degradation operations
        ops = [
            lambda x: self.apply_gaussian(x, std),
            lambda x: self.apply_speckle(x, strength),
            lambda x: self.apply_downsample(x, factor),
        ]

        # Apply operations in random permutation
        order = np.random.permutation(len(ops))
        for idx in order:
            img = ops[idx](img)

        return img.astype(np.float32)

    def augment_dataset(
        self, gt_dir: Union[str, Path], output_dir: Union[str, Path], samples_per_image: int = 2
    ) -> int:
        """
        For each .npy file in gt_dir, generates `samples_per_image` synthetic NoisyLR images
        and saves them as {stem}_synth_{i}.npy in output_dir.

        Args:
            gt_dir: Directory containing clean GT .npy files.
            output_dir: Destination directory for synthetic degraded .npy files.
            samples_per_image: Number of degraded samples per GT image.

        Returns:
            Total count of synthetic degraded files generated.
        """
        gt_path = Path(gt_dir)
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        gt_files = sorted(list(gt_path.glob("*.npy")))
        total_generated = 0

        for f in gt_files:
            gt_arr = np.load(f).astype(np.float32)
            stem = f.stem
            for i in range(samples_per_image):
                noisy_synth = self.generate_synthetic_pair(gt_arr)
                save_file = out_path / f"{stem}_synth_{i}.npy"
                np.save(save_file, noisy_synth)
                total_generated += 1

        print(f"Generated {total_generated} synthetic degraded pairs")
        return total_generated


class SyntheticRestorationDataset(Dataset):
    """
    PyTorch Dataset that dynamically creates synthetic degraded pairs on-the-fly from GT images.
    """

    def __init__(
        self,
        gt_dir: Union[str, Path],
        augmentor: Optional[SyntheticDegradationAugmentor] = None,
        normalize: bool = False,
        augment: bool = False,
    ) -> None:
        self.gt_dir = Path(gt_dir)
        self.augmentor = augmentor or SyntheticDegradationAugmentor()
        self.normalize = normalize
        self.augment = augment

        self.gt_files = sorted(list(self.gt_dir.glob("*.npy")))
        if not self.gt_files:
            raise ValueError(f"No .npy files found in gt_dir: {self.gt_dir}")

        print(f"Loaded {len(self.gt_files)} GT images for on-the-fly synthetic generation")

    def __len__(self) -> int:
        return len(self.gt_files)

    def _augment(self, gt: np.ndarray, noisylr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Applies spatial augmentations (random flips and 90-degree rotations)."""
        # Horizontal flip
        if np.random.rand() > 0.5:
            gt = np.fliplr(gt).copy()
            noisylr = np.fliplr(noisylr).copy()

        # Vertical flip
        if np.random.rand() > 0.5:
            gt = np.flipud(gt).copy()
            noisylr = np.flipud(noisylr).copy()

        # Random 90 deg rotation
        k = np.random.randint(0, 4)
        if k > 0:
            gt = np.rot90(gt, k).copy()
            noisylr = np.rot90(noisylr, k).copy()

        return gt, noisylr

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        gt_file = self.gt_files[idx]
        gt = np.load(gt_file).astype(np.float32)

        # Generate synthetic degraded low-res image dynamically
        noisylr = self.augmentor.generate_synthetic_pair(gt)

        # Optional per-image normalization for noisylr only
        if self.normalize:
            mean = float(np.mean(noisylr))
            std = float(np.std(noisylr))
            noisylr = ((noisylr - mean) / (std + 1e-6)).astype(np.float32)

        # Optional spatial augmentations
        if self.augment:
            gt, noisylr = self._augment(gt, noisylr)

        # Add channel dim [1, H, W]
        noisylr_tensor = torch.from_numpy(noisylr).unsqueeze(0).float()
        gt_tensor = torch.from_numpy(gt).unsqueeze(0).float()

        return noisylr_tensor, gt_tensor
