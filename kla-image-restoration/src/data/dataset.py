"""
Restoration Dataset implementation for paired Ground Truth and NoisyLR images.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import Dataset


class RestorationDataset(Dataset):
    """
    PyTorch Dataset for paired Ground Truth (GT) and Low-Resolution / Noisy (NoisyLR) images.
    """

    def __init__(
        self,
        gt_dir: Union[str, Path],
        noisylr_dir: Union[str, Path],
        normalize: bool = False,
        augment: bool = False,
    ) -> None:
        self.gt_dir = Path(gt_dir)
        self.noisylr_dir = Path(noisylr_dir)
        self.normalize = normalize
        self.augment = augment

        # Match pairs by filename stem using fast directory listing
        import os
        gt_files = {
            f[:-4]: self.gt_dir / f
            for f in os.listdir(self.gt_dir)
            if f.endswith(".npy")
        }
        noisy_files = {
            f[:-4]: self.noisylr_dir / f
            for f in os.listdir(self.noisylr_dir)
            if f.endswith(".npy")
        }

        common_stems = sorted(list(set(gt_files.keys()) & set(noisy_files.keys())))
        if not common_stems:
            raise ValueError(
                f"No matching .npy pairs found between {self.gt_dir} and {self.noisylr_dir}"
            )

        self.pairs: List[Tuple[Path, Path]] = [
            (gt_files[stem], noisy_files[stem]) for stem in common_stems
        ]

        print(f"Loaded {len(self.pairs)} GT/NoisyLR pairs")

    def __len__(self) -> int:
        return len(self.pairs)

    def _augment(self, gt: np.ndarray, noisylr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Applies consistent random horizontal flip, vertical flip, and 90-deg rotation."""
        # Random horizontal flip (50%)
        if np.random.rand() > 0.5:
            gt = np.fliplr(gt).copy()
            noisylr = np.fliplr(noisylr).copy()

        # Random vertical flip (50%)
        if np.random.rand() > 0.5:
            gt = np.flipud(gt).copy()
            noisylr = np.flipud(noisylr).copy()

        # Random 90 deg rotation (k in 0-3)
        k = np.random.randint(0, 4)
        if k > 0:
            gt = np.rot90(gt, k).copy()
            noisylr = np.rot90(noisylr, k).copy()

        return gt, noisylr

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        gt_path, noisylr_path = self.pairs[idx]

        # 1. Load as float32 numpy arrays
        gt = np.load(gt_path).astype(np.float32)
        noisylr = np.load(noisylr_path).astype(np.float32)

        # Spatial dimension check: gt.shape == noisylr.shape OR gt is exactly 2x noisylr in spatial dims
        assert (
            gt.shape == noisylr.shape
            or (gt.shape[0] == 2 * noisylr.shape[0] and gt.shape[1] == 2 * noisylr.shape[1])
        ), f"Incompatible shapes between GT {gt.shape} and NoisyLR {noisylr.shape}"

        # 2. Do NOT clip any values
        # 3. Optional per-image normalization for NoisyLR only
        if self.normalize:
            mean = float(np.mean(noisylr))
            std = float(np.std(noisylr))
            noisylr = ((noisylr - mean) / (std + 1e-6)).astype(np.float32)

        # 4. Optional spatial augmentation
        if self.augment:
            gt, noisylr = self._augment(gt, noisylr)

        # 5. Add channel dimension [1, H, W]
        noisylr_tensor = torch.from_numpy(noisylr).unsqueeze(0).float()
        gt_tensor = torch.from_numpy(gt).unsqueeze(0).float()

        # 6 & 7. Return (noisylr_tensor, gt_tensor)
        return noisylr_tensor, gt_tensor
