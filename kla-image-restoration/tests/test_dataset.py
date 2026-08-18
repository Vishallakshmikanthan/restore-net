"""
Self-contained unit and pipeline integration tests for dataset, augmentation, and splitting.
"""

import os
import sys
import tempfile
from pathlib import Path
import numpy as np
import pytest
import torch

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.augmentation import (
    SyntheticDegradationAugmentor,
    SyntheticRestorationDataset,
)
from src.data.dataset import RestorationDataset
from src.data.split import create_train_val_split, get_dataloaders


def test_full_data_pipeline():
    """
    Prompt 1.5 Verification: Full Data Pipeline Test
    1. Creates 10 dummy GT .npy files (128x128, float32, uniform [0,1]) and 10 matching NoisyLR
       .npy files (128x128, float32, uniform [-0.1, 1.6]) in a temp directory.
    2. Instantiates RestorationDataset with normalize=False, augment=True.
    3. Asserts len(dataset) == 10.
    4. Fetches item 0; asserts noisylr.shape == (1, 128, 128) and gt.shape == (1, 128, 128).
    5. Asserts noisylr.dtype == torch.float32.
    6. Asserts that clipping has NOT been applied: noisylr.min() can be < 0 or noisylr.max() can be > 1.
    7. Calls create_train_val_split and asserts sizes sum to 10.
    8. Instantiates SyntheticDegradationAugmentor and calls generate_synthetic_pair on a dummy image;
       asserts output shape matches input when downsample_factors=(1,).
    9. Prints "ALL DATA PIPELINE TESTS PASSED" on success.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        gt_dir = os.path.join(tmpdir, "GT")
        noisylr_dir = os.path.join(tmpdir, "NoisyLR")
        os.makedirs(gt_dir, exist_ok=True)
        os.makedirs(noisylr_dir, exist_ok=True)

        # 1. Create 10 dummy GT and matching NoisyLR files
        for i in range(10):
            # GT: uniform in [0, 1]
            gt_data = np.random.uniform(0.0, 1.0, (128, 128)).astype(np.float32)
            # NoisyLR: uniform in [-0.1, 1.6]
            noisy_data = np.random.uniform(-0.1, 1.6, (128, 128)).astype(np.float32)

            filename = f"sample_{i:04d}.npy"
            np.save(os.path.join(gt_dir, filename), gt_data)
            np.save(os.path.join(noisylr_dir, filename), noisy_data)

        # 2. Instantiate RestorationDataset with normalize=False, augment=True
        dataset = RestorationDataset(
            gt_dir=gt_dir,
            noisylr_dir=noisylr_dir,
            normalize=False,
            augment=True,
        )

        # 3. Asserts len(dataset) == 10
        assert len(dataset) == 10, f"Expected 10 items, got {len(dataset)}"

        # 4. Fetches item 0; asserts shapes
        noisylr, gt = dataset[0]
        assert noisylr.shape == (1, 128, 128), f"Expected (1, 128, 128), got {noisylr.shape}"
        assert gt.shape == (1, 128, 128), f"Expected (1, 128, 128), got {gt.shape}"

        # 5. Asserts noisylr.dtype == torch.float32
        assert noisylr.dtype == torch.float32, f"Expected torch.float32, got {noisylr.dtype}"
        assert gt.dtype == torch.float32, f"Expected torch.float32, got {gt.dtype}"

        # 6. Asserts that clipping has NOT been applied
        has_out_of_range = (noisylr.min().item() < 0.0) or (noisylr.max().item() > 1.0)
        assert has_out_of_range, "Clipping detected! NoisyLR values should be out-of-range"

        # 7. Calls create_train_val_split and asserts sizes sum to 10
        train_ds, val_ds, holdout_ds = create_train_val_split(
            dataset, train_ratio=0.70, val_ratio=0.20, seed=42
        )
        assert len(train_ds) + len(val_ds) + len(holdout_ds) == 10

        # 8. Instantiates SyntheticDegradationAugmentor and calls generate_synthetic_pair
        augmentor = SyntheticDegradationAugmentor(downsample_factors=(1,))
        dummy_img = np.random.uniform(0.0, 1.0, (128, 128)).astype(np.float32)
        synth_pair = augmentor.generate_synthetic_pair(dummy_img)
        assert synth_pair.shape == dummy_img.shape, (
            f"Expected shape {dummy_img.shape}, got {synth_pair.shape}"
        )

        # 9. Print success message
        print("\nALL DATA PIPELINE TESTS PASSED")


def test_restoration_dataset_super_resolution_shape():
    """Tests paired dataset where GT is 2x the spatial dimensions of NoisyLR (e.g. 256x256 vs 128x128)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        gt_dir = os.path.join(tmpdir, "GT")
        noisylr_dir = os.path.join(tmpdir, "NoisyLR")
        os.makedirs(gt_dir, exist_ok=True)
        os.makedirs(noisylr_dir, exist_ok=True)

        gt_data = np.random.uniform(0.0, 1.0, (256, 256)).astype(np.float32)
        noisy_data = np.random.uniform(-0.1, 1.6, (128, 128)).astype(np.float32)

        np.save(os.path.join(gt_dir, "img_001.npy"), gt_data)
        np.save(os.path.join(noisylr_dir, "img_001.npy"), noisy_data)

        dataset = RestorationDataset(gt_dir=gt_dir, noisylr_dir=noisylr_dir, normalize=True)
        noisy_t, gt_t = dataset[0]

        assert noisy_t.shape == (1, 128, 128)
        assert gt_t.shape == (1, 256, 256)


if __name__ == "__main__":
    test_full_data_pipeline()
    test_restoration_dataset_super_resolution_shape()
