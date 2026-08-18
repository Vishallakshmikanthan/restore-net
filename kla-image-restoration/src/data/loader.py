"""
Data loader builder for combining official and synthetic datasets.
"""

import os
from typing import Dict, Any, Tuple
import torch
from torch.utils.data import DataLoader, ConcatDataset

from src.data.dataset import RestorationDataset
from src.data.augmentation import SyntheticDegradationAugmentor, SyntheticRestorationDataset
from src.data.split import create_train_val_split


def build_combined_dataloader(
    gt_dir: str,
    noisylr_dir: str,
    config: Dict[str, Any],
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Builds training, validation, and holdout DataLoaders combining official dataset
    and optional synthetic degradation dataset.

    Args:
        gt_dir: Path to clean Ground Truth directory.
        noisylr_dir: Path to degraded NoisyLR directory.
        config: Configuration dictionary with data and training settings.

    Returns:
        (train_loader, val_loader, holdout_loader)
    """
    data_cfg = config.get("data", {})
    train_cfg = config.get("training", {})

    normalize = data_cfg.get("normalize", False)
    augment = data_cfg.get("augment", True)
    batch_size = train_cfg.get("batch_size", config.get("batch_size", 8))
    num_workers = train_cfg.get("num_workers", config.get("num_workers", 4))
    pin_memory = train_cfg.get("pin_memory", True) if torch.cuda.is_available() else False
    prefetch_factor = 2 if num_workers > 0 else None

    # 1. Official paired dataset
    official_dataset = RestorationDataset(
        gt_dir=gt_dir,
        noisylr_dir=noisylr_dir,
        normalize=normalize,
        augment=augment,
    )

    # 2. Synthetic dataset (optional)
    if data_cfg.get("include_synthetic", False):
        noise_range = tuple(data_cfg.get("noise_std_range", [0.01, 0.05]))
        speckle_range = tuple(data_cfg.get("speckle_range", [0.5, 1.5]))
        downsample_factors = tuple(data_cfg.get("downsample_factors", [2, 3, 4]))
        
        augmentor = SyntheticDegradationAugmentor(
            noise_std_range=noise_range,
            speckle_range=speckle_range,
            downsample_factors=downsample_factors,
        )
        synthetic_dataset = SyntheticRestorationDataset(
            gt_dir=gt_dir,
            augmentor=augmentor,
            normalize=normalize,
            augment=augment,
        )
        combined_dataset = ConcatDataset([official_dataset, synthetic_dataset])
    else:
        combined_dataset = official_dataset

    # 3. Create train / val / holdout split
    train_ratio = data_cfg.get("train_ratio", 0.70)
    val_ratio = data_cfg.get("val_ratio", 0.20)
    seed = config.get("seed", 42)

    train_ds, val_ds, holdout_ds = create_train_val_split(
        combined_dataset,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
    )

    # 4. Create DataLoaders
    train_loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": False,
    }
    eval_loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": False,
    }

    if num_workers > 0 and prefetch_factor is not None:
        train_loader_kwargs["prefetch_factor"] = prefetch_factor
        eval_loader_kwargs["prefetch_factor"] = prefetch_factor

    train_loader = DataLoader(train_ds, **train_loader_kwargs)
    val_loader = DataLoader(val_ds, **eval_loader_kwargs)
    holdout_loader = DataLoader(holdout_ds, **eval_loader_kwargs)

    return train_loader, val_loader, holdout_loader
