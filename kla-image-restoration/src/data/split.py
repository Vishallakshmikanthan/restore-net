"""
Dataset splitting and DataLoader construction utilities.
"""

from typing import Optional, Tuple
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset


def create_train_val_split(
    dataset: Dataset,
    train_ratio: float = 0.70,
    val_ratio: float = 0.20,
    seed: int = 42,
) -> Tuple[Subset, Subset, Subset]:
    """
    Splits a dataset into training, validation, and holdout subsets.

    Args:
        dataset: PyTorch Dataset object to split.
        train_ratio: Fraction of data for training (default: 0.70).
        val_ratio: Fraction of data for validation (default: 0.20).
        seed: Random seed for reproducibility (default: 42).

    Returns:
        (train_dataset, val_dataset, holdout_dataset) as torch.utils.data.Subset objects.
    """
    # Set random seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    n = len(dataset)
    indices = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_indices = indices[:n_train]
    val_indices = indices[n_train : n_train + n_val]
    holdout_indices = indices[n_train + n_val :]

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    holdout_dataset = Subset(dataset, holdout_indices)

    print(
        f"Split: {len(train_dataset)} train | {len(val_dataset)} val | {len(holdout_dataset)} holdout"
    )

    return train_dataset, val_dataset, holdout_dataset


def get_dataloaders(
    train_ds: Dataset,
    val_ds: Dataset,
    batch_size: int = 8,
    num_workers: int = 4,
    pin_memory: bool = True,
    prefetch_factor: Optional[int] = 2,
) -> Tuple[DataLoader, DataLoader]:
    """
    Creates PyTorch DataLoaders for training and validation datasets.

    Args:
        train_ds: Training dataset (torch.utils.data.Dataset).
        val_ds: Validation dataset (torch.utils.data.Dataset).
        batch_size: Batch size per iteration (default: 8).
        num_workers: Number of worker processes (default: 4).
        pin_memory: Whether to pin host memory for faster GPU transfer (default: True).
        prefetch_factor: Number of batches preloaded per worker (default: 2, used when num_workers > 0).

    Returns:
        (train_loader, val_loader) tuple.
    """
    train_kwargs = {
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    val_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }

    if num_workers > 0 and prefetch_factor is not None:
        train_kwargs["prefetch_factor"] = prefetch_factor
        val_kwargs["prefetch_factor"] = prefetch_factor

    train_loader = DataLoader(train_ds, **train_kwargs)
    val_loader = DataLoader(val_ds, **val_kwargs)

    return train_loader, val_loader
