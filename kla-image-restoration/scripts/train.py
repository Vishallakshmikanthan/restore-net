"""Main production training entry point for RestoreNet."""

import argparse
import os
import pprint
import sys
from pathlib import Path
import torch
from torch.utils.data import ConcatDataset
import yaml

# Ensure project root is in sys.path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data.augmentation import SyntheticDegradationAugmentor, SyntheticRestorationDataset
from src.data.dataset import RestorationDataset
from src.data.split import create_train_val_split, get_dataloaders
from src.models.baseline import count_parameters
from src.models.restorenet import RestoreNet
from src.training.trainer import Trainer
from src.utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train RestoreNet image restoration model.")
    parser.add_argument("--gt_dir", type=str, default=None, help="Directory containing GT .npy files")
    parser.add_argument("--noisylr_dir", type=str, default=None, help="Directory containing NoisyLR .npy files")
    parser.add_argument("--config", type=str, default="configs/train.yaml", help="Path to config YAML file")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda/cpu)")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit dataset size for fast testing")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


def print_system_info(device_name: str):
    print("=" * 60)
    print("RestoreNet Training Pipeline Initializing")
    print("=" * 60)
    if torch.cuda.is_available() and device_name.startswith("cuda"):
        gpu_name = torch.cuda.get_device_name(0)
        vram_bytes = torch.cuda.get_device_properties(0).total_memory
        vram_gb = vram_bytes / (1024 ** 3)
        print(f"GPU: {gpu_name} | VRAM: {vram_gb:.2f} GB | CUDA Version: {torch.version.cuda}")
    else:
        print(f"Running on Device: {device_name.upper()} (CPU Mode)")
    print("=" * 60)


def main():
    args = parse_args()

    # 1. Load config
    config = load_config(args.config)
    seed = config.get("seed", 42)
    set_seed(seed)

    # Resolve device
    target_device = args.device or config.get("device", "cuda")
    device_str = target_device if torch.cuda.is_available() and target_device == "cuda" else "cpu"
    print_system_info(device_str)

    # Print full config for reproducibility
    print("Configuration:")
    pprint.pprint(config)
    print("=" * 60)

    # Resolve directory paths
    gt_dir = args.gt_dir or os.path.join(config.get("data_root", "./data"), "GT")
    noisylr_dir = args.noisylr_dir or os.path.join(config.get("data_root", "./data"), "NoisyLR")

    # 2. Build Dataset
    print(f"Building official dataset from GT: {gt_dir} and NoisyLR: {noisylr_dir}")
    official_dataset = RestorationDataset(
        gt_dir=gt_dir,
        noisylr_dir=noisylr_dir,
        augment=True,
    )

    data_cfg = config.get("data", {})
    include_synthetic = data_cfg.get("include_synthetic", False)

    if include_synthetic:
        try:
            print("Including dynamic synthetic degradation pairs...")
            synth_dataset = SyntheticRestorationDataset(
                gt_dir=gt_dir,
                augmentor=SyntheticDegradationAugmentor(),
                augment=True,
            )
            dataset = ConcatDataset([official_dataset, synth_dataset])
            print(f"Combined official + synthetic dataset size: {len(dataset)}")
        except Exception as e:
            print(f"Warning: Could not load synthetic dataset ({e}). Using official dataset only.")
            dataset = official_dataset
    else:
        dataset = official_dataset

    if args.max_samples and args.max_samples < len(dataset):
        dataset = torch.utils.data.Subset(dataset, range(args.max_samples))

    # Split dataset
    train_ratio = data_cfg.get("train_ratio", 0.70)
    val_ratio = data_cfg.get("val_ratio", 0.20)
    train_ds, val_ds, holdout_ds = create_train_val_split(
        dataset,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
    )

    training_cfg = config.get("training", {})
    batch_size = args.batch_size or training_cfg.get("batch_size", 8)
    if args.epochs:
        training_cfg["epochs"] = args.epochs

    num_workers = 0 if os.name == "nt" or device_str == "cpu" else 2
    train_loader, val_loader = get_dataloaders(
        train_ds,
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(device_str == "cuda"),
    )

    # 3. Model setup
    model_cfg = config.get("model", {})
    scale_factor = model_cfg.get("scale_factor", 2)
    num_features = model_cfg.get("num_features", 64)
    num_blocks = model_cfg.get("num_blocks", 10)

    model = RestoreNet(
        scale_factor=scale_factor,
        num_features=num_features,
        num_blocks=num_blocks,
    )
    print(f"Initialized RestoreNet with {count_parameters(model):,} parameters")

    # 4. Trainer instantiation
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device_str,
    )

    # 5. Resume if requested
    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume)
        print(f"Resuming training from epoch {start_epoch}")

    # 6. Fit
    trainer.fit()
    print("Training complete. Best model saved at checkpoints/best_model.pt")


if __name__ == "__main__":
    main()
