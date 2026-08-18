"""Main production training entry point for RestoreNet."""

import argparse
import os
import pprint
import sys
from pathlib import Path
import torch
from torch.utils.data import ConcatDataset, Subset
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
    # 1. Parse arguments
    args = parse_args()

    # 2. Load config
    config = load_config(args.config)

    # 3. Override config with CLI arguments
    if "training" not in config:
        config["training"] = {}
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size

    # 4. Set config device
    if args.device is not None:
        config["device"] = args.device
    elif "device" not in config:
        config["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    device_str = config["device"]
    if device_str == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        device_str = "cpu"
        config["device"] = "cpu"

    # 5. Print system info
    print_system_info(device_str)

    # 6. Set seed
    seed = config.get("seed", 42)
    set_seed(seed)

    # 7. Pretty print config
    print("Configuration:")
    pprint.pprint(config)
    print("=" * 60)

    # 8. Determine gt_dir and noisylr_dir
    gt_dir = args.gt_dir or os.path.join(config.get("data_root", "./data"), "GT")
    noisylr_dir = args.noisylr_dir or os.path.join(config.get("data_root", "./data"), "NoisyLR")

    if not os.path.exists(gt_dir):
        print(f"Error: GT directory not found at: {gt_dir}")
        sys.exit(1)
    if not os.path.exists(noisylr_dir):
        print(f"Error: NoisyLR directory not found at: {noisylr_dir}")
        sys.exit(1)

    # 9. Build dataset
    print(f"Building official dataset from GT: {gt_dir} and NoisyLR: {noisylr_dir}")
    official_dataset = RestorationDataset(
        gt_dir=gt_dir,
        noisylr_dir=noisylr_dir,
        normalize=False,
        augment=True,
    )

    if args.max_samples is not None and args.max_samples < len(official_dataset.pairs):
        official_dataset.pairs = official_dataset.pairs[: args.max_samples]
        print(f"Truncated official dataset to {len(official_dataset.pairs)} samples (--max_samples).")

    data_cfg = config.get("data", {})
    include_synthetic = data_cfg.get("include_synthetic", False)

    if include_synthetic:
        try:
            print("Including dynamic synthetic degradation pairs...")
            augmentor = SyntheticDegradationAugmentor()
            synth_dir = os.path.join(config.get("data_root", "./data"), "NoisyLR_synth")
            samples_per_img = data_cfg.get("synthetic_samples_per_image", 2)
            # In on-the-fly mode, we wrap GT with SyntheticRestorationDataset
            synth_dataset = SyntheticRestorationDataset(
                gt_dir=gt_dir,
                augmentor=augmentor,
                augment=True,
            )
            if args.max_samples is not None and args.max_samples < len(synth_dataset.gt_files):
                synth_dataset.gt_files = synth_dataset.gt_files[: args.max_samples]
            dataset = ConcatDataset([official_dataset, synth_dataset])
            print(f"Combined official ({len(official_dataset)}) + synthetic ({len(synth_dataset)}) dataset: {len(dataset)} items.")
        except Exception as e:
            print(f"Warning: Could not configure synthetic dataset ({e}). Using official dataset only.")
            dataset = official_dataset
    else:
        dataset = official_dataset

    # 10. Split dataset & get dataloaders
    train_ratio = data_cfg.get("train_ratio", 0.70)
    val_ratio = data_cfg.get("val_ratio", 0.20)
    train_ds, val_ds, holdout_ds = create_train_val_split(
        dataset,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
    )

    batch_size = config.get("training", {}).get("batch_size", 8)
    num_workers = 0 if os.name == "nt" or device_str == "cpu" else 4
    pin_memory = (device_str == "cuda")

    train_loader, val_loader = get_dataloaders(
        train_ds,
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # 11. Build RestoreNet
    model_cfg = config.get("model", {})
    scale_factor = model_cfg.get("scale_factor", 2)
    num_features = model_cfg.get("num_features", 64)
    num_blocks = model_cfg.get("num_blocks", 10)

    model = RestoreNet(
        scale_factor=scale_factor,
        num_features=num_features,
        num_blocks=num_blocks,
    )

    # 12. Parameter count
    print(f"Initialized RestoreNet with {count_parameters(model):,} parameters")

    # 13. Trainer instantiation
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device_str,
    )

    # 14. Resume if requested
    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume)
        print(f"Resuming training from epoch {start_epoch}")

    # 15. Fit
    trainer.fit()

    # 16. Done message
    print("Training complete. Best model saved at checkpoints/best_model.pt")


if __name__ == "__main__":
    main()
