"""Standalone training script for the baseline restoration model."""

import argparse
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import yaml

try:
    from skimage.metrics import peak_signal_noise_ratio
except ImportError:
    def peak_signal_noise_ratio(image_true, image_test, data_range=1.0):
        mse = np.mean((image_true.astype(np.float64) - image_test.astype(np.float64)) ** 2)
        if mse == 0:
            return 100.0
        return float(10 * np.log10((data_range ** 2) / mse))

# Ensure project root is in sys.path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data.dataset import RestorationDataset
from src.data.split import create_train_val_split, get_dataloaders
from src.models.baseline import BaselineRestorationCNN, count_parameters
from src.utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train baseline restoration CNN model.")
    parser.add_argument("--gt_dir", type=str, default=None, help="Directory containing GT .npy files")
    parser.add_argument("--noisylr_dir", type=str, default=None, help="Directory containing NoisyLR .npy files")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml", help="Path to config YAML file")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Output directory for checkpoints")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda/cpu)")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of dataset samples (for fast debug)")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for noisylr, gt in train_loader:
        noisylr = noisylr.to(device)
        gt = gt.to(device)

        optimizer.zero_grad()
        pred = model(noisylr)
        loss = criterion(pred, gt)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(1, num_batches)


def validate(model, val_loader, device):
    model.eval()
    psnr_scores = []

    with torch.no_grad():
        for noisylr, gt in val_loader:
            noisylr = noisylr.to(device)
            pred = model(noisylr)

            pred_np = pred.detach().cpu().numpy()
            gt_np = gt.detach().cpu().numpy()

            for i in range(pred_np.shape[0]):
                p = np.clip(pred_np[i, 0], 0.0, 1.0)
                g = np.clip(gt_np[i, 0], 0.0, 1.0)
                psnr = peak_signal_noise_ratio(g, p, data_range=1.0)
                psnr_scores.append(psnr)

    return float(np.mean(psnr_scores)) if psnr_scores else 0.0


def main():
    args = parse_args()

    # 1. Load config
    config = load_config(args.config)

    # Resolve settings
    seed = config.get("seed", 42)
    set_seed(seed)

    gt_dir = args.gt_dir or os.path.join(config.get("data_root", "./data"), "GT")
    noisylr_dir = args.noisylr_dir or os.path.join(config.get("data_root", "./data"), "NoisyLR")
    output_dir = Path(args.output_dir or config.get("checkpoint_dir", "./checkpoints"))
    output_dir.mkdir(parents=True, exist_ok=True)

    target_device = args.device or config.get("device", "cuda")
    device = torch.device(target_device if torch.cuda.is_available() and target_device == "cuda" else "cpu")
    print(f"Using device: {device}")

    # Training hyperparameters
    training_cfg = config.get("training", {})
    epochs = args.epochs or training_cfg.get("epochs", 50)
    batch_size = args.batch_size or training_cfg.get("batch_size", 8)
    learning_rate = float(training_cfg.get("learning_rate", 1e-3))

    # 2. Dataset & Dataloaders
    print(f"Loading data from GT: {gt_dir}, NoisyLR: {noisylr_dir}")
    dataset = RestorationDataset(gt_dir=gt_dir, noisylr_dir=noisylr_dir, augment=True)
    if args.max_samples and args.max_samples < len(dataset):
        dataset = torch.utils.data.Subset(dataset, range(args.max_samples))

    train_ds, val_ds, holdout_ds = create_train_val_split(dataset, train_ratio=0.70, val_ratio=0.20, seed=seed)

    num_workers = 0 if os.name == "nt" or device.type == "cpu" else 2
    train_loader, val_loader = get_dataloaders(
        train_ds,
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # 3. Model
    model_cfg = config.get("model", {})
    scale_factor = model_cfg.get("scale_factor", 2)
    num_features = model_cfg.get("num_features", 64)
    num_blocks = model_cfg.get("num_blocks", 3)

    model = BaselineRestorationCNN(
        scale_factor=scale_factor,
        num_features=num_features,
        num_blocks=num_blocks,
    ).to(device)

    print(f"Model instantiated with {count_parameters(model):,} parameters")

    # 4. Criterion, Optimizer, Scheduler
    criterion = nn.L1Loss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_psnr = -float("inf")
    start_time = time.time()

    print(f"Starting baseline training for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_psnr = validate(model, val_loader, device)
        scheduler.step()

        elapsed = time.time() - epoch_start
        print(f"Epoch {epoch:03d}/{epochs:03d} | Train Loss: {train_loss:.4f} | Val PSNR: {val_psnr:6.2f} dB | Time: {elapsed:.2f}s")

        # Save best checkpoint
        if val_psnr > best_val_psnr:
            best_val_psnr = val_psnr
            best_ckpt_path = output_dir / "baseline_best.pt"
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "val_psnr": best_val_psnr,
                    "config": config,
                },
                best_ckpt_path,
            )

    # Save final checkpoint
    final_ckpt_path = output_dir / "baseline_final.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "epoch": epochs,
            "val_psnr": val_psnr,
            "config": config,
        },
        final_ckpt_path,
    )

    total_time = time.time() - start_time
    print(f"Training completed in {total_time / 60:.2f} minutes.")
    print(f"Best Val PSNR: {best_val_psnr:.2f} dB (saved to {output_dir / 'baseline_best.pt'})")
    print(f"Final checkpoint saved to {final_ckpt_path}")


if __name__ == "__main__":
    main()
