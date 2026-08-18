"""Ablation study script evaluating architecture and loss function components."""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

# Ensure project root is in sys.path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data.dataset import RestorationDataset
from src.data.split import create_train_val_split, get_dataloaders
from src.models.baseline import BaselineRestorationCNN
from src.models.restorenet import RestoreNet
from src.training.losses import RestorationLoss
from src.training.metrics import compute_lpips, compute_psnr, compute_ssim
from src.utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Run structured ablation study.")
    parser.add_argument("--gt_dir", type=str, default="data/GT", help="Directory of GT files")
    parser.add_argument("--noisylr_dir", type=str, default="data/NoisyLR", help="Directory of NoisyLR files")
    parser.add_argument("--fast_epochs", type=int, default=2, help="Number of epochs per ablation config (only used when --model_path is not provided)")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--max_samples", type=int, default=16, help="Limit dataset size for fast ablation testing")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device")
    parser.add_argument("--model_path", type=str, default=None, help="If provided, evaluate this trained checkpoint across all ablation configs (no retraining).")
    parser.add_argument("--output_json", type=str, default="results/metrics/ablation_results.json", help="Output JSON path")
    return parser.parse_args()


def _evaluate(model, val_loader, device):
    """Compute mean PSNR/SSIM/LPIPS on the validation set."""
    model.eval()
    psnr_scores, ssim_scores = [], []
    gt_tensors, pred_tensors = [], []

    with torch.no_grad():
        for noisylr, gt in val_loader:
            noisylr = noisylr.to(device)
            pred = model(noisylr)
            pred_np = pred.cpu().numpy()
            gt_np = gt.numpy()
            for i in range(pred_np.shape[0]):
                p = np.clip(pred_np[i, 0], 0.0, 1.0)
                g = np.clip(gt_np[i, 0], 0.0, 1.0)
                psnr_scores.append(compute_psnr(p, g))
                ssim_scores.append(compute_ssim(p, g))
            gt_tensors.append(gt)
            pred_tensors.append(pred.cpu())

    all_gt = torch.cat(gt_tensors, dim=0) if gt_tensors else torch.zeros(1, 1, 32, 32)
    all_pred = torch.cat(pred_tensors, dim=0) if pred_tensors else torch.zeros(1, 1, 32, 32)
    return {
        "psnr": float(np.mean(psnr_scores)) if psnr_scores else 0.0,
        "ssim": float(np.mean(ssim_scores)) if ssim_scores else 0.0,
        "lpips": float(compute_lpips(all_pred, all_gt, device="cpu")),
    }


def train_and_eval_config(
    model: nn.Module,
    loss_fn: nn.Module,
    train_loader,
    val_loader,
    epochs: int,
    device: torch.device,
) -> Dict[str, float]:
    """Train a specific configuration for fast_epochs and compute validation metrics."""
    optimizer = Adam(model.parameters(), lr=1e-3)
    model.to(device)

    for epoch in range(epochs):
        model.train()
        for noisylr, gt in train_loader:
            noisylr = noisylr.to(device)
            gt = gt.to(device)

            optimizer.zero_grad()
            pred = model(noisylr)
            loss, _ = loss_fn(pred, gt)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

    return _evaluate(model, val_loader, device)


def _load_trained_into(model, model_path: str):
    """Best-effort load of state_dict into the supplied model."""
    try:
        ckpt = torch.load(model_path, map_location="cpu")
        state_dict = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
        cleaned = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}
        model.load_state_dict(cleaned, strict=False)
        return True
    except Exception as e:
        print(f"  Warning: could not load {model_path} into this config: {e}")
        return False


def main():
    args = parse_args()
    set_seed(42)

    target_device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    device = torch.device(target_device)
    if args.model_path:
        print(f"Evaluating trained checkpoint {args.model_path} across ablation configs (no retraining).")
    else:
        print(f"Starting ablation study on device: {device} ({args.fast_epochs} epochs each)")

    # Data setup
    dataset = RestorationDataset(gt_dir=args.gt_dir, noisylr_dir=args.noisylr_dir, augment=True)
    if args.max_samples and args.max_samples < len(dataset):
        dataset = torch.utils.data.Subset(dataset, range(args.max_samples))

    train_ds, val_ds, _ = create_train_val_split(dataset, train_ratio=0.70, val_ratio=0.20, seed=42)
    train_loader, val_loader = get_dataloaders(
        train_ds, val_ds, batch_size=args.batch_size, num_workers=0, pin_memory=(device.type == "cuda")
    )

    configs = {
        "baseline_l1_only": {
            "model": lambda: BaselineRestorationCNN(scale_factor=2, num_features=64, num_blocks=3),
            "loss": lambda: RestorationLoss(lambda_pixel=1.0, lambda_ssim=0.0, lambda_lpips=0.0, device=str(device)),
        },
        "restorenet_l1_only": {
            "model": lambda: RestoreNet(scale_factor=2, num_features=64, num_blocks=10, use_attention=True),
            "loss": lambda: RestorationLoss(lambda_pixel=1.0, lambda_ssim=0.0, lambda_lpips=0.0, device=str(device)),
        },
        "restorenet_l1_ssim": {
            "model": lambda: RestoreNet(scale_factor=2, num_features=64, num_blocks=10, use_attention=True),
            "loss": lambda: RestorationLoss(lambda_pixel=1.0, lambda_ssim=0.3, lambda_lpips=0.0, device=str(device)),
        },
        "restorenet_full": {
            "model": lambda: RestoreNet(scale_factor=2, num_features=64, num_blocks=10, use_attention=True),
            "loss": lambda: RestorationLoss(lambda_pixel=1.0, lambda_ssim=0.3, lambda_lpips=0.1, device=str(device)),
        },
        "restorenet_no_attention": {
            "model": lambda: RestoreNet(scale_factor=2, num_features=64, num_blocks=10, use_attention=False),
            "loss": lambda: RestorationLoss(lambda_pixel=1.0, lambda_ssim=0.3, lambda_lpips=0.1, device=str(device)),
        },
    }

    ablation_results = {}

    for name, factory in configs.items():
        print(f"\nEvaluating ablation config: [{name}]...")
        t0 = time.time()
        m = factory["model"]()
        if args.model_path:
            # Pretrained mode: skip retraining, just evaluate on val
            if not _load_trained_into(m, args.model_path):
                print(f"  Skipping {name} due to weight load failure.")
                continue
            metrics = _evaluate(m, val_loader, device)
        else:
            l = factory["loss"]()
            metrics = train_and_eval_config(m, l, train_loader, val_loader, args.fast_epochs, device)
        elapsed = time.time() - t0
        ablation_results[name] = metrics
        print(f"Finished {name} in {elapsed:.2f}s -> PSNR: {metrics['psnr']:.2f} dB, SSIM: {metrics['ssim']:.4f}, LPIPS: {metrics['lpips']:.4f}")

    # Print markdown table
    print("\n" + "=" * 65)
    print("                    ABLATION STUDY RESULTS                       ")
    print("=" * 65)
    print("| Configuration               | PSNR (dB) | SSIM   | LPIPS  |")
    print("|-----------------------------|-----------|--------|--------|")
    for name, res in ablation_results.items():
        print(f"| {name:<27} | {res['psnr']:9.2f} | {res['ssim']:.4f} | {res['lpips']:.4f} |")
    print("=" * 65)

    # Save results JSON
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(ablation_results, f, indent=2)
    print(f"Ablation results saved to {out_path}")


if __name__ == "__main__":
    main()
