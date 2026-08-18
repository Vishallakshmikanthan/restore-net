"""
Validation module for image restoration evaluation.
"""

from typing import Dict, Optional, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.training.metrics import compute_psnr, compute_ssim, compute_lpips


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    device: Union[str, torch.device],
    verbose: bool = False,
) -> Dict[str, Union[float, int]]:
    """
    Evaluates the model on the validation dataloader computing PSNR and SSIM.

    Args:
        model: PyTorch image restoration model.
        val_loader: DataLoader containing (noisy_lr, gt) pairs.
        device: Execution device (e.g. 'cuda' or 'cpu').
        verbose: Whether to print verbose per-batch progress.

    Returns:
        Dict with keys 'val_psnr', 'val_ssim', 'n_images'.
    """
    model.eval()
    dev = torch.device(device) if isinstance(device, str) else device

    psnr_list = []
    ssim_list = []
    total_images = 0

    with torch.no_grad():
        for batch_idx, (noisylr, gt) in enumerate(val_loader):
            noisylr = noisylr.to(dev)
            gt = gt.to(dev)

            # Forward pass
            pred = model(noisylr)

            # Extract numpy arrays for per-image metric calculation
            pred_np = pred.detach().cpu().numpy()
            gt_np = gt.detach().cpu().numpy()

            b_size = pred_np.shape[0]
            total_images += b_size

            for i in range(b_size):
                p = np.clip(pred_np[i, 0], 0.0, 1.0)
                g = np.clip(gt_np[i, 0], 0.0, 1.0)
                psnr_val = compute_psnr(p, g)
                ssim_val = compute_ssim(p, g)
                psnr_list.append(psnr_val)
                ssim_list.append(ssim_val)

            if verbose and (batch_idx + 1) % 10 == 0:
                print(
                    f"Validation batch [{batch_idx + 1}/{len(val_loader)}] - "
                    f"Running PSNR: {np.mean(psnr_list):.2f} dB, SSIM: {np.mean(ssim_list):.4f}"
                )

    mean_psnr = float(np.mean(psnr_list)) if psnr_list else 0.0
    mean_ssim = float(np.mean(ssim_list)) if ssim_list else 0.0

    print(
        f"[Validation Summary] {total_images} images | "
        f"PSNR: {mean_psnr:6.2f} dB | SSIM: {mean_ssim:.4f}"
    )

    return {
        "val_psnr": mean_psnr,
        "val_ssim": mean_ssim,
        "n_images": total_images,
    }


def validate_with_lpips(
    model: nn.Module,
    val_loader: DataLoader,
    device: Union[str, torch.device],
    max_batches: int = 10,
) -> Dict[str, float]:
    """
    Evaluates the model on validation dataloader computing PSNR, SSIM, and LPIPS.

    Args:
        model: PyTorch image restoration model.
        val_loader: DataLoader containing (noisy_lr, gt) pairs.
        device: Execution device.
        max_batches: Maximum number of batches to evaluate LPIPS on (to save time).

    Returns:
        Dict with keys 'val_psnr', 'val_ssim', 'val_lpips'.
    """
    model.eval()
    dev = torch.device(device) if isinstance(device, str) else device

    psnr_list = []
    ssim_list = []
    lpips_list = []
    total_images = 0

    with torch.no_grad():
        for batch_idx, (noisylr, gt) in enumerate(val_loader):
            noisylr = noisylr.to(dev)
            gt = gt.to(dev)

            # Forward pass
            pred = model(noisylr)

            # Per-image PSNR and SSIM
            pred_np = pred.detach().cpu().numpy()
            gt_np = gt.detach().cpu().numpy()
            b_size = pred_np.shape[0]
            total_images += b_size

            for i in range(b_size):
                p = np.clip(pred_np[i, 0], 0.0, 1.0)
                g = np.clip(gt_np[i, 0], 0.0, 1.0)
                psnr_list.append(compute_psnr(p, g))
                ssim_list.append(compute_ssim(p, g))

            # Compute LPIPS on batch if within limit
            if batch_idx < max_batches:
                try:
                    lpips_val = compute_lpips(pred, gt, device=str(dev))
                    lpips_list.append(lpips_val)
                except Exception:
                    pass

    mean_psnr = float(np.mean(psnr_list)) if psnr_list else 0.0
    mean_ssim = float(np.mean(ssim_list)) if ssim_list else 0.0
    mean_lpips = float(np.mean(lpips_list)) if lpips_list else 0.0

    print(
        f"[Validation+LPIPS Summary] {total_images} images | "
        f"PSNR: {mean_psnr:6.2f} dB | SSIM: {mean_ssim:.4f} | LPIPS: {mean_lpips:.4f}"
    )

    return {
        "val_psnr": mean_psnr,
        "val_ssim": mean_ssim,
        "val_lpips": mean_lpips,
    }
