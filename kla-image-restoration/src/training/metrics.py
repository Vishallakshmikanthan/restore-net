"""Evaluation metrics for image restoration (PSNR, SSIM, LPIPS, and MetricsTracker)."""

from typing import Dict, Optional, Union
import numpy as np
import torch

try:
    from skimage.metrics import peak_signal_noise_ratio as _sk_psnr
    from skimage.metrics import structural_similarity as _sk_ssim
except ImportError:
    _sk_psnr = None
    _sk_ssim = None

_lpips_model = None


def get_lpips_model(device: str = "cpu"):
    """Lazily load and cache the AlexNet-based LPIPS model."""
    global _lpips_model
    if _lpips_model is None:
        try:
            import lpips
            _lpips_model = lpips.LPIPS(net="alex", verbose=False).eval()
            for param in _lpips_model.parameters():
                param.requires_grad = False
        except Exception as e:
            print(f"Warning: LPIPS model could not be loaded ({e}). Falling back to dummy LPIPS 0.0.")
            return None
    try:
        _lpips_model = _lpips_model.to(device)
    except Exception:
        pass
    return _lpips_model


def compute_psnr(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute Peak Signal-to-Noise Ratio (PSNR) between pred and target.

    Args:
        pred: Predicted image array (values in [0, 1]).
        target: Ground truth image array (values in [0, 1]).

    Returns:
        PSNR value in dB.
    """
    pred_c = np.clip(pred.astype(np.float64), 0.0, 1.0)
    target_c = np.clip(target.astype(np.float64), 0.0, 1.0)

    if _sk_psnr is not None:
        try:
            val = _sk_psnr(target_c, pred_c, data_range=1.0)
            if np.isinf(val):
                return 100.0
            return float(val)
        except Exception:
            pass

    mse = np.mean((target_c - pred_c) ** 2)
    if mse <= 1e-12:
        return 100.0
    return float(10.0 * np.log10(1.0 / mse))


def compute_ssim(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute Structural Similarity Index (SSIM) between pred and target.

    Args:
        pred: Predicted image array (values in [0, 1]).
        target: Ground truth image array (values in [0, 1]).

    Returns:
        SSIM score in [-1, 1].
    """
    pred_c = np.clip(pred.astype(np.float64), 0.0, 1.0)
    target_c = np.clip(target.astype(np.float64), 0.0, 1.0)

    if _sk_ssim is not None:
        try:
            return float(_sk_ssim(target_c, pred_c, data_range=1.0))
        except Exception:
            pass

    # Simple numpy SSIM implementation if skimage is unavailable
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    mu_x = np.mean(pred_c)
    mu_y = np.mean(target_c)
    sigma_x_sq = np.var(pred_c)
    sigma_y_sq = np.var(target_c)
    sigma_xy = np.mean((pred_c - mu_x) * (target_c - mu_y))

    ssim_num = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    ssim_den = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x_sq + sigma_y_sq + c2)
    return float(ssim_num / (ssim_den + 1e-12))


def compute_lpips(pred: torch.Tensor, target: torch.Tensor, device: str = "cpu") -> float:
    """Compute mean LPIPS distance using AlexNet backbone.

    Args:
        pred: [B, 1, H, W] tensor with values in [0, 1].
        target: [B, 1, H, W] tensor with values in [0, 1].
        device: Device to run LPIPS model on.

    Returns:
        Mean LPIPS distance as float.
    """
    model = get_lpips_model(device=device)
    if model is None:
        return 0.0

    # Convert grayscale [B, 1, H, W] to 3-channel [B, 3, H, W] in [-1, 1] range
    p = torch.clamp(pred, 0.0, 1.0)
    t = torch.clamp(target, 0.0, 1.0)

    if p.shape[1] == 1:
        p = p.repeat(1, 3, 1, 1)
    if t.shape[1] == 1:
        t = t.repeat(1, 3, 1, 1)

    p = (p * 2.0 - 1.0).to(device)
    t = (t * 2.0 - 1.0).to(device)

    with torch.no_grad():
        score = model(p, t).mean().item()
    return float(score)


def compute_all_metrics(pred_batch: torch.Tensor, gt_batch: torch.Tensor, device: str = "cpu") -> Dict[str, float]:
    """Compute average PSNR, SSIM, and LPIPS across a batch of tensors.

    Args:
        pred_batch: [B, 1, H, W] PyTorch tensor in [0, 1].
        gt_batch: [B, 1, H, W] PyTorch tensor in [0, 1].
        device: Compute device.

    Returns:
        Dict with keys 'psnr', 'ssim', 'lpips'.
    """
    pred_np = pred_batch.detach().cpu().numpy()
    gt_np = gt_batch.detach().cpu().numpy()

    batch_size = pred_np.shape[0]
    psnr_list = []
    ssim_list = []

    for i in range(batch_size):
        p = pred_np[i, 0]
        g = gt_np[i, 0]
        psnr_list.append(compute_psnr(p, g))
        ssim_list.append(compute_ssim(p, g))

    mean_psnr = float(np.mean(psnr_list)) if psnr_list else 0.0
    mean_ssim = float(np.mean(ssim_list)) if ssim_list else 0.0
    mean_lpips = compute_lpips(pred_batch, gt_batch, device=device)

    return {
        "psnr": mean_psnr,
        "ssim": mean_ssim,
        "lpips": mean_lpips,
    }


class MetricsTracker:
    """Tracks and computes running averages for restoration evaluation metrics."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Clear all accumulated metrics."""
        self.psnr_values = []
        self.ssim_values = []
        self.lpips_values = []

    def update(self, psnr: float, ssim: float, lpips: float = 0.0):
        """Append metric values."""
        self.psnr_values.append(psnr)
        self.ssim_values.append(ssim)
        self.lpips_values.append(lpips)

    def summary(self) -> Dict[str, float]:
        """Compute mean of each metric across all updates."""
        return {
            "psnr": float(np.mean(self.psnr_values)) if self.psnr_values else 0.0,
            "ssim": float(np.mean(self.ssim_values)) if self.ssim_values else 0.0,
            "lpips": float(np.mean(self.lpips_values)) if self.lpips_values else 0.0,
        }

    def log_string(self) -> str:
        """Format metrics as a clean logging string."""
        s = self.summary()
        return f"PSNR: {s['psnr']:6.2f} dB | SSIM: {s['ssim']:.4f} | LPIPS: {s['lpips']:.4f}"
