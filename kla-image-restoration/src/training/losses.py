"""Loss functions for image restoration (SSIM, CharbonnierLoss, and composite RestorationLoss)."""

import math
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian(window_size: int, sigma: float) -> torch.Tensor:
    gauss = torch.tensor([
        math.exp(-((x - window_size // 2) ** 2) / (2 * (sigma ** 2)))
        for x in range(window_size)
    ])
    return gauss / gauss.sum()


def _create_gaussian_window(window_size: int = 11, sigma: float = 1.5, channel: int = 1) -> torch.Tensor:
    _1D_window = _gaussian(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    return _2D_window.expand(channel, 1, window_size, window_size).contiguous()


class SSIM(nn.Module):
    """Differentiable Structural Similarity (SSIM) loss component.

    Uses an 11x11 Gaussian window (sigma=1.5) with local statistics calculated via F.conv2d.
    """

    def __init__(self, window_size: int = 11, sigma: float = 1.5, channel: int = 1):
        super().__init__()
        self.window_size = window_size
        self.channel = channel
        self.c1 = 0.01 ** 2
        self.c2 = 0.03 ** 2
        window = _create_gaussian_window(window_size, sigma, channel)
        self.register_buffer("window", window)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape (B, 1, H, W).
            y: Tensor of shape (B, 1, H, W).

        Returns:
            Scalar SSIM score averaged across batch and spatial locations.
        """
        # Ensure window is on same device and dtype
        window = self.window.to(dtype=x.dtype, device=x.device)

        mu_x = F.conv2d(x, window, padding=self.window_size // 2, groups=self.channel)
        mu_y = F.conv2d(y, window, padding=self.window_size // 2, groups=self.channel)

        mu_x_sq = mu_x.pow(2)
        mu_y_sq = mu_y.pow(2)
        mu_xy = mu_x * mu_y

        sigma_x_sq = (
            F.conv2d(x * x, window, padding=self.window_size // 2, groups=self.channel)
            - mu_x_sq
        )
        sigma_y_sq = (
            F.conv2d(y * y, window, padding=self.window_size // 2, groups=self.channel)
            - mu_y_sq
        )
        sigma_xy = (
            F.conv2d(x * y, window, padding=self.window_size // 2, groups=self.channel)
            - mu_xy
        )

        ssim_map = ((2 * mu_xy + self.c1) * (2 * sigma_xy + self.c2)) / (
            (mu_x_sq + mu_y_sq + self.c1) * (sigma_x_sq + sigma_y_sq + self.c2) + 1e-8
        )
        return ssim_map.mean()


class CharbonnierLoss(nn.Module):
    """Differentiable Charbonnier Loss (smooth L1 approximation)."""

    def __init__(self, eps: float = 0.01):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor, eps: float = None) -> torch.Tensor:
        epsilon = eps if eps is not None else self.eps
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + (epsilon ** 2)))


class RestorationLoss(nn.Module):
    """Composite loss function combining Pixel Loss (L1), Structural Loss (SSIM), and Perceptual Loss (LPIPS)."""

    def __init__(
        self,
        lambda_pixel: float = 1.0,
        lambda_ssim: float = 0.3,
        lambda_lpips: float = 0.1,
        device: str = "cuda",
    ):
        super().__init__()
        self.lambda_pixel = lambda_pixel
        self.lambda_ssim = lambda_ssim
        self.lambda_lpips = lambda_lpips
        self.target_device = device

        self.l1 = nn.L1Loss()
        self.ssim_fn = SSIM()

        self.lpips_fn = None
        if lambda_lpips > 0:
            try:
                import lpips
                self.lpips_fn = lpips.LPIPS(net="alex", verbose=False).eval()
                for param in self.lpips_fn.parameters():
                    param.requires_grad = False
            except Exception as e:
                print(f"Warning: LPIPS unavailable ({e}), using L1+SSIM only.")
                self.lpips_fn = None

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute composite loss.

        Args:
            pred: Predicted tensor of shape (B, 1, H, W).
            target: Ground truth tensor of shape (B, 1, H, W).

        Returns:
            Tuple of (total_loss_tensor, loss_dict).
        """
        pred_clipped = torch.clamp(pred, 0.0, 1.0)

        # 1. Pixel L1 loss
        l1_loss = self.l1(pred_clipped, target)

        # 2. SSIM loss (1 - SSIM)
        ssim_val = self.ssim_fn(pred_clipped, target)
        ssim_loss = 1.0 - ssim_val

        # 3. LPIPS loss
        if self.lpips_fn is not None and self.lambda_lpips > 0:
            try:
                if next(self.lpips_fn.parameters()).device != pred.device:
                    self.lpips_fn = self.lpips_fn.to(pred.device)
                p_rgb = pred_clipped.repeat(1, 3, 1, 1) * 2.0 - 1.0
                t_rgb = target.repeat(1, 3, 1, 1) * 2.0 - 1.0
                with torch.no_grad():
                    lpips_loss = self.lpips_fn(p_rgb, t_rgb).mean()
            except Exception:
                lpips_loss = torch.tensor(0.0, device=pred.device)
        else:
            lpips_loss = torch.tensor(0.0, device=pred.device)

        total_loss = (
            self.lambda_pixel * l1_loss
            + self.lambda_ssim * ssim_loss
            + self.lambda_lpips * lpips_loss
        )

        loss_dict = {
            "l1": l1_loss.item(),
            "ssim_loss": ssim_loss.item(),
            "lpips": lpips_loss.item() if isinstance(lpips_loss, torch.Tensor) else float(lpips_loss),
        }
        return total_loss, loss_dict
