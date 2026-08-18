"""Baseline CNN model for image restoration."""

import os
import sys
from pathlib import Path

# Ensure project root is in sys.path when running directly
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn as nn

from src.models.blocks import ResidualBlock


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class BaselineRestorationCNN(nn.Module):
    """Baseline restoration CNN with bilinear upsampling and residual blocks.

    Input shape:  (B, 1, H, W)
    Output shape: (B, 1, H * scale_factor, W * scale_factor)
    """

    def __init__(self, scale_factor: int = 2, num_features: int = 64, num_blocks: int = 3):
        super().__init__()
        self.scale_factor = scale_factor
        self.upsample = nn.Upsample(
            scale_factor=scale_factor,
            mode="bilinear",
            align_corners=False,
        )
        self.conv_in = nn.Conv2d(1, num_features, kernel_size=3, padding=1, bias=True)
        self.res_blocks = nn.ModuleList([
            ResidualBlock(channels=num_features) for _ in range(num_blocks)
        ])
        self.conv_out = nn.Conv2d(num_features, 1, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, 1, H, W).

        Returns:
            Output tensor of shape (B, 1, H * scale_factor, W * scale_factor).
        """
        upsampled = self.upsample(x)
        feat = self.conv_in(upsampled)
        for block in self.res_blocks:
            feat = feat + block(feat)
        out = self.conv_out(feat) + upsampled
        return out


if __name__ == "__main__":
    model = BaselineRestorationCNN(scale_factor=2, num_features=64, num_blocks=3)
    num_params = count_parameters(model)
    print(f"BaselineRestorationCNN Parameters: {num_params:,} (~{num_params / 1e6:.2f}M)")

    dummy_input = torch.randn(1, 1, 128, 128)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == (1, 1, 256, 256), f"Unexpected shape {output.shape}"
    assert not torch.isnan(output).any(), "NaN found in output"
    print("Baseline model verification passed successfully!")
