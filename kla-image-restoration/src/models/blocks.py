"""Reusable neural network blocks for image restoration models.

Includes:
- ResidualBlock: Standard residual block with two convolutions.
- ChannelAttention: Squeeze-and-Excitation channel attention module.
- UpsampleBlock: Interpolation-based upsampling layer.
- PixelShuffleBlock: Learned sub-pixel convolution upsampling layer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Standard Residual Block with two 3x3 convolutions and ReLU activation.

    Input shape:  (B, C, H, W)
    Output shape: (B, C, H, W)
    """

    def __init__(self, channels: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Output tensor of shape (B, C, H, W).
        """
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return out + residual


class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation Channel Attention module.

    Input shape:  (B, C, H, W)
    Output shape: (B, C, H, W)
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        reduced_channels = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, reduced_channels, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(reduced_channels, channels, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Attended tensor of shape (B, C, H, W).
        """
        # 1. Global average pool: (B, C, H, W) -> (B, C, 1, 1)
        w = self.avg_pool(x)
        # 2. Squeeze & Excitation: fc1 -> ReLU -> fc2 -> Sigmoid -> (B, C, 1, 1)
        w = self.relu(self.fc1(w))
        w = self.sigmoid(self.fc2(w))
        # 3. Scale input by channel attention weights
        return x * w


class UpsampleBlock(nn.Module):
    """Interpolation-based upsampling block.

    Input shape:  (B, C, H, W)
    Output shape: (B, C, H * scale_factor, W * scale_factor)
    """

    def __init__(self, scale_factor: int = 2, mode: str = "bilinear"):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Upsampled tensor of shape (B, C, H * scale_factor, W * scale_factor).
        """
        align_corners = False if self.mode in ("bilinear", "bicubic") else None
        return F.interpolate(
            x,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=align_corners,
        )


class PixelShuffleBlock(nn.Module):
    """Learned sub-pixel convolution upsampling block using PixelShuffle.

    Input shape:  (B, C, H, W)
    Output shape: (B, C, H * scale_factor, W * scale_factor)
    """

    def __init__(self, in_channels: int, scale_factor: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            in_channels * (scale_factor**2),
            kernel_size=3,
            padding=1,
            bias=True,
        )
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Upsampled tensor of shape (B, C, H * scale_factor, W * scale_factor).
        """
        return self.pixel_shuffle(self.conv(x))
