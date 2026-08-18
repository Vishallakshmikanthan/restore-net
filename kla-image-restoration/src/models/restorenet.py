"""RestoreNet: Main Image Restoration Architecture."""

import os
import sys
from pathlib import Path

# Ensure project root is in sys.path when executing directly
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn as nn

from src.models.baseline import count_parameters
from src.models.blocks import ChannelAttention, ResidualBlock


class RestoreNet(nn.Module):
    """RestoreNet image restoration model with progressive residual learning and channel attention.

    Architecture:
    1. Bilinear upsampling by `scale_factor`.
    2. Feature extraction with initial Conv2d.
    3. `num_blocks` ResidualBlocks with interleaved ChannelAttention every 5 blocks.
    4. Mid-level feature convolution.
    5. Output convolution mapping back to single channel.
    6. Global residual addition: output = upsampled + residual.

    Input shape:  (B, 1, H, W)
    Output shape: (B, 1, H * scale_factor, W * scale_factor)
    """

    def __init__(
        self,
        scale_factor: int = 2,
        num_features: int = 64,
        num_blocks: int = 10,
        use_attention: bool = True,
    ):
        super().__init__()
        self.scale_factor = scale_factor
        self.num_features = num_features
        self.num_blocks = num_blocks
        self.use_attention = use_attention

        # 1. Spatial upsampling
        self.upsample = nn.Upsample(
            scale_factor=scale_factor,
            mode="bilinear",
            align_corners=False,
        )

        # 2. Input feature extraction
        self.conv_in = nn.Conv2d(1, num_features, kernel_size=3, padding=1, bias=True)

        # 3. Residual and Attention blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(channels=num_features) for _ in range(num_blocks)
        ])
        num_attn_blocks = max(1, num_blocks // 5)
        self.attention_blocks = nn.ModuleList([
            ChannelAttention(channels=num_features, reduction=16)
            for _ in range(num_attn_blocks)
        ]) if use_attention else nn.ModuleList()

        # 4. Mid convolution and output reconstruction
        self.conv_mid = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1, bias=True)
        self.conv_out = nn.Conv2d(num_features, 1, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input low-resolution / noisy tensor of shape (B, 1, H, W).

        Returns:
            Restored high-resolution tensor of shape (B, 1, H * scale_factor, W * scale_factor).
        """
        upsampled = self.upsample(x)
        feat = self.conv_in(upsampled)

        attn_idx = 0
        for i, block in enumerate(self.res_blocks):
            feat = block(feat)
            if self.use_attention and (i + 1) % 5 == 0 and attn_idx < len(self.attention_blocks):
                feat = self.attention_blocks[attn_idx](feat)
                attn_idx += 1

        feat = self.conv_mid(feat)
        residual = self.conv_out(feat)
        return upsampled + residual


if __name__ == "__main__":
    model = RestoreNet(scale_factor=2, num_features=64, num_blocks=10)
    num_params = count_parameters(model)
    print(f"RestoreNet Parameters: {num_params:,} (~{num_params / 1e6:.2f}M)")

    dummy_input = torch.randn(1, 1, 128, 128)
    output = model(dummy_input)
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")

    assert output.shape == (1, 1, 256, 256), f"Unexpected shape {output.shape}"
    assert not torch.isnan(output).any(), "Found NaN in output"
    assert not torch.isinf(output).any(), "Found Inf in output"
    print("RestoreNet verification passed successfully!")
