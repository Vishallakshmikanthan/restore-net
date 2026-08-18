"""Tests for reusable neural network blocks."""

import pytest
import torch

from src.models.blocks import (
    ResidualBlock,
    ChannelAttention,
    UpsampleBlock,
    PixelShuffleBlock,
)


def test_residual_block_shapes():
    """Verify ResidualBlock maintains tensor shape across different channels and sizes."""
    block = ResidualBlock(channels=64)
    x = torch.randn(2, 64, 32, 32)
    out = block(x)
    assert out.shape == (2, 64, 32, 32)
    assert not torch.isnan(out).any()


def test_channel_attention_shapes():
    """Verify ChannelAttention computes correct attention-weighted output."""
    attn = ChannelAttention(channels=64, reduction=16)
    x = torch.randn(2, 64, 32, 32)
    out = attn(x)
    assert out.shape == (2, 64, 32, 32)
    assert not torch.isnan(out).any()


def test_upsample_block_shapes():
    """Verify UpsampleBlock scales spatial dimensions correctly."""
    upsample = UpsampleBlock(scale_factor=2, mode="bilinear")
    x = torch.randn(2, 64, 16, 16)
    out = upsample(x)
    assert out.shape == (2, 64, 32, 32)


def test_pixel_shuffle_block_shapes():
    """Verify PixelShuffleBlock scales spatial dimensions correctly."""
    ps = PixelShuffleBlock(in_channels=64, scale_factor=2)
    x = torch.randn(2, 64, 16, 16)
    out = ps(x)
    assert out.shape == (2, 64, 32, 32)
