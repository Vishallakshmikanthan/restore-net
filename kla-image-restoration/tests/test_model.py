"""Tests for model building blocks."""

import unittest
import torch

from src.models.blocks import (
    ResidualBlock,
    ChannelAttention,
    UpsampleBlock,
    PixelShuffleBlock,
)


class TestModelBlocks(unittest.TestCase):
    def test_residual_block(self):
        block = ResidualBlock(channels=64)
        x = torch.randn(2, 64, 32, 32)
        out = block(x)
        self.assertEqual(out.shape, (2, 64, 32, 32))
        self.assertFalse(torch.isnan(out).any())

    def test_channel_attention(self):
        attn = ChannelAttention(channels=64, reduction=16)
        x = torch.randn(2, 64, 32, 32)
        out = attn(x)
        self.assertEqual(out.shape, (2, 64, 32, 32))
        self.assertFalse(torch.isnan(out).any())

    def test_upsample_block(self):
        upsample = UpsampleBlock(scale_factor=2, mode="bilinear")
        x = torch.randn(2, 64, 16, 16)
        out = upsample(x)
        self.assertEqual(out.shape, (2, 64, 32, 32))

    def test_pixel_shuffle_block(self):
        ps = PixelShuffleBlock(in_channels=64, scale_factor=2)
        x = torch.randn(2, 64, 16, 16)
        out = ps(x)
        self.assertEqual(out.shape, (2, 64, 32, 32))


if __name__ == "__main__":
    unittest.main()
