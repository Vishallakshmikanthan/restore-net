"""Tests for model building blocks and architectures."""

import unittest
import torch

from src.models.baseline import BaselineRestorationCNN, count_parameters
from src.models.blocks import (
    ChannelAttention,
    PixelShuffleBlock,
    ResidualBlock,
    UpsampleBlock,
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


class TestBaselineCNN(unittest.TestCase):
    def test_forward_shape(self):
        model = BaselineRestorationCNN(scale_factor=2, num_features=64, num_blocks=3)
        x = torch.randn(2, 1, 128, 128)
        out = model(x)
        self.assertEqual(out.shape, (2, 1, 256, 256))

    def test_no_nan_output(self):
        model = BaselineRestorationCNN(scale_factor=2, num_features=64, num_blocks=3)
        x = torch.randn(1, 1, 64, 64)
        out = model(x)
        self.assertFalse(torch.isnan(out).any())
        self.assertFalse(torch.isinf(out).any())

    def test_parameter_count(self):
        model = BaselineRestorationCNN(scale_factor=2, num_features=64, num_blocks=3)
        params = count_parameters(model)
        self.assertGreater(params, 100_000)
        self.assertLess(params, 2_000_000)


if __name__ == "__main__":
    unittest.main()
