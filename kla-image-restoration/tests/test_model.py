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
from src.models.restorenet import RestoreNet


class TestModelBlocks(unittest.TestCase):
    def test_residual_block(self):
        block = ResidualBlock(channels=64)
        x = torch.randn(2, 64, 32, 32)
        out = block(x)
        self.assertEqual(out.shape, (2, 64, 32, 32))
        self.assertFalse(torch.isnan(out).any())

    def test_residual_block_identity_init(self):
        block = ResidualBlock(channels=64)
        with torch.no_grad():
            block.conv1.weight.zero_()
            block.conv1.bias.zero_()
            block.conv2.weight.zero_()
            block.conv2.bias.zero_()
        x = torch.randn(2, 64, 16, 16)
        out = block(x)
        self.assertTrue(torch.allclose(out, x, atol=1e-6))

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


class TestRestoreNet(unittest.TestCase):
    def test_forward_shape(self):
        model = RestoreNet(scale_factor=2, num_features=64, num_blocks=10)
        x = torch.randn(2, 1, 128, 128)
        out = model(x)
        self.assertEqual(out.shape, (2, 1, 256, 256))

    def test_no_nan_output(self):
        model = RestoreNet(scale_factor=2, num_features=64, num_blocks=10)
        x = torch.randn(1, 1, 64, 64)
        out = model(x)
        self.assertFalse(torch.isnan(out).any())
        self.assertFalse(torch.isinf(out).any())

    def test_parameter_count(self):
        model = RestoreNet(scale_factor=2, num_features=64, num_blocks=10)
        params = count_parameters(model)
        self.assertGreater(params, 500_000)
        self.assertLess(params, 3_000_000)

    def test_scale_factor_4(self):
        model = RestoreNet(scale_factor=4, num_features=64, num_blocks=10)
        x = torch.randn(1, 1, 64, 64)
        out = model(x)
        self.assertEqual(out.shape, (1, 1, 256, 256))

    def test_pixel_shuffle_upsample(self):
        model = RestoreNet(scale_factor=2, num_features=64, num_blocks=5, upsample_mode="pixel_shuffle")
        x = torch.randn(2, 1, 64, 64)
        out = model(x)
        self.assertEqual(out.shape, (2, 1, 128, 128))
        self.assertFalse(torch.isnan(out).any())


if __name__ == "__main__":
    unittest.main()
