"""Unit tests for restoration loss functions."""

import unittest
import torch

from src.training.losses import CharbonnierLoss, RestorationLoss, SSIM


class TestSSIM(unittest.TestCase):
    def test_identical_images(self):
        ssim = SSIM()
        img = torch.rand(2, 1, 64, 64)
        score = ssim(img, img)
        self.assertAlmostEqual(score.item(), 1.0, places=3)

    def test_range(self):
        ssim = SSIM()
        img1 = torch.rand(2, 1, 64, 64)
        img2 = torch.rand(2, 1, 64, 64)
        score = ssim(img1, img2)
        self.assertGreaterEqual(score.item(), -1.0)
        self.assertLessEqual(score.item(), 1.0)

    def test_different_images(self):
        ssim = SSIM()
        img = torch.ones(2, 1, 64, 64)
        zeros = torch.zeros(2, 1, 64, 64)
        score = ssim(img, zeros)
        self.assertLess(score.item(), 0.5)


class TestCharbonnierLoss(nn_module_test := unittest.TestCase):
    def test_charbonnier(self):
        loss_fn = CharbonnierLoss(eps=0.01)
        pred = torch.rand(2, 1, 32, 32)
        target = torch.rand(2, 1, 32, 32)
        loss = loss_fn(pred, target)
        self.assertGreater(loss.item(), 0.0)


class TestRestorationLoss(unittest.TestCase):
    def test_loss_is_positive(self):
        loss_fn = RestorationLoss(lambda_pixel=1.0, lambda_ssim=0.3, lambda_lpips=0.0)
        pred = torch.rand(2, 1, 64, 64)
        target = torch.rand(2, 1, 64, 64)
        total_loss, loss_dict = loss_fn(pred, target)
        self.assertGreater(total_loss.item(), 0.0)
        self.assertIn("l1", loss_dict)
        self.assertIn("ssim_loss", loss_dict)
        self.assertIn("lpips", loss_dict)

    def test_loss_zero_on_identical(self):
        loss_fn = RestorationLoss(lambda_pixel=1.0, lambda_ssim=0.3, lambda_lpips=0.0)
        img = torch.rand(2, 1, 64, 64)
        total_loss, loss_dict = loss_fn(img, img)
        self.assertLess(total_loss.item(), 0.01)


if __name__ == "__main__":
    unittest.main()
