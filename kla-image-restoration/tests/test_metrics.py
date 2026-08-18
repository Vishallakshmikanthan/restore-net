"""Unit tests for restoration metrics."""

import unittest
import numpy as np
import torch

from src.training.metrics import (
    MetricsTracker,
    compute_all_metrics,
    compute_lpips,
    compute_psnr,
    compute_ssim,
)


class TestMetrics(unittest.TestCase):
    def test_psnr_identical(self):
        arr = np.random.rand(64, 64).astype(np.float32)
        score = compute_psnr(arr, arr)
        self.assertGreater(score, 50.0)

    def test_ssim_identical(self):
        arr = np.random.rand(64, 64).astype(np.float32)
        score = compute_ssim(arr, arr)
        self.assertAlmostEqual(score, 1.0, places=3)

    def test_psnr_noisy(self):
        arr = np.random.rand(64, 64).astype(np.float32)
        noisy = np.clip(arr + np.random.normal(0, 0.1, arr.shape), 0.0, 1.0).astype(np.float32)
        clean_psnr = compute_psnr(arr, arr)
        noisy_psnr = compute_psnr(noisy, arr)
        self.assertLess(noisy_psnr, clean_psnr)

    def test_compute_all_metrics(self):
        pred = torch.rand(2, 1, 32, 32)
        gt = torch.rand(2, 1, 32, 32)
        res = compute_all_metrics(pred, gt, device="cpu")
        self.assertIn("psnr", res)
        self.assertIn("ssim", res)
        self.assertIn("lpips", res)
        self.assertIsInstance(res["psnr"], float)
        self.assertIsInstance(res["ssim"], float)
        self.assertIsInstance(res["lpips"], float)

    def test_metrics_tracker(self):
        tracker = MetricsTracker()
        for i in range(10):
            tracker.update(psnr=30.0 + i, ssim=0.9, lpips=0.1)
        summary = tracker.summary()
        self.assertAlmostEqual(summary["psnr"], 34.5, places=2)
        self.assertAlmostEqual(summary["ssim"], 0.9, places=3)
        self.assertAlmostEqual(summary["lpips"], 0.1, places=3)
        self.assertIn("PSNR", tracker.log_string())


if __name__ == "__main__":
    unittest.main()
