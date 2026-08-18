"""Complete Training Engine for RestoreNet."""

import glob
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.training.losses import RestorationLoss
from src.training.metrics import MetricsTracker, compute_psnr, compute_ssim


class Trainer:
    """Trainer orchestrator for Restoration models."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        config: Dict[str, Any],
        device: str = "cuda",
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or {}

        # Resolve device
        target_device = device or self.config.get("device", "cuda")
        self.device = torch.device(
            target_device if torch.cuda.is_available() and target_device == "cuda" else "cpu"
        )
        self.model = self.model.to(self.device)

        # Loss configuration
        loss_cfg = self.config.get("loss", {})
        self.criterion = RestorationLoss(
            lambda_pixel=loss_cfg.get("lambda_pixel", 1.0),
            lambda_ssim=loss_cfg.get("lambda_ssim", 0.3),
            lambda_lpips=loss_cfg.get("lambda_lpips", 0.1),
            device=str(self.device),
        ).to(self.device)

        # Training hyperparameters
        training_cfg = self.config.get("training", {})
        self.epochs = training_cfg.get("epochs", 100)
        self.lr = float(training_cfg.get("learning_rate", 1e-3))
        self.betas = tuple(training_cfg.get("betas", [0.9, 0.999]))
        self.weight_decay = float(training_cfg.get("weight_decay", 0.0))
        self.grad_clip = float(training_cfg.get("gradient_clip", 1.0))

        # Optimizer & Scheduler
        self.optimizer = Adam(
            self.model.parameters(),
            lr=self.lr,
            betas=self.betas,
            weight_decay=self.weight_decay,
        )
        scheduler_cfg = training_cfg.get("scheduler", {})
        t_max = scheduler_cfg.get("T_max", self.epochs)
        eta_min = float(scheduler_cfg.get("eta_min", 1e-6))
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=t_max, eta_min=eta_min)

        # Mixed precision (AMP)
        mp_cfg = self.config.get("mixed_precision", {})
        self.use_amp = bool(mp_cfg.get("enabled", False)) and self.device.type == "cuda"
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # Metrics Tracker
        self.tracker = MetricsTracker()

        # Directories
        self.checkpoint_dir = Path(self.config.get("checkpoint_dir", "./checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.log_dir = Path(self.config.get("log_dir", "./logs"))
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Logging / TensorBoard
        logging_cfg = self.config.get("logging", {})
        self.log_every_n_batches = logging_cfg.get("log_every_n_batches", 50)
        self.use_tb = bool(logging_cfg.get("tensorboard", False))
        self.writer = None
        if self.use_tb:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=str(self.log_dir))
            except Exception as e:
                print(f"Tensorboard SummaryWriter not initialized: {e}")

        # Checkpointing settings
        ckpt_cfg = self.config.get("checkpointing", {})
        self.save_every_n_epochs = ckpt_cfg.get("save_every_n_epochs", 5)
        self.keep_last_n = ckpt_cfg.get("keep_last_n", 3)
        self.patience = ckpt_cfg.get("patience", 20)

        self.best_val_psnr = -float("inf")
        self.best_epoch = 0
        self.saved_checkpoints = []

    def train_epoch(self, epoch: int) -> float:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, (noisylr, gt) in enumerate(self.train_loader):
            noisylr = noisylr.to(self.device)
            gt = gt.to(self.device)

            self.optimizer.zero_grad()

            if self.use_amp:
                with torch.cuda.amp.autocast():
                    pred = self.model(noisylr)
                    loss, loss_dict = self.criterion(pred, gt)
                self.scaler.scale(loss).backward()
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                pred = self.model(noisylr)
                loss, loss_dict = self.criterion(pred, gt)
                loss.backward()
                if self.grad_clip > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
                self.optimizer.step()

            batch_loss = loss.item()
            total_loss += batch_loss
            num_batches += 1

            if (batch_idx + 1) % self.log_every_n_batches == 0:
                print(f"Epoch [{epoch:03d}/{self.epochs:03d}] Batch [{batch_idx+1}/{len(self.train_loader)}] Loss: {batch_loss:.4f}")

        avg_loss = total_loss / max(1, num_batches)
        if self.writer is not None:
            self.writer.add_scalar("Train/Loss", avg_loss, epoch)
            self.writer.add_scalar("Train/LR", self.optimizer.param_groups[0]["lr"], epoch)
        return avg_loss

    def val_epoch(self, epoch: int) -> Dict[str, float]:
        """Run validation on the validation set."""
        self.model.eval()
        psnr_list = []
        ssim_list = []

        with torch.no_grad():
            for noisylr, gt in self.val_loader:
                noisylr = noisylr.to(self.device)
                pred = self.model(noisylr)

                pred_np = pred.detach().cpu().numpy()
                gt_np = gt.detach().cpu().numpy()

                for i in range(pred_np.shape[0]):
                    p = np.clip(pred_np[i, 0], 0.0, 1.0)
                    g = np.clip(gt_np[i, 0], 0.0, 1.0)
                    psnr_list.append(compute_psnr(p, g))
                    ssim_list.append(compute_ssim(p, g))

        mean_psnr = float(np.mean(psnr_list)) if psnr_list else 0.0
        mean_ssim = float(np.mean(ssim_list)) if ssim_list else 0.0

        if self.writer is not None:
            self.writer.add_scalar("Val/PSNR", mean_psnr, epoch)
            self.writer.add_scalar("Val/SSIM", mean_ssim, epoch)

        return {"val_psnr": mean_psnr, "val_ssim": mean_ssim}

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint dictionary and manage rolling history."""
        ckpt = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "config": self.config,
        }

        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(ckpt, best_path)

        periodic_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(ckpt, periodic_path)
        self.saved_checkpoints.append(periodic_path)

        # Keep only last keep_last_n periodic checkpoints
        while len(self.saved_checkpoints) > self.keep_last_n:
            old_ckpt = self.saved_checkpoints.pop(0)
            if old_ckpt.exists() and old_ckpt.name != "best_model.pt":
                try:
                    old_ckpt.unlink()
                except Exception:
                    pass

    def load_checkpoint(self, path: str) -> int:
        """Load checkpoint and restore model/optimizer states."""
        ckpt = torch.load(path, map_location=self.device)
        if isinstance(ckpt, dict) and "model_state" in ckpt:
            self.model.load_state_dict(ckpt["model_state"])
            if "optimizer_state" in ckpt and self.optimizer:
                self.optimizer.load_state_dict(ckpt["optimizer_state"])
            start_epoch = ckpt.get("epoch", 0)
            print(f"Loaded checkpoint from {path} at epoch {start_epoch}")
            return start_epoch
        else:
            self.model.load_state_dict(ckpt)
            print(f"Loaded raw model weights from {path}")
            return 0

    def fit(self):
        """Execute full training and evaluation loop."""
        print(f"Beginning training on device {self.device} for {self.epochs} epochs...")
        start_time = time.time()
        epochs_no_improve = 0

        for epoch in range(1, self.epochs + 1):
            epoch_start = time.time()
            train_loss = self.train_epoch(epoch)
            val_metrics = self.val_epoch(epoch)
            self.scheduler.step()

            val_psnr = val_metrics["val_psnr"]
            val_ssim = val_metrics["val_ssim"]
            is_best = val_psnr > self.best_val_psnr

            if is_best:
                self.best_val_psnr = val_psnr
                self.best_epoch = epoch
                epochs_no_improve = 0
                self.save_checkpoint(epoch, val_metrics, is_best=True)
            else:
                epochs_no_improve += 1

            if epoch % self.save_every_n_epochs == 0:
                self.save_checkpoint(epoch, val_metrics, is_best=False)

            elapsed = time.time() - epoch_start
            best_tag = " (NEW BEST)" if is_best else ""
            print(
                f"Epoch {epoch:03d}/{self.epochs:03d} | Train Loss: {train_loss:.4f} | "
                f"Val PSNR: {val_psnr:6.2f} dB | Val SSIM: {val_ssim:.4f} | Time: {elapsed:.2f}s{best_tag}"
            )

            # Early stopping
            if epochs_no_improve >= self.patience:
                print(f"Early stopping triggered after {self.patience} epochs without improvement.")
                break

        total_time = time.time() - start_time
        if self.writer is not None:
            self.writer.close()

        print("=" * 60)
        print(f"Training Complete! Total Time: {total_time / 60:.2f} minutes")
        print(f"Best Val PSNR: {self.best_val_psnr:6.2f} dB at epoch {self.best_epoch}")
        print(f"Best model saved to {self.checkpoint_dir / 'best_model.pt'}")
        print("=" * 60)
