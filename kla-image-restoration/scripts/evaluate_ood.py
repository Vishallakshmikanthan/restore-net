"""Out-of-Distribution (OOD) Robustness and Stress Testing Evaluation."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import scipy.ndimage
import torch

# Ensure project root is in sys.path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data.augmentation import SyntheticDegradationAugmentor
from src.models.restorenet import RestoreNet
from src.training.metrics import compute_lpips, compute_psnr, compute_ssim


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate model on Out-Of-Distribution (OOD) degraded images.")
    parser.add_argument("--gt_dir", type=str, default="data/GT", help="Path to clean GT images")
    parser.add_argument("--model_path", type=str, default="checkpoints/best_model.pt", help="Path to model weights")
    parser.add_argument("--output_dir", type=str, default="results/metrics", help="Directory to save OOD report")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Inference device")
    parser.add_argument("--max_images", type=int, default=10, help="Max GT images to test")
    return parser.parse_args()


def load_model(model_path: str, device: torch.device):
    model = RestoreNet(scale_factor=2, num_features=64, num_blocks=10)
    if os.path.exists(model_path):
        try:
            ckpt = torch.load(model_path, map_location=device)
            state_dict = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
            model.load_state_dict(state_dict, strict=False)
            print(f"Loaded weights from {model_path}")
        except Exception as e:
            print(f"Notice: Loading weights failed ({e}). Using initialized model.")
    else:
        print(f"Notice: Model {model_path} not found. Using initialized model.")
    model.to(device)
    model.eval()
    return model


def resize_to_match(arr: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Bilinearly resample array to match target 2D shape."""
    if arr.shape == target_shape:
        return arr
    zoom_factors = (target_shape[0] / arr.shape[0], target_shape[1] / arr.shape[1])
    return scipy.ndimage.zoom(arr, zoom=zoom_factors, order=1).astype(np.float32)


def main():
    args = parse_args()

    gt_path = Path(args.gt_dir)
    target_device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    device = torch.device(target_device)
    print(f"Running OOD robustness evaluation on device: {device}")

    model = load_model(args.model_path, device)

    # Aggressive OOD Augmentor
    ood_augmentor = SyntheticDegradationAugmentor(
        noise_std_range=(0.05, 0.15),
        speckle_range=(0.3, 2.0),
        downsample_factors=(3, 4, 6),
    )

    gt_files = sorted([gt_path / f for f in os.listdir(gt_path) if f.endswith(".npy")])
    if args.max_images:
        gt_files = gt_files[: args.max_images]

    if not gt_files:
        print(f"No GT files found in {gt_path}")
        sys.exit(1)

    print(f"Testing OOD degradation on {len(gt_files)} images (3 OOD variants each)...")

    ood_psnr_list = []
    ood_ssim_list = []
    ood_lpips_list = []

    with torch.inference_mode():
        for f in gt_files:
            gt_arr = np.load(f).astype(np.float32)
            gt_c = np.clip(gt_arr, 0.0, 1.0)
            gt_tensor = torch.from_numpy(gt_c).unsqueeze(0).unsqueeze(0).to(device)

            for _ in range(3):
                ood_input = ood_augmentor.generate_synthetic_pair(gt_arr)
                input_tensor = torch.from_numpy(ood_input).unsqueeze(0).unsqueeze(0).to(device)

                pred_tensor = model(input_tensor)
                pred_c = np.clip(pred_tensor.squeeze().cpu().numpy(), 0.0, 1.0)

                # Match spatial dimension to GT if downsampling factor differed from 2
                pred_matched = resize_to_match(pred_c, gt_c.shape)

                ood_psnr_list.append(compute_psnr(pred_matched, gt_c))
                ood_ssim_list.append(compute_ssim(pred_matched, gt_c))

                pred_matched_t = torch.from_numpy(pred_matched).unsqueeze(0).unsqueeze(0)
                ood_lpips_list.append(compute_lpips(pred_matched_t, gt_tensor.cpu(), device="cpu"))

    # Sanity Check: Pure uniform random noise [0, 1]
    noise_psnr_list = []
    noise_ssim_list = []
    noise_lpips_list = []

    with torch.inference_mode():
        for _ in range(5):
            noise_img = np.random.uniform(0.0, 1.0, size=(128, 128)).astype(np.float32)
            noise_tensor = torch.from_numpy(noise_img).unsqueeze(0).unsqueeze(0).to(device)

            pred_tensor = model(noise_tensor)
            pred_c = np.clip(pred_tensor.squeeze().cpu().numpy(), 0.0, 1.0)

            # Compare against 256x256 upsampled noise target
            noise_target = resize_to_match(noise_img, pred_c.shape)
            noise_target_t = torch.from_numpy(noise_target).unsqueeze(0).unsqueeze(0)

            noise_psnr_list.append(compute_psnr(pred_c, noise_target))
            noise_ssim_list.append(compute_ssim(pred_c, noise_target))
            noise_lpips_list.append(compute_lpips(pred_tensor.cpu(), noise_target_t, device="cpu"))

    ood_summary = {
        "ood_psnr_mean": float(np.mean(ood_psnr_list)),
        "ood_psnr_std": float(np.std(ood_psnr_list)),
        "ood_ssim_mean": float(np.mean(ood_ssim_list)),
        "ood_ssim_std": float(np.std(ood_ssim_list)),
        "ood_lpips_mean": float(np.mean(ood_lpips_list)),
        "ood_lpips_std": float(np.std(ood_lpips_list)),
    }

    noise_summary = {
        "noise_psnr_mean": float(np.mean(noise_psnr_list)),
        "noise_ssim_mean": float(np.mean(noise_ssim_list)),
        "noise_lpips_mean": float(np.mean(noise_lpips_list)),
    }

    # Load in-distribution metrics if available
    in_dist_summary_file = Path("results/metrics/results_summary.json")
    in_dist_psnr = None
    if in_dist_summary_file.exists():
        try:
            with open(in_dist_summary_file, "r") as f:
                d = json.load(f)
                r = d.get("restorenet", {}).get("summary", {})
                in_dist_psnr = r.get("psnr_mean")
        except Exception:
            pass

    # Print Report
    print("\n" + "=" * 65)
    print("                     OOD ROBUSTNESS REPORT                        ")
    print("=" * 65)
    if in_dist_psnr is not None:
        delta = ood_summary["ood_psnr_mean"] - in_dist_psnr
        print(f"  In-Distribution PSNR   : {in_dist_psnr:6.2f} dB")
        print(f"  OOD Synthetic PSNR     : {ood_summary['ood_psnr_mean']:6.2f} dB ± {ood_summary['ood_psnr_std']:.2f}")
        print(f"  OOD delta PSNR         : {delta:+6.2f} dB vs in-distribution")
    else:
        print(f"  OOD Synthetic PSNR     : {ood_summary['ood_psnr_mean']:6.2f} dB ± {ood_summary['ood_psnr_std']:.2f}")

    print(f"  OOD Synthetic SSIM     : {ood_summary['ood_ssim_mean']:7.4f} ± {ood_summary['ood_ssim_std']:.4f}")
    print(f"  OOD Synthetic LPIPS    : {ood_summary['ood_lpips_mean']:7.4f} ± {ood_summary['ood_lpips_std']:.4f}")
    print(f"  Pure Noise PSNR (Worst): {noise_summary['noise_psnr_mean']:6.2f} dB")
    print("=" * 65)

    # Save to JSON
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "ood_results.json"

    report_payload = {
        "ood_metrics": ood_summary,
        "noise_metrics": noise_summary,
        "in_distribution_psnr": in_dist_psnr,
    }

    with open(report_path, "w") as f:
        json.dump(report_payload, f, indent=2)

    print(f"OOD report saved to {report_path}")


if __name__ == "__main__":
    main()
