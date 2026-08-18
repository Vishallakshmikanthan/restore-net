"""Evaluation script computing full quantitative metrics (PSNR, SSIM, LPIPS) against Ground Truth."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import torch

# Ensure project root is in sys.path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.training.metrics import compute_lpips, compute_psnr, compute_ssim


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate restoration predictions against Ground Truth.")
    parser.add_argument("--gt_dir", type=str, required=True, help="Directory containing GT .npy files")
    parser.add_argument("--pred_dir", type=str, required=True, help="Directory containing predicted .npy files")
    parser.add_argument("--baseline_dir", type=str, default="results/baseline_outputs", help="Optional baseline outputs directory for comparison")
    parser.add_argument("--output_json", type=str, default="results/metrics/results_summary.json", help="Path to save output JSON metrics")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Compute device for LPIPS")
    parser.add_argument("--verbose", action="store_true", help="Print per-image metrics")
    parser.add_argument("--max_images", type=int, default=None, help="Limit number of evaluated images")
    return parser.parse_args()


def evaluate_folder(gt_dir: Path, pred_dir: Path, device: str = "cpu", max_images: int = None, verbose: bool = False) -> Dict[str, Any]:
    """Evaluate predictions in pred_dir against GT files in gt_dir."""
    gt_files = {f[:-4]: gt_dir / f for f in os.listdir(gt_dir) if f.endswith(".npy")}
    pred_files = {f[:-4]: pred_dir / f for f in os.listdir(pred_dir) if f.endswith(".npy")}

    common_stems = sorted(list(set(gt_files.keys()) & set(pred_files.keys())))
    if not common_stems:
        print(f"Warning: No matching files found between {gt_dir} and {pred_dir}")
        return {}

    if max_images and max_images < len(common_stems):
        common_stems = common_stems[:max_images]

    per_image_results = []
    psnr_scores = []
    ssim_scores = []
    gt_tensors = []
    pred_tensors = []

    for stem in common_stems:
        gt_arr = np.load(gt_files[stem]).astype(np.float32)
        pred_arr = np.load(pred_files[stem]).astype(np.float32)

        # Clip [0, 1] for evaluation
        gt_c = np.clip(gt_arr, 0.0, 1.0)
        pred_c = np.clip(pred_arr, 0.0, 1.0)

        psnr_val = compute_psnr(pred_c, gt_c)
        ssim_val = compute_ssim(pred_c, gt_c)

        psnr_scores.append(psnr_val)
        ssim_scores.append(ssim_val)

        item = {
            "image": stem,
            "psnr": psnr_val,
            "ssim": ssim_val,
        }
        per_image_results.append(item)

        if verbose:
            print(f"[{stem}] PSNR: {psnr_val:6.2f} dB | SSIM: {ssim_val:.4f}")

        # Gather tensors for batched LPIPS
        gt_t = torch.from_numpy(gt_c).unsqueeze(0).unsqueeze(0)
        pred_t = torch.from_numpy(pred_c).unsqueeze(0).unsqueeze(0)
        gt_tensors.append(gt_t)
        pred_tensors.append(pred_t)

    # Batched LPIPS computation
    lpips_scores = []
    batch_size = 16
    for i in range(0, len(gt_tensors), batch_size):
        b_gt = torch.cat(gt_tensors[i : i + batch_size], dim=0)
        b_pred = torch.cat(pred_tensors[i : i + batch_size], dim=0)
        score = compute_lpips(b_pred, b_gt, device=device)
        lpips_scores.extend([score] * b_gt.shape[0])

    for i, item in enumerate(per_image_results):
        item["lpips"] = lpips_scores[i] if i < len(lpips_scores) else 0.0

    summary = {
        "num_images": len(common_stems),
        "psnr_mean": float(np.mean(psnr_scores)),
        "psnr_std": float(np.std(psnr_scores)),
        "ssim_mean": float(np.mean(ssim_scores)),
        "ssim_std": float(np.std(ssim_scores)),
        "lpips_mean": float(np.mean(lpips_scores)) if lpips_scores else 0.0,
        "lpips_std": float(np.std(lpips_scores)) if lpips_scores else 0.0,
    }

    return {"per_image": per_image_results, "summary": summary}


def print_results_table(summary: Dict[str, Any], title: str = "EVALUATION RESULTS"):
    print("\n" + "+" + "-" * 45 + "+")
    print(f"|{title.center(45)}|")
    print("+" + "-" * 12 + "+" + "-" * 32 + "+")
    print("|  Metric    |   Mean        ±  Std           |")
    print("+" + "-" * 12 + "+" + "-" * 32 + "+")
    print(f"|  PSNR      |  {summary['psnr_mean']:6.2f} dB    ± {summary['psnr_std']:5.2f}          |")
    print(f"|  SSIM      |  {summary['ssim_mean']:7.4f}     ± {summary['ssim_std']:6.4f}         |")
    print(f"|  LPIPS     |  {summary['lpips_mean']:7.4f}     ± {summary['lpips_std']:6.4f}         |")
    print("+" + "-" * 12 + "+" + "-" * 32 + "+\n")


def main():
    args = parse_args()

    gt_dir = Path(args.gt_dir)
    pred_dir = Path(args.pred_dir)
    baseline_dir = Path(args.baseline_dir)

    target_device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"

    print(f"Evaluating predictions from {pred_dir} against GT {gt_dir}...")
    main_results = evaluate_folder(
        gt_dir,
        pred_dir,
        device=target_device,
        max_images=args.max_images,
        verbose=args.verbose,
    )

    if not main_results:
        print("Evaluation could not be performed.")
        sys.exit(1)

    print_results_table(main_results["summary"], title="RESTORENET EVALUATION RESULTS")

    # Baseline comparison if available
    baseline_results = None
    if baseline_dir.exists():
        print(f"Evaluating Baseline comparison from {baseline_dir}...")
        baseline_results = evaluate_folder(
            gt_dir,
            baseline_dir,
            device=target_device,
            max_images=args.max_images,
            verbose=False,
        )
        if baseline_results and "summary" in baseline_results:
            print_results_table(baseline_results["summary"], title="BASELINE CNN EVALUATION RESULTS")

    # Save to JSON
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    json_payload = {
        "restorenet": main_results,
        "baseline": baseline_results,
    }

    with open(output_path, "w") as f:
        json.dump(json_payload, f, indent=2)

    print(f"Results summary successfully saved to {output_path}")


if __name__ == "__main__":
    main()
