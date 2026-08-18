#!/usr/bin/env python3
"""
Dataset Inspection Script for KLA Image Restoration Dataset.

Analyzes Ground Truth (GT) and NoisyLR image directories (.npy files),
computes statistics over sample pairs, verifies pair matching, and prints
a structured validation report.
"""

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect GT and NoisyLR dataset directories and report statistics."
    )
    parser.add_argument(
        "--gt_dir",
        type=str,
        required=True,
        help="Path to Ground Truth (GT) directory containing .npy files.",
    )
    parser.add_argument(
        "--noisylr_dir",
        type=str,
        required=True,
        help="Path to NoisyLR directory containing .npy files.",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=50,
        help="Number of matched pairs to randomly sample for detailed statistics (default: 50).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42).",
    )
    return parser.parse_args()


def get_npy_files(directory: Path) -> Dict[str, Path]:
    """Returns a dict mapping stem (filename without extension) to Path for all .npy files."""
    if not directory.exists() or not directory.is_dir():
        return {}
    return {f.stem: f for f in directory.glob("*.npy") if f.is_file()}


def inspect_dataset(
    gt_dir_str: str, noisylr_dir_str: str, sample_size: int = 50, seed: int = 42
) -> int:
    gt_dir = Path(gt_dir_str)
    noisylr_dir = Path(noisylr_dir_str)

    print("=" * 70)
    print("           KLA IMAGE RESTORATION DATASET INSPECTION REPORT")
    print("=" * 70)
    print(f"GT Directory      : {gt_dir.resolve() if gt_dir.exists() else gt_dir}")
    print(f"NoisyLR Directory : {noisylr_dir.resolve() if noisylr_dir.exists() else noisylr_dir}")
    print(f"Sample Size       : {sample_size}")
    print("-" * 70)

    # 1. List .npy files
    gt_files = get_npy_files(gt_dir)
    noisylr_files = get_npy_files(noisylr_dir)

    gt_count = len(gt_files)
    noisylr_count = len(noisylr_files)

    # Match pairs by stem
    matched_stems = sorted(list(set(gt_files.keys()) & set(noisylr_files.keys())))
    num_matched = len(matched_stems)

    # 4. Exit with code 1 if no pairs are found
    if num_matched == 0:
        print("\n[ERROR] No matched .npy pairs found between GT and NoisyLR directories!")
        print(f"  GT .npy files found     : {gt_count}")
        print(f"  NoisyLR .npy files found: {noisylr_count}")
        print("=" * 70)
        return 1

    # 2. Random sample of pairs
    rng = random.Random(seed)
    sampled_stems = rng.sample(matched_stems, min(sample_size, num_matched))

    gt_shapes: Set[Tuple[int, ...]] = set()
    gt_dtypes: Set[str] = set()
    gt_mins, gt_maxs, gt_means, gt_stds = [], [], [], []

    noisy_shapes: Set[Tuple[int, ...]] = set()
    noisy_dtypes: Set[str] = set()
    noisy_mins, noisy_maxs, noisy_means, noisy_stds = [], [], [], []

    for stem in sampled_stems:
        # Load GT without clipping, ensure float32
        gt_raw = np.load(gt_files[stem])
        gt_arr = gt_raw.astype(np.float32)
        gt_shapes.add(gt_arr.shape)
        gt_dtypes.add(str(gt_arr.dtype))
        gt_mins.append(float(np.min(gt_arr)))
        gt_maxs.append(float(np.max(gt_arr)))
        gt_means.append(float(np.mean(gt_arr)))
        gt_stds.append(float(np.std(gt_arr)))

        # Load NoisyLR without clipping, ensure float32
        noisy_raw = np.load(noisylr_files[stem])
        noisy_arr = noisy_raw.astype(np.float32)
        noisy_shapes.add(noisy_arr.shape)
        noisy_dtypes.add(str(noisy_arr.dtype))
        noisy_mins.append(float(np.min(noisy_arr)))
        noisy_maxs.append(float(np.max(noisy_arr)))
        noisy_means.append(float(np.mean(noisy_arr)))
        noisy_stds.append(float(np.std(noisy_arr)))

    # Global min/max across sampled images
    overall_gt_min = min(gt_mins)
    overall_gt_max = max(gt_maxs)
    avg_gt_mean = float(np.mean(gt_means))
    avg_gt_std = float(np.mean(gt_stds))
    all_in_01 = (overall_gt_min >= 0.0) and (overall_gt_max <= 1.0)

    overall_noisy_min = min(noisy_mins)
    overall_noisy_max = max(noisy_maxs)
    avg_noisy_mean = float(np.mean(noisy_means))
    avg_noisy_std = float(np.mean(noisy_stds))
    has_negative_values = overall_noisy_min < 0.0
    has_values_above_1 = overall_noisy_max > 1.0

    # 3. Print structured report
    print("\n--- GT IMAGES ---")
    print(f"  Count               : {gt_count}")
    print(f"  Sampled Analyzed    : {len(sampled_stems)}")
    print(f"  Unique Shapes       : {sorted(list(gt_shapes))}")
    print(f"  Data Types          : {sorted(list(gt_dtypes))}")
    print(f"  Value Range (Min/Max): [{overall_gt_min:.6f}, {overall_gt_max:.6f}]")
    print(f"  Mean across sample  : {avg_gt_mean:.6f}")
    print(f"  Std across sample   : {avg_gt_std:.6f}")
    print(f"  All in [0, 1] Flag  : {all_in_01} {'(PASS)' if all_in_01 else '(FAIL - values outside [0, 1])'}")

    print("\n--- NOISYLR IMAGES ---")
    print(f"  Count               : {noisylr_count}")
    print(f"  Sampled Analyzed    : {len(sampled_stems)}")
    print(f"  Unique Shapes       : {sorted(list(noisy_shapes))}")
    print(f"  Data Types          : {sorted(list(noisy_dtypes))}")
    print(f"  Value Range (Min/Max): [{overall_noisy_min:.6f}, {overall_noisy_max:.6f}]")
    print(f"  Mean across sample  : {avg_noisy_mean:.6f}")
    print(f"  Std across sample   : {avg_noisy_std:.6f}")
    print(f"  Has Negative Values : {has_negative_values}")
    print(f"  Has Values Above 1  : {has_values_above_1}")

    print("\n--- PAIR VALIDATION ---")
    print(f"  GT Files Total      : {gt_count}")
    print(f"  NoisyLR Files Total : {noisylr_count}")
    print(f"  Matched Pairs       : {num_matched}")
    equal_counts = (gt_count == noisylr_count == num_matched)
    print(f"  Counts Equal & Match: {equal_counts}")
    if not equal_counts:
        missing_in_noisy = set(gt_files.keys()) - set(noisylr_files.keys())
        missing_in_gt = set(noisylr_files.keys()) - set(gt_files.keys())
        if missing_in_noisy:
            print(f"  Missing in NoisyLR  : {len(missing_in_noisy)} files (e.g., {list(missing_in_noisy)[:3]})")
        if missing_in_gt:
            print(f"  Missing in GT       : {len(missing_in_gt)} files (e.g., {list(missing_in_gt)[:3]})")

    print("\n" + "=" * 70)
    print("Dataset inspection completed successfully.")
    print("=" * 70)
    return 0


def main() -> None:
    args = parse_args()
    exit_code = inspect_dataset(
        gt_dir_str=args.gt_dir,
        noisylr_dir_str=args.noisylr_dir,
        sample_size=args.sample_size,
        seed=args.seed,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
