"""Standalone inference script for KLA Hackathon evaluation.

Restores high-resolution ground truth images from low-resolution / noisy inputs (.npy).
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn

# Ensure project root is in sys.path
script_dir = Path(__file__).resolve().parent
kla_dir = script_dir / "kla-image-restoration"
for p in [str(script_dir), str(kla_dir)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.models.restorenet import RestoreNet


def parse_args():
    parser = argparse.ArgumentParser(description="RestoreNet Inference Pipeline")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing input .npy files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save restored .npy outputs")
    parser.add_argument("--model_path", type=str, default="checkpoints/best_model.pt", help="Path to model checkpoint (.pt)")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Inference device")
    parser.add_argument("--batch_size", type=int, default=1, help="Inference batch size")
    parser.add_argument("--use_compile", action="store_true", help="Use torch.compile for optimization")
    parser.add_argument("--save_dtype", type=str, default="float32", choices=["float32", "uint8"], help="Output dtype")
    parser.add_argument("--verbose", action="store_true", help="Print verbose per-image timing")
    parser.add_argument("--max_images", type=int, default=None, help="Optional limit on number of images to process")
    return parser.parse_args()


def load_model(model_path: str, device: torch.device) -> nn.Module:
    """Load RestoreNet model weights safely."""
    model = RestoreNet(scale_factor=2, num_features=64, num_blocks=10)

    # Check both relative to cwd and relative to kla-image-restoration
    resolved_path = None
    for candidate in [model_path, f"kla-image-restoration/{model_path}"]:
        if os.path.exists(candidate):
            resolved_path = candidate
            break

    if resolved_path:
        try:
            ckpt = torch.load(resolved_path, map_location=device)
            if isinstance(ckpt, dict) and "model_state" in ckpt:
                state_dict = ckpt["model_state"]
            else:
                state_dict = ckpt

            # Handle DataParallel prefix if present
            cleaned_state = {}
            for k, v in state_dict.items():
                if k.startswith("module."):
                    cleaned_state[k[7:]] = v
                else:
                    cleaned_state[k] = v

            model.load_state_dict(cleaned_state, strict=False)
            print(f"Successfully loaded checkpoint from {resolved_path}")
        except Exception as e:
            print(f"Warning: Failed to load weights from {resolved_path} ({e}). Using initialized weights.")
    else:
        print(f"Warning: Model path {model_path} not found. Running with initialized weights.")

    model.to(device)
    model.eval()
    return model


def main():
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # 1. Validate paths
    if not input_dir.exists():
        print(f"Error: input_dir does not exist: {input_dir}")
        sys.exit(1)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. Find and sort .npy files using fast scan
    input_files = sorted([
        input_dir / f for f in os.listdir(input_dir) if f.endswith(".npy")
    ])

    if not input_files:
        print(f"Error: No .npy files found in {input_dir}")
        sys.exit(1)

    if args.max_images and args.max_images < len(input_files):
        input_files = input_files[: args.max_images]

    # Resolve device
    target_device = args.device
    if target_device == "cuda" and not torch.cuda.is_available():
        target_device = "cpu"
    device = torch.device(target_device)
    print(f"Running inference on {len(input_files)} images using device: {device}")

    # 3. Load model
    model = load_model(args.model_path, device)

    # 4. Optional compilation
    if args.use_compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("Model compiled with torch.compile(mode='reduce-overhead')")
        except Exception as e:
            print(f"Compilation skipped: {e}")

    # 5. Process files with inference_mode
    timings: List[float] = []
    total_start = time.time()

    with torch.inference_mode():
        for idx, file_path in enumerate(input_files):
            t0 = time.perf_counter()

            # a. Load as float32 numpy array (DO NOT clip on load)
            img_np = np.load(file_path).astype(np.float32)

            # b. Add batch and channel dimensions: [1, 1, H, W]
            tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).to(device)

            # c. Forward pass
            out_tensor = model(tensor)

            # d. Extract to numpy
            out_np = out_tensor.squeeze().cpu().numpy()

            # e. Clip ONLY at save time to [0, 1]
            out_clipped = np.clip(out_np, 0.0, 1.0)
            if args.save_dtype == "uint8":
                out_to_save = (out_clipped * 255.0).round().astype(np.uint8)
            else:
                out_to_save = out_clipped.astype(np.float32)

            # f. Save output
            out_file = output_dir / file_path.name
            np.save(out_file, out_to_save)

            dt_ms = (time.perf_counter() - t0) * 1000.0
            timings.append(dt_ms)

            if args.verbose:
                print(f"[{idx+1}/{len(input_files)}] {file_path.name} -> {dt_ms:.2f} ms")

    total_time = time.time() - total_start
    avg_ms = float(np.mean(timings))
    std_ms = float(np.std(timings))
    fps = len(input_files) / max(1e-6, total_time)

    # 6. Summary table
    print("\n" + "=" * 65)
    print("                     INFERENCE SUMMARY REPORT                     ")
    print("=" * 65)
    print(f"  Total Images Processed : {len(input_files):,}")
    print(f"  Total Wall Clock Time  : {total_time:.2f} s")
    print(f"  Avg Latency per Image  : {avg_ms:6.2f} ms ± {std_ms:.2f} ms")
    print(f"  Throughput             : {fps:6.2f} images/sec")
    print(f"  Saved Output Directory : {output_dir}")
    print("=" * 65)

    sys.exit(0)


if __name__ == "__main__":
    main()
