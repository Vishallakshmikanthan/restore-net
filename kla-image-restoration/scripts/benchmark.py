"""Hardware latency and throughput benchmarking script for RestoreNet."""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import torch
import torch.nn as nn

# Ensure project root is in sys.path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.baseline import BaselineRestorationCNN
from src.models.restorenet import RestoreNet


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark model inference runtime.")
    parser.add_argument("--model_path", type=str, default="checkpoints/best_model.pt", help="Path to model weights")
    parser.add_argument("--device", type=str, default="cuda", help="Target execution device")
    parser.add_argument("--image_size", type=int, default=128, help="Spatial dimension of input tensor")
    parser.add_argument("--warmup_runs", type=int, default=10, help="Number of warmup forward passes")
    parser.add_argument("--benchmark_runs", type=int, default=50, help="Number of benchmark iterations")
    parser.add_argument("--output_json", type=str, default="results/benchmarks/benchmark_results.json", help="Destination JSON path")
    return parser.parse_args()


def load_benchmark_model(model_path: str, device: torch.device) -> nn.Module:
    model = RestoreNet(scale_factor=2, num_features=64, num_blocks=10)
    if os.path.exists(model_path):
        try:
            ckpt = torch.load(model_path, map_location=device)
            state_dict = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
            model.load_state_dict(state_dict, strict=False)
            print(f"Loaded model weights from {model_path}")
        except Exception as e:
            print(f"Warning: Could not load weights ({e}). Benchmarking with initialized weights.")
    else:
        print(f"Warning: {model_path} not found. Benchmarking with initialized weights.")
    model.to(device)
    model.eval()
    return model


def benchmark_mode(
    model: nn.Module,
    dummy_input: torch.Tensor,
    device: torch.device,
    warmup_runs: int,
    benchmark_runs: int,
    mode_name: str,
    use_inference_mode: bool = True,
) -> Dict[str, float]:
    """Benchmark a specific execution configuration."""
    # Warmup
    for _ in range(warmup_runs):
        if use_inference_mode:
            with torch.inference_mode():
                _ = model(dummy_input)
        else:
            with torch.no_grad():
                _ = model(dummy_input)

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Benchmark timed iterations
    latencies = []
    for _ in range(benchmark_runs):
        t0 = time.perf_counter()
        if use_inference_mode:
            with torch.inference_mode():
                _ = model(dummy_input)
        else:
            with torch.no_grad():
                _ = model(dummy_input)

        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000.0
        latencies.append(dt)

    mean_ms = float(np.mean(latencies))
    std_ms = float(np.std(latencies))
    min_ms = float(np.min(latencies))
    max_ms = float(np.max(latencies))
    p95_ms = float(np.percentile(latencies, 95))
    throughput = (1000.0 / mean_ms) if mean_ms > 0 else 0.0

    return {
        "mode": mode_name,
        "mean_ms": mean_ms,
        "std_ms": std_ms,
        "min_ms": min_ms,
        "max_ms": max_ms,
        "p95_ms": p95_ms,
        "throughput": throughput,
    }


def main():
    args = parse_args()

    # Resolve device
    target_device = args.device
    if target_device == "cuda" and not torch.cuda.is_available():
        target_device = "cpu"
    device = torch.device(target_device)
    print(f"Benchmarking on device: {device}")

    model = load_benchmark_model(args.model_path, device)
    dummy_input = torch.randn(1, 1, args.image_size, args.image_size, device=device)

    results = {}

    # 1. Eager Mode (torch.inference_mode)
    print("Testing Eager Mode (torch.inference_mode)...")
    res_eager = benchmark_mode(
        model, dummy_input, device, args.warmup_runs, args.benchmark_runs,
        mode_name="Eager (inference_mode)", use_inference_mode=True,
    )
    results["eager_inference_mode"] = res_eager

    # 2. Eager Mode (torch.no_grad)
    print("Testing Eager Mode (torch.no_grad)...")
    res_no_grad = benchmark_mode(
        model, dummy_input, device, args.warmup_runs, args.benchmark_runs,
        mode_name="Eager (no_grad)", use_inference_mode=False,
    )
    results["eager_no_grad"] = res_no_grad

    # 3. torch.compile mode (if supported)
    if hasattr(torch, "compile") and int(torch.__version__.split(".")[0]) >= 2:
        try:
            print("Testing torch.compile(mode='reduce-overhead')...")
            compiled_model = torch.compile(model, mode="reduce-overhead")
            res_compile = benchmark_mode(
                compiled_model, dummy_input, device, args.warmup_runs, args.benchmark_runs,
                mode_name="torch.compile", use_inference_mode=True,
            )
            results["torch_compile"] = res_compile
        except Exception as e:
            print(f"torch.compile skipped ({e})")

    # Print summary table
    print("\n" + "=" * 70)
    print(f"{'Mode':<26} | {'Mean (ms)':<10} | {'Std':<6} | {'P95 (ms)':<9} | {'Throughput':<12}")
    print("-" * 70)
    best_ms = float("inf")
    for key, val in results.items():
        mean_v = val["mean_ms"]
        std_v = val["std_ms"]
        p95_v = val["p95_ms"]
        tp_v = val["throughput"]
        best_ms = min(best_ms, mean_v)
        print(f"{val['mode']:<26} | {mean_v:9.2f}  | {std_v:5.2f} | {p95_v:8.2f}  | {tp_v:6.2f} img/s")
    print("=" * 70)

    target_check = "[PASS]" if best_ms < 100.0 else "[NOTICE]"
    print(f"{target_check} Target: <100ms end-to-end. Current best: {best_ms:.2f} ms")

    # Save to JSON
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Benchmark results saved to {out_json}")


if __name__ == "__main__":
    main()
