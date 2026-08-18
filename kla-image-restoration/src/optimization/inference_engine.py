import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure project root is in sys.path when running standalone
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import torch
import torch.nn as nn


class OptimizedInferenceEngine:
    """Optimized inference execution engine supporting PyTorch 2.x compilation and CUDA caching."""

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        use_compile: bool = False,
        batch_size: int = 4,
    ):
        self.target_device = device
        self.device = torch.device(
            device if torch.cuda.is_available() and device == "cuda" else "cpu"
        )
        self.model = model.to(self.device)
        self.model.eval()
        self.batch_size = batch_size
        self.use_compile = use_compile

        # PyTorch 2.x compilation
        if use_compile and hasattr(torch, "compile") and int(torch.__version__.split(".")[0]) >= 2:
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                print("InferenceEngine: compiled model with torch.compile")
            except Exception as e:
                print(f"InferenceEngine: compilation failed ({e}), using eager mode")

        # Warmup forward passes
        self._warmup(n_warmup=5)

    def _warmup(self, n_warmup: int = 5):
        """Warm up CUDA kernels and JIT caches."""
        dummy = torch.randn(1, 1, 128, 128, device=self.device)
        with torch.inference_mode():
            for _ in range(n_warmup):
                _ = self.model(dummy)
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    def preprocess(self, npy_path: str) -> torch.Tensor:
        """Load .npy as float32 tensor (no clipping on load)."""
        arr = np.load(npy_path).astype(np.float32)
        tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
        return tensor

    def infer_single(self, tensor: torch.Tensor) -> torch.Tensor:
        """Run forward pass under inference_mode and return CPU tensor."""
        tensor = tensor.to(self.device)
        with torch.inference_mode():
            out = self.model(tensor)
        return out.cpu()

    def postprocess(self, tensor: torch.Tensor) -> np.ndarray:
        """Extract tensor to numpy, clip to [0, 1] range as float32."""
        arr = tensor.squeeze().numpy()
        return np.clip(arr, 0.0, 1.0).astype(np.float32)

    def process_directory(self, input_dir: str, output_dir: str, max_files: int = None) -> Dict[str, List[float]]:
        """Process all .npy files in a directory and measure timing breakdown."""
        in_path = Path(input_dir)
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        files = sorted([in_path / f for f in os.listdir(in_path) if f.endswith(".npy")])
        if max_files:
            files = files[:max_files]

        timings = {
            "disk_io": [],
            "inference": [],
            "postprocess": [],
            "total": [],
        }

        for file_path in files:
            t_start = time.perf_counter()

            # 1. IO Preprocess
            t_io_start = time.perf_counter()
            tensor = self.preprocess(str(file_path))
            t_io = (time.perf_counter() - t_io_start) * 1000.0

            # 2. Inference
            t_inf_start = time.perf_counter()
            out_tensor = self.infer_single(tensor)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            t_inf = (time.perf_counter() - t_inf_start) * 1000.0

            # 3. Postprocess and Save
            t_post_start = time.perf_counter()
            out_np = self.postprocess(out_tensor)
            np.save(out_path / file_path.name, out_np)
            t_post = (time.perf_counter() - t_post_start) * 1000.0

            t_total = (time.perf_counter() - t_start) * 1000.0

            timings["disk_io"].append(t_io)
            timings["inference"].append(t_inf)
            timings["postprocess"].append(t_post)
            timings["total"].append(t_total)

        # Print Benchmark Table
        print("\n" + "=" * 65)
        print("                  INFERENCE LATENCY BREAKDOWN                  ")
        print("=" * 65)
        print(f"  Stage          | Mean Latency (ms) | Std Dev (ms)")
        print("-" * 65)
        for stage, latencies in timings.items():
            mean_l = float(np.mean(latencies))
            std_l = float(np.std(latencies))
            print(f"  {stage.capitalize():14s} | {mean_l:14.2f} ms | {std_l:8.2f} ms")
        print("=" * 65)

        return timings

    def benchmark(self, n_images: int = 50, image_size: Tuple[int, int] = (128, 128)) -> Dict[str, float]:
        """Benchmark inference engine on synthetic random inputs."""
        latencies = []
        dummy = torch.randn(1, 1, *image_size, device=self.device)

        with torch.inference_mode():
            for _ in range(n_images):
                t0 = time.perf_counter()
                _ = self.model(dummy)
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                dt = (time.perf_counter() - t0) * 1000.0
                latencies.append(dt)

        mean_ms = float(np.mean(latencies))
        std_ms = float(np.std(latencies))
        throughput = 1000.0 / mean_ms if mean_ms > 0 else 0.0

        print(f"Benchmark ({n_images} runs): {mean_ms:.2f} ms ± {std_ms:.2f} ms ({throughput:.2f} img/s)")
        return {
            "mean_ms": mean_ms,
            "std_ms": std_ms,
            "throughput": throughput,
        }


if __name__ == "__main__":
    from src.models.restorenet import RestoreNet
    model = RestoreNet()
    engine = OptimizedInferenceEngine(model, device="cpu")
    engine.benchmark(n_images=10, image_size=(128, 128))
