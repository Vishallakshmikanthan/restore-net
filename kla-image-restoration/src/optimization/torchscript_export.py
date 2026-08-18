"""TorchScript and torch.compile model export and verification utilities."""

import os
import sys
from pathlib import Path
from typing import Tuple
import torch
import torch.nn as nn

# Ensure project root is in sys.path
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def compile_model(model: nn.Module, mode: str = "reduce-overhead") -> nn.Module:
    """Compile PyTorch model using torch.compile with warmup."""
    major_version = int(torch.__version__.split(".")[0])
    if major_version < 2 or not hasattr(torch, "compile"):
        raise RuntimeError(f"torch.compile requires PyTorch >= 2.0. Installed version: {torch.__version__}")

    compiled = torch.compile(model, mode=mode)
    # Warmup pass
    dummy = torch.randn(1, 1, 128, 128)
    with torch.no_grad():
        _ = compiled(dummy)
    return compiled


def export_torchscript(
    model: nn.Module,
    save_path: str,
    example_input_shape: Tuple[int, int, int, int] = (1, 1, 128, 128),
) -> Path:
    """Trace and export a model to TorchScript (.pt format), verifying correctness."""
    model.eval()
    dummy_input = torch.randn(*example_input_shape)

    with torch.no_grad():
        traced_model = torch.jit.trace(model, dummy_input)

    out_path = Path(save_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    traced_model.save(str(out_path))

    # Verify loaded model matches original
    loaded = load_torchscript(str(out_path))
    with torch.no_grad():
        orig_out = model(dummy_input)
        loaded_out = loaded(dummy_input)
        max_diff = (orig_out - loaded_out).abs().max().item()

    assert max_diff < 1e-4, f"Export verification failed: max difference is {max_diff}"

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"Exported TorchScript model to {out_path} ({size_mb:.2f} MB)")
    print(f"Verification output diff: {max_diff:.2e} (tolerance: 1e-4)")
    return out_path


def load_torchscript(path: str) -> nn.Module:
    """Load a TorchScript serialized model."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"TorchScript file not found at: {path}")
    return torch.jit.load(path)


def compare_outputs(original_model: nn.Module, exported_model: nn.Module, n_tests: int = 10):
    """Verify that original and exported models produce numerically identical outputs."""
    original_model.eval()
    exported_model.eval()

    with torch.no_grad():
        for i in range(n_tests):
            x = torch.randn(1, 1, 128, 128)
            out_orig = original_model(x)
            out_exp = exported_model(x)
            diff = (out_orig - out_exp).abs().max().item()
            assert diff < 1e-4, f"Mismatch at test {i+1}: max diff {diff:.2e}"

    print(f"[PASS] Compiled/exported model outputs match original within tolerance across {n_tests} tests.")


if __name__ == "__main__":
    from src.models.baseline import BaselineRestorationCNN
    model = BaselineRestorationCNN(scale_factor=2, num_features=64, num_blocks=3)
    out_file = export_torchscript(model, "checkpoints/baseline_traced.pt")
    loaded = load_torchscript(str(out_file))
    compare_outputs(model, loaded)
