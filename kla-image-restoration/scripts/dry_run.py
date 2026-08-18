"""End-to-end dry run verification smoke test for KLA Hackathon submission."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
kla_dir = project_root
for p in [str(project_root), str(project_root.parent)]:
    if p not in sys.path:
        sys.path.insert(0, p)


def resolve_file(rel_path: str) -> Path:
    candidates = [project_root / rel_path, project_root.parent / rel_path]
    for p in candidates:
        if p.exists():
            return p
    return None


def run_dry_run():
    print("=" * 65)
    print("       RestoreNet Submission Smoke Test & Dry-Run Suite       ")
    print("=" * 65)

    ready_items = []
    pending_items = []

    # 1. Check all required files
    required_files = [
        "inference.py",
        "scripts/train.py",
        "src/models/restorenet.py",
        "src/data/dataset.py",
        "src/data/augmentation.py",
        "src/data/split.py",
        "configs/train.yaml",
        "requirements.txt",
        "README.md",
    ]
    all_files_ok = True
    for f in required_files:
        found_path = resolve_file(f)
        if not found_path:
            print(f"  [FAIL] Missing required file: {f}")
            all_files_ok = False
        else:
            print(f"  [OK] Found {f}")

    if all_files_ok:
        ready_items.append("Required source files and data modules present")
    else:
        pending_items.append("Some required source files are missing")

    # 2. Check src/training/validation.py is not empty
    val_file = resolve_file("src/training/validation.py")
    if val_file and val_file.stat().st_size > 50:
        print(f"  [OK] Found non-empty validation module ({val_file.stat().st_size} bytes)")
        ready_items.append("Validation module implemented")
    else:
        print("  [WARN] src/training/validation.py is empty or missing")
        pending_items.append("src/training/validation.py is empty or missing")

    # 3. Check trained checkpoint
    best_model = resolve_file("checkpoints/best_model.pt")
    if best_model and best_model.stat().st_size > 1000:
        print(f"  [OK] Trained checkpoint found at {best_model}")
        ready_items.append(f"Trained model checkpoint ({best_model.name})")
    else:
        print("  [WARNING] checkpoints/best_model.pt not found or empty (training pending)")
        pending_items.append("Trained checkpoint (checkpoints/best_model.pt)")

    # 4. Check dependencies & imports
    imports_to_test = ["torch", "numpy", "scipy", "yaml"]
    imports_ok = True
    for mod in imports_to_test:
        try:
            __import__(mod)
            print(f"  [OK] Successfully imported {mod}")
        except ImportError as e:
            print(f"  [FAIL] Import error for {mod}: {e}")
            imports_ok = False

    if imports_ok:
        ready_items.append("Core Python dependencies verified")
    else:
        pending_items.append("Core Python dependencies missing")

    # 5. Device check
    import torch
    if torch.cuda.is_available():
        print(f"  [OK] CUDA Available: {torch.cuda.get_device_name(0)}")
    else:
        print("  [NOTICE] CUDA not available, using CPU mode for testing.")

    # 6. Run inference smoke test on 3 dummy images
    test_dir = project_root / "dry_run_test"
    out_dir = project_root / "dry_run_outputs"
    test_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        print("  [OK] Generating dummy unclipped .npy test images...")
        for i in range(3):
            dummy_arr = (np.random.rand(128, 128) * 1.45 - 0.05).astype(np.float32)
            np.save(test_dir / f"test_{i:03d}.npy", dummy_arr)

        inf_script = resolve_file("inference.py")
        cmd = [
            sys.executable,
            str(inf_script),
            "--input_dir", str(test_dir),
            "--output_dir", str(out_dir),
            "--device", "cpu",
        ]
        if best_model:
            cmd.extend(["--model_path", str(best_model)])

        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"  [FAIL] Inference failed:\n{res.stderr}")
            pending_items.append("Inference smoke test execution")
        else:
            out_files = list(out_dir.glob("*.npy"))
            if len(out_files) == 3:
                all_valid = True
                for f in out_files:
                    arr = np.load(f)
                    if arr.min() < 0.0 or arr.max() > 1.0:
                        all_valid = False
                    if np.isnan(arr).any() or np.isinf(arr).any():
                        all_valid = False
                if all_valid:
                    print("  [OK] Inference pipeline & [0, 1] output bounds verified")
                    ready_items.append("Inference CLI pipeline and bounds validation")
                else:
                    pending_items.append("Inference output bounds validation")
            else:
                pending_items.append("Inference output file count mismatch")

    finally:
        shutil.rmtree(test_dir, ignore_errors=True)
        shutil.rmtree(out_dir, ignore_errors=True)

    # 7. Print quantitative evaluation metrics if available
    metrics_file = resolve_file("results/metrics/results_summary.json")
    if metrics_file:
        try:
            with open(metrics_file, "r") as f:
                metrics_data = json.load(f)
            summary = metrics_data.get("restorenet", {}).get("summary", {})
            if summary:
                print("\n" + "-" * 65)
                print("LATEST EVALUATION METRICS (from results_summary.json):")
                print(f"  PSNR  : {summary.get('psnr_mean', 0.0):6.2f} dB (± {summary.get('psnr_std', 0.0):.2f})")
                print(f"  SSIM  : {summary.get('ssim_mean', 0.0):6.4f}   (± {summary.get('ssim_std', 0.0):.4f})")
                print(f"  LPIPS : {summary.get('lpips_mean', 0.0):6.4f}   (± {summary.get('lpips_std', 0.0):.4f})")
                print("-" * 65)
                ready_items.append(f"Evaluation metrics recorded (PSNR: {summary.get('psnr_mean', 0.0):.2f} dB)")
        except Exception as e:
            print(f"  [NOTICE] Could not parse results_summary.json: {e}")

    # 8. Summary checklist
    print("\n" + "=" * 65)
    print("                    DRY-RUN STATUS REPORT                    ")
    print("=" * 65)

    print("\n  [READY]")
    for item in ready_items:
        print(f"    - {item}")

    if pending_items:
        print("\n  [PENDING]")
        for item in pending_items:
            print(f"    - {item}")
    else:
        print("\n  [ALL CHECKS PASSED - PRODUCTION READY]")

    print("=" * 65 + "\n")
    return 0 if not pending_items else 1


if __name__ == "__main__":
    sys.exit(run_dry_run())
