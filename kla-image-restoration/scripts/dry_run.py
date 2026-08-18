"""End-to-end dry run verification smoke test for KLA Hackathon submission."""

import os
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def check_file(rel_path: str) -> bool:
    candidates = [project_root / rel_path, project_root / "kla-image-restoration" / rel_path]
    for p in candidates:
        if p.exists():
            return True
    return False


def run_dry_run():
    print("=" * 65)
    print("       RestoreNet Submission Smoke Test & Dry-Run Suite       ")
    print("=" * 65)

    checklist = {}

    # 1. Check all required files
    required_files = [
        "inference.py",
        "scripts/train.py",
        "src/models/restorenet.py",
        "src/data/dataset.py",
        "configs/train.yaml",
        "requirements.txt",
        "README.md",
    ]
    all_files_ok = True
    for f in required_files:
        if not check_file(f):
            print(f"[FAIL] Missing required file: {f}")
            all_files_ok = False
        else:
            print(f"  [OK] Found {f}")
    checklist["Required Files"] = all_files_ok

    # 2. Check imports
    imports_to_test = ["torch", "numpy", "scipy", "skimage", "yaml", "tqdm"]
    imports_ok = True
    for mod in imports_to_test:
        try:
            __import__(mod)
            print(f"  [OK] Successfully imported {mod}")
        except ImportError as e:
            print(f"[FAIL] Import error for {mod}: {e}")
            imports_ok = False
    checklist["Dependencies"] = imports_ok

    # 3. Check CUDA availability
    import torch
    if torch.cuda.is_available():
        print(f"  [OK] CUDA Available: {torch.cuda.get_device_name(0)}")
    else:
        print("  [NOTICE] CUDA not available, using CPU mode for testing.")

    # 4. Create temp test directory
    test_dir = project_root / "dry_run_test"
    out_dir = project_root / "dry_run_outputs"
    test_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Generate 10 dummy .npy files in [-0.05, 1.4] range
        print("  [OK] Generating 10 dummy unclipped .npy test images...")
        for i in range(10):
            dummy_arr = (np.random.rand(128, 128) * 1.45 - 0.05).astype(np.float32)
            np.save(test_dir / f"test_{i:03d}.npy", dummy_arr)

        # 5. Run inference.py programmatically
        inf_script = project_root / "inference.py"
        if not inf_script.exists():
            inf_script = project_root / "kla-image-restoration" / "inference.py"

        cmd = [
            sys.executable,
            str(inf_script),
            "--input_dir", str(test_dir),
            "--output_dir", str(out_dir),
            "--device", "cpu",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"[FAIL] Inference failed with return code {res.returncode}:\n{res.stderr}")
            checklist["Inference Execution"] = False
        else:
            checklist["Inference Execution"] = True

        # 6. Verify output counts
        out_files = list(out_dir.glob("*.npy"))
        checklist["Output File Count"] = (len(out_files) == 10)
        print(f"  [OK] Verified output file count: {len(out_files)}/10")

        # 7 & 8. Verify outputs are loadable, float32/bounded in [0, 1]
        all_valid = True
        for f in out_files:
            arr = np.load(f)
            if arr.min() < 0.0 or arr.max() > 1.0:
                print(f"[FAIL] Output values out of bounds [0, 1]: [{arr.min()}, {arr.max()}] in {f.name}")
                all_valid = False
            if np.isnan(arr).any() or np.isinf(arr).any():
                print(f"[FAIL] Output contains NaN/Inf: {f.name}")
                all_valid = False
        checklist["Output Validations [0,1]"] = all_valid

    finally:
        # 11. Cleanup temp dirs
        shutil.rmtree(test_dir, ignore_errors=True)
        shutil.rmtree(out_dir, ignore_errors=True)

    # 12. Final Checklist Output
    print("\n" + "=" * 65)
    print("                    FINAL SUBMISSION CHECKLIST                    ")
    print("=" * 65)
    all_passed = True
    for item, status in checklist.items():
        sym = "[PASS]" if status else "[FAIL]"
        if not status:
            all_passed = False
        print(f"  {sym:<6} {item}")

    print("=" * 65)
    if all_passed:
        print("  *** READY FOR SUBMISSION ***")
        return 0
    else:
        print("  *** SUBMISSION VERIFICATION FAILED ***")
        return 1


if __name__ == "__main__":
    sys.exit(run_dry_run())
