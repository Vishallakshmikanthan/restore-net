"""Submission archive packaging script for KLA Hackathon."""

import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def create_submission():
    print("=" * 65)
    print("         Packaging RestoreNet Final Submission Archive           ")
    print("=" * 65)

    # 1. Run dry_run.py programmatically
    print("\nStep 1: Running dry-run validation suite...")
    dry_run_script = project_root / "scripts" / "dry_run.py"
    if not dry_run_script.exists():
        dry_run_script = project_root / "kla-image-restoration" / "scripts" / "dry_run.py"

    res = subprocess.run([sys.executable, str(dry_run_script)], capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[ERROR] Dry-run failed. Cannot create submission archive.\n{res.stdout}\n{res.stderr}")
        sys.exit(1)
    print("[PASS] Dry-run validation succeeded.")

    # 2. Verify checkpoint exists
    print("\nStep 2: Verifying best_model checkpoint...")
    ckpt_path = project_root / "checkpoints" / "best_model.pt"
    if not ckpt_path.exists():
        ckpt_path = project_root / "kla-image-restoration" / "checkpoints" / "best_model.pt"

    if not ckpt_path.exists():
        print(f"[ERROR] Model checkpoint not found at {ckpt_path}. Run training first.")
        sys.exit(1)
    print(f"[PASS] Found model checkpoint: {ckpt_path} ({os.path.getsize(ckpt_path) / (1024*1024):.2f} MB)")

    # 3. Create timestamped archive
    timestamp = time.strftime("%Y%m%d_%H%M")
    archive_name = f"kla_submission_{timestamp}.tar.gz"
    archive_path = project_root / archive_name

    source_base = project_root
    if (project_root / "kla-image-restoration" / "src").exists():
        source_base = project_root / "kla-image-restoration"

    files_to_pack = [
        ("README.md", project_root / "README.md"),
        ("inference.py", project_root / "inference.py"),
        ("requirements.txt", project_root / "requirements.txt"),
        ("pyproject.toml", project_root / "pyproject.toml"),
        ("configs", source_base / "configs"),
        ("src", source_base / "src"),
        ("scripts/train.py", source_base / "scripts" / "train.py"),
        ("scripts/evaluate.py", source_base / "scripts" / "evaluate.py"),
        ("scripts/benchmark.py", source_base / "scripts" / "benchmark.py"),
        ("scripts/dry_run.py", source_base / "scripts" / "dry_run.py"),
        ("checkpoints/best_model.pt", ckpt_path),
    ]

    optional_files = [
        ("solution_presentation.pptx", project_root / "solution_presentation.pptx"),
        ("results/metrics/results_summary.json", source_base / "results" / "metrics" / "results_summary.json"),
    ]

    print(f"\nStep 3: Creating archive {archive_name}...")
    with tarfile.open(archive_path, "w:gz") as tar:
        for arcname, fpath in files_to_pack:
            if fpath.exists():
                tar.add(fpath, arcname=arcname)
                print(f"  + Added: {arcname}")
            else:
                print(f"  - Missing required: {arcname} ({fpath})")

        for arcname, fpath in optional_files:
            if fpath.exists():
                tar.add(fpath, arcname=arcname)
                print(f"  + Added (Optional): {arcname}")

    archive_size_mb = os.path.getsize(archive_path) / (1024 * 1024)
    print(f"\nStep 4: Archive size: {archive_size_mb:.2f} MB")
    assert archive_size_mb < 500.0, f"Archive exceeds 500MB limit: {archive_size_mb:.2f} MB"

    # 4. Verify test extraction in temporary directory
    print("\nStep 5: Verifying archive integrity via clean extraction...")
    with tempfile.TemporaryDirectory() as temp_dir:
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(temp_dir)

        extracted_inf = Path(temp_dir) / "inference.py"
        assert extracted_inf.exists(), "inference.py not found in extracted root!"
        # Check syntax
        res_compile = subprocess.run([sys.executable, "-m", "py_compile", str(extracted_inf)], capture_output=True)
        assert res_compile.returncode == 0, f"Extracted inference.py failed syntax check: {res_compile.stderr}"
        print("[PASS] Archive verified: cleanly extracted and inference.py is valid.")

    print("\n" + "=" * 65)
    print(f"[SUCCESS] Submission archive created: {archive_name}")
    print("  Upload to KLA hackathon portal and verify receipt email.")
    print("=" * 65)


if __name__ == "__main__":
    create_submission()
