"""Generate synthetic NoisyLR pairs from GT images."""
import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data.augmentation import SyntheticDegradationAugmentor


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic NoisyLR pairs from GT images.")
    parser.add_argument("--gt_dir", required=True, help="Path to GT images directory")
    parser.add_argument("--output_dir", required=True, help="Path to output directory for synthetic pairs")
    parser.add_argument("--samples_per_image", type=int, default=2, help="Number of synthetic samples per GT image")
    args = parser.parse_args()

    augmentor = SyntheticDegradationAugmentor()
    count = augmentor.augment_dataset(args.gt_dir, args.output_dir, args.samples_per_image)
    print(f"Done. Generated {count} synthetic pairs.")


if __name__ == "__main__":
    main()
