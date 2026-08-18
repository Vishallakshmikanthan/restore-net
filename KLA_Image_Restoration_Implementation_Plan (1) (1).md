# KLA SEMICON India Hackathon 2026

## AI-Based Restoration of Degraded Images for Semiconductor Inspection

### Complete Implementation Guide & Technical Specification

**Status:** Master Technical Document v1.0  
**Last Updated:** August 2026  
**Target Submission:** Phase 1 Deadline (16 August 2026\)

---

## TABLE OF CONTENTS

1. [Executive Summary](#executive-summary)  
2. [Project Identity & Positioning](#project-identity--positioning)  
3. [KLA Problem Interpretation](#kla-problem-interpretation)  
4. [Official Requirements vs Assumptions](#official-requirements-vs-assumptions)  
5. [System Architecture](#system-architecture)  
6. [Mathematical Formulation](#mathematical-formulation)  
7. [Dataset Analysis & Specifications](#dataset-analysis--specifications)  
8. [Data Pipeline & Processing](#data-pipeline--processing)  
9. [Baseline Model Design](#baseline-model-design)  
10. [Candidate Architecture Evaluation](#candidate-architecture-evaluation)  
11. [Final Architecture Selection](#final-architecture-selection)  
12. [Loss Function Design](#loss-function-design)  
13. [Training Strategy](#training-strategy)  
14. [Validation & Evaluation](#validation--evaluation)  
15. [GPU Optimization](#gpu-optimization)  
16. [Inference Pipeline](#inference-pipeline)  
17. [Repository Structure](#repository-structure)  
18. [Implementation Guide](#implementation-guide)  
19. [Testing & Verification](#testing--verification)  
20. [Deployment & Demo](#deployment--demo)  
21. [Presentation Strategy](#presentation-strategy)  
22. [Risk Register & Mitigation](#risk-register--mitigation)  
23. [Development Roadmap](#development-roadmap)  
24. [Build Order](#build-order)

---

## EXECUTIVE SUMMARY

### Project Name

**RestoreNet: Degradation-Aware Fidelity-First Image Restoration**

### One-Line Pitch

*Restore signal. Preserve structure. Accelerate inspection—an AI system for semiconductor image restoration that prioritizes fidelity over hallucination.*

### Technical Innovation

A hybrid CNN-based restoration architecture combining:

- Degradation-aware feature extraction (implicit noise/resolution modeling)  
- Multi-scale residual learning for efficient upsampling  
- Fidelity-preserving loss function (pixel \+ structural \+ perceptual)  
- H100-optimized inference pipeline with \<100ms end-to-end latency  
- Out-of-distribution generalization testing

### Why It's Competitive

- **Fidelity-First Philosophy:** Avoids hallucinated details; reconstructs only information supported by the degraded observation  
- **Engineering Rigor:** Clean data pipeline, reproducible training, comprehensive ablations  
- **Practical Optimization:** Full inference stack optimized for H100 (not just model forward pass)  
- **OOD Robustness:** Validated on unfamiliar image content within the benchmark degradation family  
- **Deployment Ready:** Standalone inference script, no hardcoded paths, reproducible from scratch

### Main Risk

Training convergence and OOD generalization if architecture is too aggressive; mitigation through staged training and conservative architecture selection.

### Recommended Development Strategy

**Tier 2 (Competitive Balance)** — Quality-focused model with proven convergence, not bleeding-edge research:

1. Build bulletproof baseline (simple CNN \+ L1)  
2. Iterate loss function (add SSIM, perceptual)  
3. Scale architecture (add residuals, multi-scale)  
4. Optimize inference (torch.compile, batching)  
5. Validate OOD robustness  
6. Final polish and presentation

**Development Time:** 4-5 weeks  
**Critical Path:** Data inspection → Baseline → Final Model → Optimization → Presentation

---

## PROJECT IDENTITY & POSITIONING

### Project Positioning Matrix

| Aspect | Positioning |
| :---- | :---- |
| **Technical Category** | Image Restoration \+ Super-Resolution (Joint) |
| **Scope** | Semiconductor inspection-specific |
| **Target Metric** | Balanced: PSNR ↑ SSIM ↑ LPIPS ↓ Runtime ↓ |
| **Innovation Type** | Engineering \+ Integration (not novel architecture research) |
| **Deployment Model** | Offline batch processing on H100 |
| **User Base** | Inspection system engineers, process technicians |

### Core Value Proposition

**For KLA Judges:**

- Demonstrates deep understanding of restoration physics (degradation model, forward process)  
- Shows engineering discipline (reproducibility, testing, ablations)  
- Achieves strong metrics without inflated parameters  
- Optimizes complete pipeline (not just model accuracy)

**For Semiconductor Inspection:**

- Recovers fine structures destroyed by noise and downsampling  
- Reduces false positives in downstream defect detection  
- Maintains deterministic output (no stochasticity)  
- Runs at production throughput

---

## KLA PROBLEM INTERPRETATION

### Official Problem Statement (From KLA Materials)

**Task:**  
Given a degraded, noisy, low-resolution image (NoisyLR), restore it to match a clean, full-resolution ground-truth image (GT).

**Degradation Model:**  
Three mechanisms applied in **unknown order**:

1. **Speckle Noise** — Multiplicative; models sensor/optical artifacts  
2. **Additive Gaussian Noise** — Zero-mean; models electronic noise  
3. **Downsampling** — Reduces spatial resolution; loses high-frequency information

**Example Forward Process (Unknown Order):**

x \= clean GT image

y₁ \= D(x) or N\_G(x) or N\_S(x) — first degradation

y₂ \= apply second degradation to y₁

y \= apply third degradation to y₂

NoisyLR \= y (observed)

**Key Constraints (From KLA Webinars & Documentation):**

- GT values: normalized to \[0, 1\]  
- NoisyLR values: may extend outside \[0, 1\] (intentional feature)  
- Image sizes: approximately 256×256 or 512×512 in evaluation  
- Model does NOT need to identify degradation order explicitly  
- Evaluation includes both in-distribution and out-of-distribution content  
- **End-to-end timing** includes: disk I/O \+ preprocess \+ CPU→GPU \+ model \+ GPU→CPU \+ postprocess \+ save  
- KLA uses fixed but undisclosed weighting of PSNR, SSIM, and LPIPS

### Critical Insight: Undisclosed Degradation Order

The model should learn a **unified restoration transform** that works regardless of order.

**Implication:** A single-stage model is appropriate; explicit degradation identification is unnecessary overhead.

### Dataset Characteristics (From Test\_NoisyLR.zip Analysis)

Sample:          000000.npy to 000002.npy

Shape:           128×128 grayscale (uint8 → float32)

Dtype:           float32

Range:           \[-0.0159, 1.5406\] (outside \[0,1\] ✓)

Mean:            0.4–0.7 (typical)

Std:             0.19–0.29 (reasonable variance)

Behavior:        Some negative values; some \>1.0 (degradation artifacts)

**Interpretation:**

- NoisyLR intentionally extends outside \[0,1\]  
- Do not blindly clip or normalize during preprocessing  
- Preserve this "outside-range" information for model learning  
- Clipping/normalization applied only at output (if needed)

---

## OFFICIAL REQUIREMENTS VS ASSUMPTIONS

### A. OFFICIAL KLA REQUIREMENTS

**MANDATORY:**

| Requirement | Source | Status |
| :---- | :---- | :---- |
| Handle speckle, Gaussian, downsampling | Problem Stmt | ✓ Required |
| Generalize to OOD content | Problem Stmt | ✓ Required |
| End-to-end inference optimized | Problem Stmt | ✓ Required |
| Standalone inference script (input/output dirs) | Section C | ✓ Required |
| Reproducible training code | Section D | ✓ Required |
| PSNR, SSIM, LPIPS reported | Section D | ✓ Required |
| At least one baseline comparison | Section D | ✓ Required |
| Model weights \+ config provided | Section 4 | ✓ Required |
| README with exact commands | Section 4 | ✓ Required |
| No manual code editing by evaluators | Section C | ✓ Required |
| GitHub repository accessible | Section 4 | ✓ Required |
| Presentation PPT/PPTX | Section 5 | ✓ Required |

**ALLOWED:**

- Any architecture (CNN, transformer, hybrid, algorithm unrolling)  
- Pretrained weights (with license disclosure)  
- External datasets (with license disclosure)  
- Synthetic degradation augmentation  
- Frequency-domain methods  
- Custom loss functions

**NOT ALLOWED:**

- Using hidden test labels for training  
- Hardcoded outputs  
- Code requiring manual edits  
- Inaccessible models/datasets  
- Undisclosed external resources

---

### B. RECOMMENDED DESIGN DECISIONS (NOT KLA REQUIREMENTS)

| Decision | Rationale |
| :---- | :---- |
| Grayscale processing | Test dataset is grayscale; simpler pipeline |
| Single-stage architecture | Unknown degradation order; multi-stage adds complexity |
| Residual learning | Standard for restoration; proven convergence |
| Mixed-precision training | Faster, stable; modern PyTorch supports well |
| Synthetic augmentation | Approved by KLA; increases dataset diversity |
| H100 optimization | KLA evaluation platform; directly measurable |

---

### C. EXPERIMENTAL HYPOTHESES (OPTIONAL INNOVATIONS)

| Hypothesis | Test | If True | If False |
| :---- | :---- | :---- | :---- |
| Frequency loss improves SSIM | Add Fourier loss term | Include in final | Remove, use pixel only |
| Degradation consistency helps | Forward model \+ cycle loss | Investigate further | Skip; too complex |
| Hybrid CNN-Transformer better | Train both; compare | Use hybrid | Use CNN-only (simpler) |

---

### D. OPTIONAL FUTURE ENHANCEMENTS

- Multi-scale cascade restoration  
- Learned degradation model (implicit)  
- Attention-based selective restoration  
- Diffusion-based refinement (post-processing)  
- Ensembles (multiple models combined)

**Note:** Do NOT implement optional enhancements during Phase 1; focus on strong engineering of core model.

---

## SYSTEM ARCHITECTURE

### High-Level Pipeline

┌─────────────────────────────────────────────────────────────────────┐

│                     TRAINING PIPELINE                               │

└─────────────────────────────────────────────────────────────────────┘

Dataset (Official KLA)

    ↓

\[Data Validation & Analysis\]

    ├─ Pair GT/NoisyLR matching

    ├─ Numerical range check

    ├─ Shape uniformity

    └─ Sample visualization

    ↓

\[Train/Val Split\]

    ├─ 70% training

    ├─ 20% validation (clean)

    └─ 10% internal holdout (robustness)

    ↓

\[Augmentation Pipeline\]

    ├─ Geometric: flip, rotate, crop

    ├─ Synthetic: additional degraded pairs

    └─ Intensity: safe transformations

    ↓

\[Data Loader\]

    ├─ Patch extraction (if needed)

    ├─ Batch assembly

    └─ GPU transfer

    ↓

\[Training Loop\]

    ├─ Forward pass

    ├─ Loss computation (multi-term)

    ├─ Backward pass

    ├─ Optimizer step (Adam \+ scheduler)

    ├─ Validation checkpoint

    └─ EMA update (if used)

    ↓

\[Best Model Selection\]

    ├─ Validation PSNR/SSIM/LPIPS

    ├─ OOD robustness score

    └─ Runtime performance

    ↓

\[Checkpoint Export\]

    └─ Model weights \+ config

┌─────────────────────────────────────────────────────────────────────┐

│                     INFERENCE PIPELINE                              │

└─────────────────────────────────────────────────────────────────────┘

Input Directory (NoisyLR .npy files)

    ↓

\[Disk I/O\]

    └─ Read file to memory

    ↓

\[Preprocessing\]

    ├─ Load and verify shape

    ├─ Handle out-of-range values (NO clipping)

    ├─ Normalize if needed (model-specific)

    └─ Convert to tensor

    ↓

\[CPU→GPU Transfer\]

    └─ Pinned memory transfer

    ↓

\[Model Inference\]

    ├─ Forward pass (single or batch)

    ├─ torch.no\_grad() context

    └─ Optional: torch.inference\_mode()

    ↓

\[GPU→CPU Transfer\]

    └─ Non-blocking if possible

    ↓

\[Postprocessing\]

    ├─ Denormalize if needed

    ├─ Apply clipping (if needed)

    └─ Ensure \[0,1\] or per-spec output

    ↓

\[Saving Output\]

    ├─ Convert to appropriate dtype

    ├─ Write .npy or .png per spec

    └─ Preserve filename convention

    ↓

Output Directory (Restored images)

    ↓

\[Evaluation (KLA)\]

    ├─ Compare with hidden GT

    ├─ Compute PSNR/SSIM/LPIPS

    └─ Measure end-to-end latency

### Component Breakdown

**Data Pipeline Component:**

- `src/data/dataset.py` — GT/NoisyLR loading, validation  
- `src/data/augmentation.py` — Geometric & synthetic transformations  
- `src/data/loader.py` — Batch assembly, prefetching

**Model Component:**

- `src/models/restoration_net.py` — Core architecture  
- `src/models/blocks.py` — Residual, upsampling blocks

**Training Component:**

- `src/training/trainer.py` — Training loop  
- `src/training/losses.py` — Multi-term loss functions  
- `src/training/metrics.py` — PSNR, SSIM, LPIPS computation

**Inference Component:**

- `scripts/inference.py` — Standalone inference CLI  
- `src/inference/engine.py` — Optimized inference code

**Optimization Component:**

- `src/optimization/torchscript_export.py` — torch.compile() if beneficial  
- `src/optimization/tensorrt_export.py` — TensorRT if beneficial (benchmark first)

---

## MATHEMATICAL FORMULATION

### Degradation Forward Model

**Generic formulation (order-agnostic):**

y \= D(N\_G(N\_S(x)))  or any permutation

where:

  x ∈ ℝ^(H×W) \= clean GT image, normalized to \[0,1\]

  N\_S(·) \= Speckle degradation (multiplicative)

  N\_G(·) \= Additive Gaussian noise

  D(·) \= Bilinear/bicubic downsampling by factor ↓s

  y ∈ ℝ^((H/s)×(W/s)) \= NoisyLR, may extend outside \[0,1\]

**Speckle Model:**

N\_S(x) \= x ⊙ η,  where η \~ Gamma(α, α) or Uniform\[0.5, 1.5\]

(element-wise multiplication)

**Gaussian Noise Model:**

N\_G(x) \= x \+ ε,  where ε \~ 𝒩(0, σ²I)

**Downsampling:**

D(x) \= Bilinear\_interpolate(x, scale=1/s)

### Restoration Inverse Model

**Our learned model:**

x̂ \= G\_θ(y)

where:

  G\_θ \= neural network with parameters θ

  x̂ ∈ ℝ^(H×W) \= restored image (full resolution)

  Goal: x̂ ≈ x (match ground truth as closely as possible)

**Key Property:**

- G\_θ must learn the inverse of composite degradation  
- No explicit identification of degradation order required  
- Single-stage restoration is appropriate

### Loss Function Design

**Multi-term objective:**

L\_total \= λ\_pixel · L\_pixel \+ λ\_ssim · L\_ssim \+ λ\_lpips · L\_lpips

        \+ λ\_freq · L\_freq \+ λ\_grad · L\_grad

where λ\_\* are learned/fixed weights

#### Loss Components

**1\. Pixel Loss (L1 or Charbonnier):**

L\_pixel \= 1/N ∑\_i |x̂\_i \- x\_i| \+ ε

Rationale: Direct reconstruction error; robust to outliers (L1 \> L2)

Hyperparameter: ε \= 0.01 (Charbonnier parameter)

**2\. Structural Similarity Loss:**

L\_ssim \= 1 \- SSIM(x̂, x)

Rationale: Preserves edge alignment, local structure

Computation: Sliding window (11×11), σ=1.5

**3\. Perceptual Loss (LPIPS proxy):**

L\_lpips ≈ 1/C ∑\_layers ||F\_l(x̂) \- F\_l(x)||\_2

Rationale: Perceptually-aligned feature matching

Implementation: Use pretrained AlexNet or VGG features (freeze)

**4\. Frequency Domain Loss (Optional):**

L\_freq \= 1/N ∑\_i ||FFT(x̂\_i) \- FFT(x\_i)||\_2

Rationale: Encourages high-frequency accuracy

Note: Expensive; consider only if ablation shows benefit

**5\. Gradient/Edge Loss (Optional):**

L\_grad \= 1/N ∑\_i ||∇x̂\_i \- ∇x\_i||\_1

Rationale: Preserves edges and fine structures

Implementation: Sobel filters (precomputed)

#### Recommended Loss Weight Configuration

**Phase 1 (Conservative):**

λ\_pixel \= 1.0   (anchor term)

λ\_ssim \= 0.3

λ\_lpips \= 0.1

λ\_freq \= 0.0    (disable; expensive)

λ\_grad \= 0.0    (disable; test later)

**If ablation shows benefit:**

λ\_freq → 0.05–0.1

λ\_grad → 0.05–0.1

---

## DATASET ANALYSIS & SPECIFICATIONS

### Official KLA Dataset

**Source:**  
Google Drive: `https://drive.google.com/drive/folders/1VKiFW-kDk9-q5XRPu3nrl08OM94EwzV6`

**Structure:**

dataset/

  ├─ GT/

  │   ├─ 000000.npy

  │   ├─ 000001.npy

  │   └─ ... (paired images)

  └─ NoisyLR/

      ├─ 000000.npy

      ├─ 000001.npy

      └─ ...

**File Format:**

- `.npy` — NumPy binary format  
- Dtype: `float32`  
- Shape: Variable (typically 256×256 or 512×512)  
- Grayscale (single channel)

### Test Dataset Analysis (Test\_NoisyLR.zip)

From direct inspection:

| Property | Value |
| :---- | :---- |
| **File Count** | 400 files (000000.npy—000399.npy) |
| **Dimensions** | 128×128 pixels |
| **Dtype** | float32 |
| **Range** | \[-0.0159, 1.5406\] |
| **Mean** | 0.4–0.7 |
| **Std Dev** | 0.19–0.29 |
| **Outside \[0,1\]** | ✓ YES (both negative and \>1.0) |
| **Pattern** | Degraded images with noise \+ resolution loss |

### Data Characteristics

**Numerical Handling:**

| Aspect | Details |
| :---- | :---- |
| **Input (NoisyLR)** | Preserve as-is; do NOT clip |
| **Model Input** | Normalize if needed (architecture-dependent) |
| **Model Output** | Raw logits or normalized \[0,1\] |
| **Metric Computation** | Use officially clipped version \[0,1\] |
| **Saved Output** | Clip to \[0,1\]; save as float32 .npy or uint8 .png |

### Dataset Statistics (Estimated from Sample)

**Expected full dataset characteristics:**

Training Set (\~350 pairs):

  \- Image shapes: Mix of 128×128, 256×256, 512×512

  \- Content: Semiconductor die images (assumed)

  \- Degradation levels: Moderate noise, 2x–4x downsampling

  \- Storage: \~350 × (2 files) × \~100KB ≈ 70MB

Validation Set (\~50 pairs):

  \- Same statistics as training

  \- Clean split: no source image overlap

Hidden Test Set (unknown, \~100–200):

  \- In-distribution: Similar degradation & content

  \- Out-of-distribution: Different image classes/content

  \- Degradation mechanisms: SAME (speckle, Gaussian, downsampling)

  \- Degradation levels: VARY (within similar range)

### Important Handling Notes

**DO:**

- Load NoisyLR with values potentially outside \[0,1\]  
- Preserve negative values and values \>1.0 during preprocessing  
- Apply per-image normalization only if model requires bounded input  
- Clip output to \[0,1\] only at inference output stage  
- Save metric computation on officially clipped values

**DON'T:**

- Blindly clip NoisyLR to \[0,1\] during loading  
- Use global dataset statistics for normalization (per-image is safer)  
- Normalize GT to mean=0, std=1 during training (preserve \[0,1\] scale)  
- Save outputs outside \[0,1\] (clip at save time)

---

## DATA PIPELINE & PROCESSING

### Stage 0: Dataset Inspection

**Objective:** Understand data distribution before training.

**Script: `scripts/inspect_dataset.py`**

\#\!/usr/bin/env python3

import numpy as np

import os

from pathlib import Path

def inspect\_dataset(gt\_dir, noisylr\_dir, sample\_size=50):

    """Inspect GT and NoisyLR datasets."""

    

    gt\_files \= sorted(Path(gt\_dir).glob('\*.npy'))\[:sample\_size\]

    noisylr\_files \= sorted(Path(noisylr\_dir).glob('\*.npy'))\[:sample\_size\]

    

    print("=" \* 80\)

    print("DATASET INSPECTION REPORT")

    print("=" \* 80\)

    

    \# GT statistics

    print("\\nGT IMAGES:")

    gt\_shapes, gt\_ranges \= \[\], \[\]

    for fpath in gt\_files:

        arr \= np.load(fpath)

        gt\_shapes.append(arr.shape)

        gt\_ranges.append((arr.min(), arr.max(), arr.mean(), arr.std()))

    

    print(f"  File count (sampled): {len(gt\_files)}")

    print(f"  Unique shapes: {set(gt\_shapes)}")

    print(f"  Value range: \[{np.min(\[r\[0\] for r in gt\_ranges\]):.4f}, " 

          f"{np.max(\[r\[1\] for r in gt\_ranges\]):.4f}\]")

    print(f"  Mean (avg across samples): {np.mean(\[r\[2\] for r in gt\_ranges\]):.4f}")

    print(f"  All values in \[0,1\]? {all(r\[0\] \>= 0 and r\[1\] \<= 1 for r in gt\_ranges)}")

    

    \# NoisyLR statistics

    print("\\nNOISYLR IMAGES:")

    noisylr\_shapes, noisylr\_ranges \= \[\], \[\]

    for fpath in noisylr\_files:

        arr \= np.load(fpath)

        noisylr\_shapes.append(arr.shape)

        noisylr\_ranges.append((arr.min(), arr.max(), arr.mean(), arr.std()))

    

    print(f"  File count (sampled): {len(noisylr\_files)}")

    print(f"  Unique shapes: {set(noisylr\_shapes)}")

    print(f"  Value range: \[{np.min(\[r\[0\] for r in noisylr\_ranges\]):.4f}, "

          f"{np.max(\[r\[1\] for r in noisylr\_ranges\]):.4f}\]")

    print(f"  Mean (avg across samples): {np.mean(\[r\[2\] for r in noisylr\_ranges\]):.4f}")

    print(f"  Outside \[0,1\]? {any(r\[0\] \< 0 or r\[1\] \> 1 for r in noisylr\_ranges)}")

    print(f"  Negative values present? {any(r\[0\] \< 0 for r in noisylr\_ranges)}")

    

    \# Pair validation

    print("\\nPAIR VALIDATION:")

    pair\_count \= 0

    for gt\_file, noisylr\_file in zip(gt\_files, noisylr\_files):

        if gt\_file.stem \== noisylr\_file.stem:

            pair\_count \+= 1

    print(f"  Matched pairs: {pair\_count} / {len(gt\_files)}")

    

    print("\\n" \+ "=" \* 80\)

if \_\_name\_\_ \== '\_\_main\_\_':

    import argparse

    parser \= argparse.ArgumentParser()

    parser.add\_argument('--gt\_dir', required=True, help='GT directory')

    parser.add\_argument('--noisylr\_dir', required=True, help='NoisyLR directory')

    parser.add\_argument('--sample\_size', type=int, default=50)

    args \= parser.parse\_args()

    inspect\_dataset(args.gt\_dir, args.noisylr\_dir, args.sample\_size)

**Usage:**

python scripts/inspect\_dataset.py \--gt\_dir data/GT \--noisylr\_dir data/NoisyLR

**Expected Output:**

GT IMAGES:

  File count: 50

  Unique shapes: {(256, 256), (512, 512), (128, 128)}

  Value range: \[0.0001, 1.0000\]

  All in \[0,1\]? True

NOISYLR IMAGES:

  File count: 50

  Unique shapes: {(128, 128), (256, 256), (512, 512)}

  Value range: \[-0.0159, 1.5406\]

  Outside \[0,1\]? True

  Negative values? True

---

### Stage 1: Data Loading & Validation

**Module: `src/data/dataset.py`**

import torch

from torch.utils.data import Dataset

import numpy as np

from pathlib import Path

from typing import Tuple, Optional

class RestorationDataset(Dataset):

    """

    Paired GT/NoisyLR dataset loader.

    

    Key design decisions:

    \- Load images exactly as-is (preserve out-of-range values)

    \- Per-image normalization only if specified

    \- No automatic clipping

    \- Lazy loading (load on demand)

    """

    

    def \_\_init\_\_(self, 

                 gt\_dir: str, 

                 noisylr\_dir: str,

                 normalize: bool \= False,

                 augment: bool \= False):

        """

        Args:

            gt\_dir: Path to GT images

            noisylr\_dir: Path to NoisyLR images

            normalize: If True, normalize NoisyLR to mean=0, std=1 (per-image)

            augment: If True, apply data augmentation

        """

        self.gt\_dir \= Path(gt\_dir)

        self.noisylr\_dir \= Path(noisylr\_dir)

        self.normalize \= normalize

        self.augment \= augment

        

        \# Find all paired files

        gt\_files \= sorted(self.gt\_dir.glob('\*.npy'))

        noisylr\_files \= sorted(self.noisylr\_dir.glob('\*.npy'))

        

        \# Validate pairing

        self.pairs \= \[\]

        for gt\_file in gt\_files:

            noisylr\_file \= self.noisylr\_dir / gt\_file.name

            if noisylr\_file.exists():

                self.pairs.append((gt\_file, noisylr\_file))

        

        if len(self.pairs) \== 0:

            raise ValueError("No paired GT/NoisyLR files found")

        

        print(f"Loaded {len(self.pairs)} GT/NoisyLR pairs")

    

    def \_\_len\_\_(self) \-\> int:

        return len(self.pairs)

    

    def \_\_getitem\_\_(self, idx: int) \-\> Tuple\[torch.Tensor, torch.Tensor\]:

        gt\_path, noisylr\_path \= self.pairs\[idx\]

        

        \# Load as float32, preserve values

        gt \= np.load(gt\_path).astype(np.float32)

        noisylr \= np.load(noisylr\_path).astype(np.float32)

        

        \# Validate shapes match (or at least compatible for upsampling)

        assert gt.shape \== noisylr.shape or \\

               (gt.shape\[0\] \== 2 \* noisylr.shape\[0\] and 

                gt.shape\[1\] \== 2 \* noisylr.shape\[1\]), \\

               f"Shape mismatch: GT {gt.shape} vs NoisyLR {noisylr.shape}"

        

        \# Optional normalization (per-image, NOT global)

        if self.normalize:

            noisylr \= (noisylr \- noisylr.mean()) / (noisylr.std() \+ 1e-6)

        

        \# Optional augmentation

        if self.augment:

            gt, noisylr \= self.\_augment(gt, noisylr)

        

        \# Convert to tensor (add channel dimension)

        gt\_tensor \= torch.from\_numpy(gt\[None, ...\])  \# \[1, H, W\]

        noisylr\_tensor \= torch.from\_numpy(noisylr\[None, ...\])  \# \[1, H, W\]

        

        return noisylr\_tensor, gt\_tensor

    

    def \_augment(self, gt: np.ndarray, noisylr: np.ndarray) \-\> Tuple\[np.ndarray, np.ndarray\]:

        """Apply geometric augmentations."""

        \# Random horizontal flip

        if np.random.rand() \> 0.5:

            gt \= np.fliplr(gt)

            noisylr \= np.fliplr(noisylr)

        

        \# Random vertical flip

        if np.random.rand() \> 0.5:

            gt \= np.flipud(gt)

            noisylr \= np.flipud(noisylr)

        

        \# Random 90-degree rotation

        k \= np.random.randint(0, 4\)

        gt \= np.rot90(gt, k)

        noisylr \= np.rot90(noisylr, k)

        

        return gt, noisylr

**Key Properties:**

- Lazy loading (efficient memory)  
- No automatic clipping of out-of-range values  
- Per-image normalization if needed  
- Deterministic pair matching

---

### Stage 2: Train/Validation Split

\# src/data/split.py

def create\_train\_val\_split(dataset, train\_ratio=0.7, val\_ratio=0.2, seed=42):

    """

    Create non-leaking train/val/holdout split.

    

    Args:

        dataset: RestorationDataset

        train\_ratio: 0.7 (70%)

        val\_ratio: 0.2 (20%)

        seed: Random seed for reproducibility

    

    Returns:

        train\_dataset, val\_dataset, holdout\_dataset

    """

    np.random.seed(seed)

    torch.manual\_seed(seed)

    

    n \= len(dataset)

    indices \= np.random.permutation(n)

    

    train\_size \= int(train\_ratio \* n)

    val\_size \= int(val\_ratio \* n)

    

    train\_idx \= indices\[:train\_size\]

    val\_idx \= indices\[train\_size:train\_size \+ val\_size\]

    holdout\_idx \= indices\[train\_size \+ val\_size:\]

    

    train\_dataset \= Subset(dataset, train\_idx)

    val\_dataset \= Subset(dataset, val\_idx)

    holdout\_dataset \= Subset(dataset, holdout\_idx)

    

    return train\_dataset, val\_dataset, holdout\_dataset

---

### Stage 3: Augmentation Strategy

\# src/data/augmentation.py

class SyntheticDegradationAugmentor:

    """

    Generate additional training pairs by applying controlled degradations to GT images.

    

    IMPORTANT: This augmentation is APPROVED by KLA ("You may create extra 

    synthetic degraded pairs from the provided GT images").

    """

    

    def \_\_init\_\_(self, 

                 noise\_std\_range=(0.01, 0.05),

                 speckle\_range=(0.5, 1.5),

                 downsample\_factors=(2, 3, 4)):

        self.noise\_std\_range \= noise\_std\_range

        self.speckle\_range \= speckle\_range

        self.downsample\_factors \= downsample\_factors

    

    def apply\_speckle(self, img: np.ndarray, strength: float) \-\> np.ndarray:

        """Multiplicative speckle noise."""

        speckle \= np.random.uniform(

            low=1.0 / strength,

            high=strength,

            size=img.shape

        )

        return np.clip(img \* speckle, 0, 1\)

    

    def apply\_gaussian(self, img: np.ndarray, std: float) \-\> np.ndarray:

        """Additive Gaussian noise."""

        noise \= np.random.normal(0, std, img.shape)

        return img \+ noise  \# NOTE: Allow values outside \[0,1\]

    

    def apply\_downsample(self, img: np.ndarray, factor: int) \-\> np.ndarray:

        """Bilinear downsampling."""

        from scipy.ndimage import zoom

        return zoom(img, 1.0 / factor, order=1)  \# order=1 → bilinear

    

    def generate\_synthetic\_pair(self, gt: np.ndarray) \-\> np.ndarray:

        """

        Generate a random synthetic degradation.

        Order is randomized to avoid pattern learning.

        """

        x \= gt.copy()

        

        \# Random degradation order

        order \= np.random.permutation(\[0, 1, 2\])

        operations \= \[

            ('speckle', lambda x: self.apply\_speckle(x, 

                np.random.uniform(\*self.speckle\_range))),

            ('gaussian', lambda x: self.apply\_gaussian(x, 

                np.random.uniform(\*self.noise\_std\_range))),

            ('downsample', lambda x: self.apply\_downsample(x, 

                np.random.choice(self.downsample\_factors)))

        \]

        

        for idx in order:

            op\_name, op\_func \= operations\[idx\]

            x \= op\_func(x)

        

        return x

    

    def augment\_dataset(self, gt\_dir: str, output\_dir: str, 

                       samples\_per\_image: int \= 2):

        """

        Generate synthetic pairs and save to output directory.

        

        Usage:

            augmentor \= SyntheticDegradationAugmentor()

            augmentor.augment\_dataset('data/GT', 'data/NoisyLR\_synth', samples\_per\_image=2)

        """

        Path(output\_dir).mkdir(parents=True, exist\_ok=True)

        

        gt\_files \= sorted(Path(gt\_dir).glob('\*.npy'))

        synthetic\_count \= 0

        

        for gt\_file in gt\_files:

            gt \= np.load(gt\_file).astype(np.float32)

            

            for i in range(samples\_per\_image):

                synthetic\_noisylr \= self.generate\_synthetic\_pair(gt)

                

                \# Save with unique name

                output\_name \= f"{gt\_file.stem}\_synth\_{i}.npy"

                np.save(Path(output\_dir) / output\_name, synthetic\_noisylr)

                synthetic\_count \+= 1

        

        print(f"Generated {synthetic\_count} synthetic degraded pairs")

        return synthetic\_count

**Usage in DataLoader:**

\# Combine official \+ synthetic pairs

official\_dataset \= RestorationDataset('data/GT', 'data/NoisyLR')

synthetic\_noisylr \= 'data/NoisyLR\_synth'

\# Generate synthetic pairs

augmentor \= SyntheticDegradationAugmentor()

augmentor.augment\_dataset('data/GT', synthetic\_noisylr, samples\_per\_image=2)

\# Combine datasets

from torch.utils.data import ConcatDataset

combined\_dataset \= ConcatDataset(\[official\_dataset, synthetic\_dataset\])

**Advantage:** Increases dataset size (official \+ 2× synthetic) without leaking into validation.

---

### Stage 4: DataLoader Configuration

\# configs/data.yaml

dataset:

  gt\_dir: "${data\_root}/GT"

  noisylr\_dir: "${data\_root}/NoisyLR"

  normalize: false  \# False: preserve out-of-range values

  augment: true     \# Enable geometric augmentations

split:

  train\_ratio: 0.70

  val\_ratio: 0.20

  holdout\_ratio: 0.10

  seed: 42

loader:

  batch\_size: 8

  num\_workers: 4

  pin\_memory: true

  shuffle\_train: true

  prefetch\_factor: 2  \# Prefetch 2 batches

---

## BASELINE MODEL DESIGN

### Purpose

Establish a simple, fast-to-train baseline for comparison and sanity checking.

### Architecture

Input: \[B, 1, H\_lr, W\_lr\]

  ↓

\[Upsampling Module\]

  ├─ Bilinear upsample × 2 or 4 (factor depends on training pair downsampling)

  └─ Output: \[B, 64, H\_gt, W\_gt\]

  ↓

\[Residual CNN Block × 3\]

  ├─ Conv 3×3 (64→64)

  ├─ ReLU

  ├─ Conv 3×3 (64→64)

  ├─ Add residual connection

  └─ Output: \[B, 64, H\_gt, W\_gt\]

  ↓

\[Output Layer\]

  ├─ Conv 1×1 (64→1)

  └─ Output: \[B, 1, H\_gt, W\_gt\]

  ↓

Predicted restoration

### Implementation

\# src/models/baseline.py

import torch

import torch.nn as nn

import torch.nn.functional as F

class BaselineRestorationCNN(nn.Module):

    """

    Simple baseline: Upsample \+ residual convolutions.

    

    Design:

    \- Bilinear upsampling (deterministic, simple)

    \- 3 residual blocks (light)

    \- L1 loss (robust)

    

    Expected performance:

    \- PSNR: \~24-26 dB (on 256×256)

    \- Training time: \< 2 hours (single GPU)

    """

    

    def \_\_init\_\_(self, 

                 scale\_factor: int \= 2,

                 num\_features: int \= 64,

                 num\_blocks: int \= 3):

        super().\_\_init\_\_()

        

        self.scale\_factor \= scale\_factor

        

        \# Upsampling

        self.upsample \= nn.Upsample(scale\_factor=scale\_factor, mode='bilinear', 

                                    align\_corners=False)

        

        \# First conv (1 channel → features)

        self.conv\_in \= nn.Conv2d(1, num\_features, kernel\_size=3, padding=1)

        

        \# Residual blocks

        self.res\_blocks \= nn.ModuleList(\[

            self.\_make\_res\_block(num\_features)

            for \_ in range(num\_blocks)

        \])

        

        \# Output conv (features → 1 channel)

        self.conv\_out \= nn.Conv2d(num\_features, 1, kernel\_size=3, padding=1)

    

    def \_make\_res\_block(self, channels):

        """Simple residual block."""

        return nn.Sequential(

            nn.Conv2d(channels, channels, kernel\_size=3, padding=1),

            nn.ReLU(inplace=True),

            nn.Conv2d(channels, channels, kernel\_size=3, padding=1)

        )

    

    def forward(self, x):

        \# Upsample

        x \= self.upsample(x)

        

        \# Extract features

        feat \= self.conv\_in(x)

        

        \# Residual blocks

        for block in self.res\_blocks:

            feat \= feat \+ block(feat)  \# Residual connection

        

        \# Reconstruct

        out \= self.conv\_out(feat)

        

        \# Optional: Add residual connection from upsampled input

        out \= out \+ x

        

        return out

### Training Configuration

\# configs/baseline.yaml

model:

  name: baseline

  scale\_factor: 2  \# or 4 (infer from data)

  num\_features: 64

  num\_blocks: 3

training:

  epochs: 50

  learning\_rate: 1e-3

  optimizer: Adam

  scheduler: CosineAnnealingLR

  loss: L1

  device: cuda

checkpointing:

  save\_every\_n\_epochs: 5

  best\_metric: psnr  \# Save based on val PSNR

### Training Script

\# scripts/train\_baseline.py

import torch

import torch.nn as nn

import torch.optim as optim

from torch.utils.data import DataLoader

def train\_baseline(cfg):

    """Train baseline model."""

    

    \# Model

    model \= BaselineRestorationCNN(

        scale\_factor=cfg.model.scale\_factor,

        num\_features=cfg.model.num\_features,

        num\_blocks=cfg.model.num\_blocks

    ).to('cuda')

    

    \# Data

    from src.data.dataset import RestorationDataset

    dataset \= RestorationDataset(cfg.dataset.gt\_dir, cfg.dataset.noisylr\_dir)

    train\_loader \= DataLoader(dataset, batch\_size=cfg.training.batch\_size, 

                              shuffle=True, num\_workers=4, pin\_memory=True)

    

    \# Loss & Optimizer

    criterion \= nn.L1Loss()

    optimizer \= optim.Adam(model.parameters(), lr=cfg.training.learning\_rate)

    scheduler \= optim.lr\_scheduler.CosineAnnealingLR(optimizer, 

                                                     T\_max=cfg.training.epochs)

    

    \# Training loop

    best\_loss \= float('inf')

    for epoch in range(cfg.training.epochs):

        epoch\_loss \= 0.0

        for i, (noisylr, gt) in enumerate(train\_loader):

            noisylr \= noisylr.to('cuda')

            gt \= gt.to('cuda')

            

            \# Forward

            pred \= model(noisylr)

            loss \= criterion(pred, gt)

            

            \# Backward

            optimizer.zero\_grad()

            loss.backward()

            torch.nn.utils.clip\_grad\_norm\_(model.parameters(), max\_norm=1.0)

            optimizer.step()

            

            epoch\_loss \+= loss.item()

            

            if (i \+ 1\) % 10 \== 0:

                print(f"Epoch {epoch+1}, Batch {i+1}: Loss \= {loss.item():.4f}")

        

        \# Scheduler step

        scheduler.step()

        

        \# Save checkpoint

        if (epoch \+ 1\) % 5 \== 0:

            ckpt\_path \= f"checkpoints/baseline\_epoch\_{epoch+1}.pt"

            torch.save(model.state\_dict(), ckpt\_path)

            print(f"Saved checkpoint: {ckpt\_path}")

        

        avg\_loss \= epoch\_loss / len(train\_loader)

        print(f"Epoch {epoch+1} complete. Avg Loss \= {avg\_loss:.4f}")

    

    return model

if \_\_name\_\_ \== '\_\_main\_\_':

    import yaml

    with open('configs/baseline.yaml') as f:

        cfg \= yaml.safe\_load(f)

    train\_baseline(cfg)

### Expected Baseline Performance

**Metrics (on validation set, 256×256 images):**

PSNR: 24.5–26.0 dB

SSIM: 0.70–0.75

LPIPS: 0.20–0.25

Runtime: 50–80 ms/image (H100)

### Purpose of Baseline

- **Sanity check:** Verify data pipeline, loss computation, metrics  
- **Anchor:** Quantify improvement from final model  
- **Fallback:** If final model doesn't converge, baseline is submission-ready

---

## CANDIDATE ARCHITECTURE EVALUATION

### Architecture Families to Consider

#### Family A: CNN Restoration

**Concept:** Multi-scale residual CNN with skip connections.

**Pros:**

- Proven convergence  
- Fast inference  
- Well-understood hyperparameter tuning  
- Minimal dependencies

**Cons:**

- Limited receptive field (may struggle with large textures)  
- Requires stacking for global context

**Candidate: NAF-style Restoration (Normalized Adaptive Filters)**

Input

  ↓

\[Multi-scale feature extraction\]

  ├─ Res block (scale 1×)

  ├─ Res block (scale 2× downsampled)

  └─ Res block (scale 4× downsampled)

  ↓

\[Adaptive filtering\]

  ├─ Channel attention (CA)

  └─ Spatial attention (SA)

  ↓

\[Multi-scale reconstruction\]

  ├─ Upsample \+ merge

  └─ Residual refinement

  ↓

Output

**Expected performance:** PSNR 26–28 dB, runtime 40–60ms

---

#### Family B: Vision Transformers

**Concept:** Self-attention over image patches.

**Pros:**

- Global context from start  
- Proven for high-res tasks  
- Recent architectures (Swin, etc.)

**Cons:**

- EXPENSIVE (memory, compute)  
- Slower inference than CNN  
- Requires large training data for good performance  
- Overkill for 256×256 images?

**Risk Assessment:** TOO RISKY for hackathon—may not fit memory, may have convergence issues.

**Recommendation:** SKIP for Phase 1; keep as backup option.

---

#### Family C: Hybrid CNN-Transformer

**Concept:** Local CNN context \+ global Transformer attention.

**Pros:**

- Combines strengths of both  
- Recent papers show good results  
- Better generalization

**Cons:**

- Complex implementation  
- Higher risk of bugs  
- Slower than pure CNN

**Candidate: Simple CNN→Transformer→CNN pipeline**

Input \[B, 1, H, W\]

  ↓

\[CNN Feature Extraction: 3 residual blocks\]

  → \[B, 64, H, W\]

  ↓

\[Reshape to patches\]

  → \[B, (H/P)×(W/P), P²×64\] where P=patch\_size=4

  ↓

\[Transformer: 4 layers, 8 heads\]

  → \[B, (H/P)×(W/P), P²×64\]

  ↓

\[Reshape back to spatial\]

  → \[B, 64, H, W\]

  ↓

\[CNN Reconstruction: 3 residual blocks \+ output conv\]

  → \[B, 1, H, W\]

**Expected performance:** PSNR 27–29 dB, runtime 60–100ms

---

#### Family D: Algorithm Unrolling

**Concept:** "Unroll" iterative restoration as a neural network.

**Idea:**

x₀ \= Upsample(NoisyLR)

for k in range(K):

    x\_k \= RestoreStep(x\_{k-1}, learned\_params)

return x\_K

Each step can be learned (e.g., gradient descent with learned step size).

**Pros:**

- Interpretable (mimics classical optimization)  
- Can incorporate domain knowledge

**Cons:**

- Requires careful initialization  
- More parameters for same depth  
- Slower inference (K sequential steps)

**Recommendation:** SKIP for Phase 1; complex without clear benefit.

---

### Decision Matrix

| Criterion | CNN | Transformer | Hybrid | Unrolling |
| :---- | :---- | :---- | :---- | :---- |
| **PSNR potential** | 26–28 | 28–30 | 27–29 | 26–28 |
| **Inference speed** | 40–60ms | 150–300ms | 60–100ms | 80–120ms |
| **Memory footprint** | Low | High | Medium | Medium |
| **Implementation risk** | Low | High | Medium | Medium |
| **Hacka hackathon fit** | Excellent | Poor | Good | Fair |
| **OOD generalization** | Fair | Better | Better | Fair |

**RECOMMENDED: Hybrid CNN-Transformer** (best balance of quality, speed, implementability)

**FALLBACK: Pure CNN** (if hybrid has issues)

---

## FINAL ARCHITECTURE SELECTION

### Recommended Architecture: "RestoreNet"

**Design Philosophy:**

- **Simplicity first:** Reduce unnecessary complexity  
- **Fidelity focus:** Preserve information, don't hallucinate  
- **Practical efficiency:** Optimize full pipeline

### Architecture Details

\# src/models/restorenet.py

import torch

import torch.nn as nn

import torch.nn.functional as F

class ResidualBlock(nn.Module):

    """Standard residual block with ReLU."""

    def \_\_init\_\_(self, channels, kernel\_size=3, padding=1):

        super().\_\_init\_\_()

        self.conv1 \= nn.Conv2d(channels, channels, kernel\_size, padding=padding)

        self.relu \= nn.ReLU(inplace=True)

        self.conv2 \= nn.Conv2d(channels, channels, kernel\_size, padding=padding)

    

    def forward(self, x):

        return x \+ self.conv2(self.relu(self.conv1(x)))

class ChannelAttention(nn.Module):

    """Squeeze-and-excitation (CA) block."""

    def \_\_init\_\_(self, channels, reduction=16):

        super().\_\_init\_\_()

        self.fc1 \= nn.Conv2d(channels, channels // reduction, 1\)

        self.fc2 \= nn.Conv2d(channels // reduction, channels, 1\)

    

    def forward(self, x):

        \# Global average pooling

        avg \= F.adaptive\_avg\_pool2d(x, 1\)

        \# FC layers

        attn \= self.fc2(F.relu(self.fc1(avg)))

        \# Sigmoid \+ scale

        return x \* torch.sigmoid(attn)

class RestoreNet(nn.Module):

    """

    RestoreNet: Multi-scale CNN for image restoration

    

    Key features:

    \- Explicit upsampling (2× or 4×)

    \- Multi-scale residual feature extraction

    \- Channel attention

    \- Residual learning (output is predicted residual \+ upsampled input)

    

    Hyperparameters:

    \- scale\_factor: 2 or 4 (upsampling factor)

    \- num\_features: 64 (hidden channel count)

    \- num\_blocks: 10 (residual blocks per scale)

    \- num\_scales: 3 (single-scale or multi-scale)

    """

    

    def \_\_init\_\_(self, 

                 scale\_factor=2,

                 num\_features=64,

                 num\_blocks=10):

        super().\_\_init\_\_()

        

        self.scale\_factor \= scale\_factor

        

        \# Upsampling (explicit, not learned)

        self.upsample \= nn.Sequential(

            nn.Upsample(scale\_factor=scale\_factor, mode='bilinear', 

                       align\_corners=False)

        )

        

        \# Initial feature extraction

        self.conv\_in \= nn.Conv2d(1, num\_features, kernel\_size=3, padding=1)

        

        \# Main residual blocks

        self.res\_blocks \= nn.ModuleList(\[

            ResidualBlock(num\_features)

            for \_ in range(num\_blocks)

        \])

        

        \# Attention blocks (every 5th block)

        self.attention\_blocks \= nn.ModuleList(\[

            ChannelAttention(num\_features)

            for \_ in range(num\_blocks // 5\)

        \])

        

        \# Feature refinement

        self.conv\_mid \= nn.Conv2d(num\_features, num\_features, kernel\_size=3, padding=1)

        

        \# Output layer

        self.conv\_out \= nn.Conv2d(num\_features, 1, kernel\_size=3, padding=1)

    

    def forward(self, x):

        \# Upsample

        upsampled \= self.upsample(x)

        

        \# Extract features

        feat \= self.conv\_in(upsampled)

        

        \# Residual blocks with attention

        attn\_idx \= 0

        for i, block in enumerate(self.res\_blocks):

            feat \= block(feat)

            

            \# Apply attention every 5th block

            if (i \+ 1\) % 5 \== 0 and attn\_idx \< len(self.attention\_blocks):

                feat \= self.attention\_blocks\[attn\_idx\](feat)

                attn\_idx \+= 1

        

        \# Mid refinement

        feat \= self.conv\_mid(feat)

        

        \# Reconstruct (residual learning)

        residual \= self.conv\_out(feat)

        output \= upsampled \+ residual  \# Add residual connection from upsampled input

        

        return output

\# Model size estimation

def count\_parameters(model):

    return sum(p.numel() for p in model.parameters() if p.requires\_grad)

if \_\_name\_\_ \== '\_\_main\_\_':

    model \= RestoreNet(scale\_factor=2, num\_features=64, num\_blocks=10)

    print(f"Parameter count: {count\_parameters(model) / 1e6:.2f}M")

    

    \# Expected: \~1.5–2.0M parameters (reasonable size)

### Architecture Summary

Input: \[B, 1, H\_lr, W\_lr\]  (degraded, low-res)

  ↓

Bilinear upsample × scale\_factor

  → \[B, 1, H\_gt, W\_gt\]  (upsampled baseline)

  ↓

Conv 3×3 (1→64)

  → \[B, 64, H\_gt, W\_gt\]

  ↓

10× (Residual Block \+ occasional Channel Attention)

  → \[B, 64, H\_gt, W\_gt\]  (learned features)

  ↓

Conv 3×3 (64→1)

  → \[B, 1, H\_gt, W\_gt\]  (predicted residual)

  ↓

Add upsampled baseline (residual connection)

  → \[B, 1, H\_gt, W\_gt\]  (final output)

### Key Design Choices

| Choice | Rationale |
| :---- | :---- |
| **Explicit upsample** | Deterministic, no learned interpolation (simpler) |
| **Residual learning** | Output learns δ \= restored \- upsampled (easier optimization) |
| **Channel Attention** | Focuses model on important channels (light) |
| **No spatial attention** | Adds complexity; CA is sufficient |
| **10 residual blocks** | Balance between capacity and speed (\~1.6M params) |
| **Single scale** | Simpler than multi-scale; sufficient for 256×512 images |
| **No batch norm** | ResNets often trained without BN in restoration; stable training |

### Model Complexity

Parameter count: \~1.6 million

Memory per image (forward): \~50 MB (single pass, batch=1)

Inference time per image: 35–50 ms (H100)

---

## LOSS FUNCTION DESIGN

### Final Loss Configuration

\# src/training/losses.py

import torch

import torch.nn as nn

import torch.nn.functional as F

import torchvision.models as models

class RestorationLoss(nn.Module):

    """

    Multi-term loss for image restoration.

    

    Components:

    1\. L1 (pixel fidelity)

    2\. SSIM (structural similarity)

    3\. LPIPS (perceptual quality)

    """

    

    def \_\_init\_\_(self, 

                 lambda\_pixel=1.0,

                 lambda\_ssim=0.3,

                 lambda\_lpips=0.1,

                 device='cuda'):

        super().\_\_init\_\_()

        

        self.lambda\_pixel \= lambda\_pixel

        self.lambda\_ssim \= lambda\_ssim

        self.lambda\_lpips \= lambda\_lpips

        

        \# L1 loss

        self.l1\_loss \= nn.L1Loss()

        

        \# SSIM loss

        self.ssim\_fn \= SSIM()

        

        \# LPIPS loss (using pretrained AlexNet features)

        self.lpips\_fn \= LPIPS(net='alex', version='0.1').to(device).eval()

        for param in self.lpips\_fn.parameters():

            param.requires\_grad \= False

    

    def forward(self, pred, target):

        """

        Args:

            pred: Predicted image \[B, 1, H, W\] in range \[0,1\] (after clipping)

            target: Ground truth \[B, 1, H, W\] in range \[0,1\]

        

        Returns:

            loss: Scalar loss value

        """

        \# Clip pred to \[0,1\] for fair comparison

        pred\_clipped \= torch.clamp(pred, 0, 1\)

        

        \# L1 pixel loss

        l1\_loss \= self.l1\_loss(pred\_clipped, target)

        

        \# SSIM loss

        ssim\_loss \= 1 \- self.ssim\_fn(pred\_clipped, target)

        

        \# LPIPS loss

        with torch.no\_grad():

            lpips\_loss \= self.lpips\_fn(2 \* pred\_clipped \- 1,  \# Normalize to \[-1,1\]

                                       2 \* target \- 1).mean()

        

        \# Combine

        total\_loss \= (self.lambda\_pixel \* l1\_loss \+

                     self.lambda\_ssim \* ssim\_loss \+

                     self.lambda\_lpips \* lpips\_loss)

        

        return total\_loss

class SSIM(nn.Module):

    """Structural Similarity Index (SSIM)."""

    

    def \_\_init\_\_(self, window\_size=11, sigma=1.5):

        super().\_\_init\_\_()

        self.window\_size \= window\_size

        self.sigma \= sigma

        self.register\_buffer('window', self.\_create\_window(window\_size, sigma))

    

    def \_create\_window(self, window\_size, sigma):

        """Gaussian window."""

        gauss \= torch.Tensor(\[

            torch.exp(torch.tensor(-x\*\*2 / (2 \* sigma\*\*2)))

            for x in range(-(window\_size // 2), window\_size // 2 \+ 1\)

        \])

        window \= gauss / gauss.sum()

        window \= window.unsqueeze(1) \* window.unsqueeze(0)

        return window.unsqueeze(0).unsqueeze(0)  \# \[1, 1, W, W\]

    

    def forward(self, x, y):

        """Compute SSIM between x and y."""

        \# Constants

        C1 \= 0.01 \*\* 2

        C2 \= 0.03 \*\* 2

        

        \# Mean

        mu\_x \= F.conv2d(x, self.window, padding=self.window\_size // 2\)

        mu\_y \= F.conv2d(y, self.window, padding=self.window\_size // 2\)

        

        \# Variance

        mu\_xx \= F.conv2d(x \* x, self.window, padding=self.window\_size // 2\)

        mu\_yy \= F.conv2d(y \* y, self.window, padding=self.window\_size // 2\)

        mu\_xy \= F.conv2d(x \* y, self.window, padding=self.window\_size // 2\)

        

        sigma\_xx \= mu\_xx \- mu\_x \*\* 2

        sigma\_yy \= mu\_yy \- mu\_y \*\* 2

        sigma\_xy \= mu\_xy \- mu\_x \* mu\_y

        

        \# SSIM

        numerator \= (2 \* mu\_x \* mu\_y \+ C1) \* (2 \* sigma\_xy \+ C2)

        denominator \= (mu\_x \*\* 2 \+ mu\_y \*\* 2 \+ C1) \* (sigma\_xx \+ sigma\_yy \+ C2)

        

        ssim\_map \= numerator / (denominator \+ 1e-8)

        return ssim\_map.mean()

class LPIPS(nn.Module):

    """LPIPS: Learned Perceptual Image Patch Similarity."""

    

    def \_\_init\_\_(self, net='alex', version='0.1'):

        super().\_\_init\_\_()

        \# Use pretrained AlexNet

        self.net \= models.alexnet(pretrained=True)

        \# Extract features (we'll use layer 1-5)

        self.layer\_names \= \['relu1', 'relu2', 'relu3', 'relu4', 'relu5'\]

        self.register\_buffer('mean', torch.tensor(\[0.485, 0.456, 0.406\]).view(1, 3, 1, 1))

        self.register\_buffer('std', torch.tensor(\[0.229, 0.224, 0.225\]).view(1, 3, 1, 1))

    

    def forward(self, x, y):

        """Compute LPIPS distance between x and y."""

        \# x, y should be in \[-1, 1\]

        \# Normalize to ImageNet

        x \= (x \+ 1\) / 2  \# \[-1,1\] → \[0,1\]

        y \= (y \+ 1\) / 2

        

        x \= (x \- self.mean) / self.std

        y \= (y \- self.mean) / self.std

        

        \# Duplicate grayscale to RGB

        if x.shape\[1\] \== 1:

            x \= x.repeat(1, 3, 1, 1\)

            y \= y.repeat(1, 3, 1, 1\)

        

        \# Extract features at multiple layers

        features\_x \= self.\_forward\_to\_layer(x)

        features\_y \= self.\_forward\_to\_layer(y)

        

        \# Compute L2 distance between features

        dist \= torch.stack(\[

            torch.nn.functional.l2\_normalize(fx \- fy, dim=1).mean()

            for fx, fy in zip(features\_x, features\_y)

        \]).mean()

        

        return dist

    

    def \_forward\_to\_layer(self, x):

        """Extract features at multiple layers."""

        features \= \[\]

        for layer in \[0, 3, 6, 8, 10\]:  \# Layers corresponding to relu1-5

            x \= self.net.features\[layer\](x)

            features.append(x)

        return features

### Loss Weight Configuration

**Recommended (Phase 1):**

loss:

  lambda\_pixel: 1.0   \# L1 pixel reconstruction error

  lambda\_ssim: 0.3    \# Structural consistency

  lambda\_lpips: 0.1   \# Perceptual quality

  

  \# Advanced (optional, disable initially):

  lambda\_freq: 0.0    \# Frequency domain (FFT)

  lambda\_grad: 0.0    \# Gradient/edge consistency

**Rationale:**

- L1 as anchor (pixel fidelity)  
- SSIM balances local structure (0.3× helps without dominating)  
- LPIPS provides perceptual alignment (light weight to avoid hallucination)

**If ablation shows benefit, adjust:**

\# More structural emphasis:

lambda\_pixel: 1.0

lambda\_ssim: 0.5    \# ↑ increase from 0.3

lambda\_lpips: 0.1

\# More perceptual emphasis:

lambda\_pixel: 0.7   \# ↓ decrease from 1.0

lambda\_ssim: 0.3

lambda\_lpips: 0.2   \# ↑ increase from 0.1

---

## TRAINING STRATEGY

### Training Configuration

\# configs/train.yaml

model:

  name: restorenet

  scale\_factor: 2

  num\_features: 64

  num\_blocks: 10

  pretrained: null

training:

  epochs: 100

  batch\_size: 8

  learning\_rate: 1e-3

  optimizer: Adam

  betas: \[0.9, 0.999\]

  weight\_decay: 0

  

  scheduler:

    name: CosineAnnealingLR

    T\_max: 100

    eta\_min: 1e-6

  

  warmup\_epochs: 5

  gradient\_clip: 1.0

  

  loss:

    lambda\_pixel: 1.0

    lambda\_ssim: 0.3

    lambda\_lpips: 0.1

mixed\_precision:

  enabled: true

  opt\_level: O2

data:

  train\_ratio: 0.70

  val\_ratio: 0.20

  include\_synthetic: true  \# Use synthetic augmentation

  synthetic\_samples\_per\_image: 2

  

checkpointing:

  save\_every\_n\_epochs: 5

  keep\_last\_n: 3

  best\_metric: val\_psnr

  patience: 20  \# Early stopping

logging:

  tensorboard: true

  log\_every\_n\_batches: 50

  save\_dir: ./logs

### Training Pipeline

\# scripts/train.py

import torch

import torch.nn as nn

import torch.optim as optim

from torch.utils.data import DataLoader, ConcatDataset

import yaml

from pathlib import Path

import sys

from datetime import datetime

\# Import project modules

sys.path.insert(0, '.')

from src.models.restorenet import RestoreNet

from src.data.dataset import RestorationDataset

from src.data.augmentation import SyntheticDegradationAugmentor

from src.training.losses import RestorationLoss

from src.training.metrics import compute\_psnr, compute\_ssim

def load\_config(config\_path):

    with open(config\_path) as f:

        return yaml.safe\_load(f)

def train\_epoch(model, loader, optimizer, loss\_fn, device, epoch, total\_epochs):

    """Train for one epoch."""

    model.train()

    total\_loss \= 0.0

    

    for batch\_idx, (noisylr, gt) in enumerate(loader):

        noisylr \= noisylr.to(device)

        gt \= gt.to(device)

        

        \# Forward pass

        optimizer.zero\_grad()

        pred \= model(noisylr)

        loss \= loss\_fn(pred, gt)

        

        \# Backward pass

        loss.backward()

        torch.nn.utils.clip\_grad\_norm\_(model.parameters(), max\_norm=1.0)

        optimizer.step()

        

        total\_loss \+= loss.item()

        

        if (batch\_idx \+ 1\) % 50 \== 0:

            print(f"\[Epoch {epoch+1}/{total\_epochs}\] "

                  f"Batch {batch\_idx+1}/{len(loader)}: Loss \= {loss.item():.4f}")

    

    avg\_loss \= total\_loss / len(loader)

    return avg\_loss

def val\_epoch(model, loader, device):

    """Validation epoch \- compute metrics."""

    model.eval()

    psnr\_values, ssim\_values \= \[\], \[\]

    

    with torch.no\_grad():

        for noisylr, gt in loader:

            noisylr \= noisylr.to(device)

            gt \= gt.to(device)

            

            \# Forward pass

            pred \= model(noisylr)

            

            \# Compute metrics

            pred\_clipped \= torch.clamp(pred, 0, 1\)

            psnr \= compute\_psnr(pred\_clipped, gt)

            ssim \= compute\_ssim(pred\_clipped, gt)

            

            psnr\_values.append(psnr)

            ssim\_values.append(ssim)

    

    avg\_psnr \= torch.stack(psnr\_values).mean().item()

    avg\_ssim \= torch.stack(ssim\_values).mean().item()

    

    return avg\_psnr, avg\_ssim

def main():

    \# Load config

    cfg \= load\_config('configs/train.yaml')

    

    \# Device

    device \= torch.device('cuda' if torch.cuda.is\_available() else 'cpu')

    print(f"Using device: {device}")

    

    \# Model

    model \= RestoreNet(

        scale\_factor=cfg\['model'\]\['scale\_factor'\],

        num\_features=cfg\['model'\]\['num\_features'\],

        num\_blocks=cfg\['model'\]\['num\_blocks'\]

    ).to(device)

    

    param\_count \= sum(p.numel() for p in model.parameters())

    print(f"Model parameters: {param\_count / 1e6:.2f}M")

    

    \# Data

    dataset \= RestorationDataset(

        gt\_dir=cfg\['data'\]\['gt\_dir'\],

        noisylr\_dir=cfg\['data'\]\['noisylr\_dir'\],

        normalize=False,  \# Preserve out-of-range values

        augment=True

    )

    

    \# Add synthetic data if enabled

    if cfg\['data'\]\['include\_synthetic'\]:

        augmentor \= SyntheticDegradationAugmentor()

        print("Generating synthetic degradation pairs...")

        augmentor.augment\_dataset(

            cfg\['data'\]\['gt\_dir'\],

            'data/NoisyLR\_synth',

            samples\_per\_image=cfg\['data'\]\['synthetic\_samples\_per\_image'\]

        )

        

        \# Load synthetic dataset

        synthetic\_dataset \= RestorationDataset(

            gt\_dir=cfg\['data'\]\['gt\_dir'\],

            noisylr\_dir='data/NoisyLR\_synth',

            normalize=False,

            augment=False  \# Already augmented during generation

        )

        

        dataset \= ConcatDataset(\[dataset, synthetic\_dataset\])

    

    \# Split

    train\_size \= int(cfg\['data'\]\['train\_ratio'\] \* len(dataset))

    val\_size \= int(cfg\['data'\]\['val\_ratio'\] \* len(dataset))

    test\_size \= len(dataset) \- train\_size \- val\_size

    

    train\_dataset, val\_dataset, test\_dataset \= torch.utils.data.random\_split(

        dataset,

        \[train\_size, val\_size, test\_size\],

        generator=torch.Generator().manual\_seed(42)

    )

    

    train\_loader \= DataLoader(

        train\_dataset,

        batch\_size=cfg\['training'\]\['batch\_size'\],

        shuffle=True,

        num\_workers=4,

        pin\_memory=True,

        prefetch\_factor=2

    )

    

    val\_loader \= DataLoader(

        val\_dataset,

        batch\_size=cfg\['training'\]\['batch\_size'\],

        shuffle=False,

        num\_workers=4,

        pin\_memory=True

    )

    

    \# Loss & Optimizer

    loss\_fn \= RestorationLoss(

        lambda\_pixel=cfg\['loss'\]\['lambda\_pixel'\],

        lambda\_ssim=cfg\['loss'\]\['lambda\_ssim'\],

        lambda\_lpips=cfg\['loss'\]\['lambda\_lpips'\],

        device=device

    )

    

    optimizer \= optim.Adam(

        model.parameters(),

        lr=cfg\['training'\]\['learning\_rate'\],

        betas=tuple(cfg\['training'\]\['betas'\]),

        weight\_decay=cfg\['training'\]\['weight\_decay'\]

    )

    

    scheduler \= optim.lr\_scheduler.CosineAnnealingLR(

        optimizer,

        T\_max=cfg\['training'\]\['epochs'\],

        eta\_min=cfg\['training'\]\['scheduler'\]\['eta\_min'\]

    )

    

    \# Training loop

    best\_psnr \= 0.0

    patience\_counter \= 0

    

    for epoch in range(cfg\['training'\]\['epochs'\]):

        \# Train

        train\_loss \= train\_epoch(model, train\_loader, optimizer, loss\_fn, device, epoch, cfg\['training'\]\['epochs'\])

        

        \# Validate

        val\_psnr, val\_ssim \= val\_epoch(model, val\_loader, device)

        

        \# Log

        print(f"\\n\[Epoch {epoch+1}/{cfg\['training'\]\['epochs'\]}\]")

        print(f"  Train Loss: {train\_loss:.4f}")

        print(f"  Val PSNR: {val\_psnr:.2f} dB | Val SSIM: {val\_ssim:.4f}")

        

        \# Scheduler step

        scheduler.step()

        

        \# Save best checkpoint

        if val\_psnr \> best\_psnr:

            best\_psnr \= val\_psnr

            patience\_counter \= 0

            

            ckpt\_path \= f"checkpoints/best\_model.pt"

            torch.save({

                'epoch': epoch,

                'model\_state': model.state\_dict(),

                'optimizer\_state': optimizer.state\_dict(),

                'val\_psnr': val\_psnr,

                'val\_ssim': val\_ssim,

            }, ckpt\_path)

            print(f"  → Saved best checkpoint: {ckpt\_path}")

        else:

            patience\_counter \+= 1

            if patience\_counter \>= cfg\['training'\]\['patience'\]:

                print(f"Early stopping triggered (patience={cfg\['training'\]\['patience'\]})")

                break

        

        \# Periodic checkpoint

        if (epoch \+ 1\) % cfg\['checkpointing'\]\['save\_every\_n\_epochs'\] \== 0:

            ckpt\_path \= f"checkpoints/model\_epoch\_{epoch+1}.pt"

            torch.save(model.state\_dict(), ckpt\_path)

    

    print(f"\\nTraining complete. Best PSNR: {best\_psnr:.2f} dB")

    return model

if \_\_name\_\_ \== '\_\_main\_\_':

    main()

### Critical Training Tips

**1\. Seed Management:**

def set\_seed(seed=42):

    import numpy as np

    import random

    random.seed(seed)

    np.random.seed(seed)

    torch.manual\_seed(seed)

    torch.cuda.manual\_seed\_all(seed)

    torch.backends.cudnn.deterministic \= True

    torch.backends.cudnn.benchmark \= False

set\_seed(42)

**2\. Gradient Clipping:**

torch.nn.utils.clip\_grad\_norm\_(model.parameters(), max\_norm=1.0)

(Prevents gradient explosion with multi-term loss)

**3\. Learning Rate Scheduling:**

- Warmup: Linear increase over 5 epochs  
- Main: Cosine annealing over remaining epochs  
- Final LR: 1e-6 (very small)

**4\. Numerical Stability:**

- Use mixed precision (AMP) if memory is tight  
- Normalize loss terms if scales differ significantly  
- Monitor for NaNs/Infs during first few batches

**5\. Early Stopping:**

- Monitor validation PSNR  
- Stop if no improvement for 20 epochs  
- Save best checkpoint separately

---

## VALIDATION & EVALUATION

### Validation Strategy

\# src/training/validation.py

import torch

import torch.nn as nn

import torch.nn.functional as F

from skimage.metrics import peak\_signal\_noise\_ratio, structural\_similarity

import lpips as lpips\_module

class MetricsComputer:

    """Compute PSNR, SSIM, LPIPS for validation."""

    

    def \_\_init\_\_(self, device='cuda'):

        self.device \= device

        self.lpips\_fn \= lpips\_module.LPIPS(net='alex', version='0.1').to(device).eval()

    

    def compute\_psnr(self, pred, target):

        """PSNR in dB (higher is better)."""

        pred\_np \= pred.detach().cpu().numpy().squeeze()

        target\_np \= target.detach().cpu().numpy().squeeze()

        

        \# Clip predictions to \[0,1\]

        pred\_np \= np.clip(pred\_np, 0, 1\)

        

        \# Compute PSNR

        psnr \= peak\_signal\_noise\_ratio(target\_np, pred\_np, data\_range=1.0)

        return psnr

    

    def compute\_ssim(self, pred, target):

        """SSIM (higher is better, max=1)."""

        pred\_np \= pred.detach().cpu().numpy().squeeze()

        target\_np \= target.detach().cpu().numpy().squeeze()

        

        pred\_np \= np.clip(pred\_np, 0, 1\)

        

        ssim \= structural\_similarity(target\_np, pred\_np, data\_range=1.0)

        return ssim

    

    def compute\_lpips(self, pred, target):

        """LPIPS (lower is better)."""

        with torch.no\_grad():

            \# Normalize to \[-1, 1\]

            pred\_norm \= 2 \* torch.clamp(pred, 0, 1\) \- 1

            target\_norm \= 2 \* target \- 1

            

            \# Duplicate to RGB if grayscale

            if pred\_norm.shape\[1\] \== 1:

                pred\_norm \= pred\_norm.repeat(1, 3, 1, 1\)

                target\_norm \= target\_norm.repeat(1, 3, 1, 1\)

            

            lpips\_val \= self.lpips\_fn(pred\_norm, target\_norm).mean()

        

        return lpips\_val.item()

def validate\_with\_metrics(model, val\_loader, device, save\_dir=None):

    """

    Comprehensive validation with all metrics.

    

    Returns:

        dict: {

            'psnr': float,

            'ssim': float,

            'lpips': float,

            'predictions': list\[np.ndarray\],

            'targets': list\[np.ndarray\]

        }

    """

    model.eval()

    metrics\_computer \= MetricsComputer(device=device)

    

    psnr\_list \= \[\]

    ssim\_list \= \[\]

    lpips\_list \= \[\]

    predictions \= \[\]

    targets \= \[\]

    

    with torch.no\_grad():

        for batch\_idx, (noisylr, gt) in enumerate(val\_loader):

            noisylr \= noisylr.to(device)

            gt \= gt.to(device)

            

            \# Inference

            pred \= model(noisylr)

            

            \# Clip to \[0,1\]

            pred\_clipped \= torch.clamp(pred, 0, 1\)

            

            \# Compute metrics (per-image)

            batch\_size \= pred\_clipped.shape\[0\]

            for i in range(batch\_size):

                psnr \= metrics\_computer.compute\_psnr(pred\_clipped\[i:i+1\], gt\[i:i+1\])

                ssim \= metrics\_computer.compute\_ssim(pred\_clipped\[i:i+1\], gt\[i:i+1\])

                lpips\_val \= metrics\_computer.compute\_lpips(pred\_clipped\[i:i+1\], gt\[i:i+1\])

                

                psnr\_list.append(psnr)

                ssim\_list.append(ssim)

                lpips\_list.append(lpips\_val)

                

                predictions.append(pred\_clipped\[i\].cpu().numpy())

                targets.append(gt\[i\].cpu().numpy())

            

            if (batch\_idx \+ 1\) % 10 \== 0:

                print(f"Validated {batch\_idx+1} batches...")

    

    \# Aggregate

    results \= {

        'psnr': np.mean(psnr\_list),

        'psnr\_std': np.std(psnr\_list),

        'ssim': np.mean(ssim\_list),

        'ssim\_std': np.std(ssim\_list),

        'lpips': np.mean(lpips\_list),

        'lpips\_std': np.std(lpips\_list),

        'predictions': predictions,

        'targets': targets,

    }

    

    return results

### OOD Generalization Testing

\# src/training/ood\_validation.py

class OODValidator:

    """Evaluate model on out-of-distribution data."""

    

    def \_\_init\_\_(self, model, device='cuda'):

        self.model \= model

        self.device \= device

        self.metrics \= MetricsComputer(device=device)

    

    def test\_ood\_content(self, model\_dir, test\_pairs):

        """

        Test on unfamiliar image content.

        

        Args:

            test\_pairs: List of (noisylr\_path, gt\_path) tuples for OOD content

        

        Returns:

            dict: Metrics aggregated over OOD test set

        """

        results \= {'psnr': \[\], 'ssim': \[\], 'lpips': \[\]}

        

        for noisylr\_path, gt\_path in test\_pairs:

            noisylr \= torch.from\_numpy(np.load(noisylr\_path)).unsqueeze(0).unsqueeze(0).to(self.device)

            gt \= torch.from\_numpy(np.load(gt\_path)).unsqueeze(0).unsqueeze(0).to(self.device)

            

            with torch.no\_grad():

                pred \= self.model(noisylr)

            

            pred\_clipped \= torch.clamp(pred, 0, 1\)

            

            psnr \= self.metrics.compute\_psnr(pred\_clipped, gt)

            ssim \= self.metrics.compute\_ssim(pred\_clipped, gt)

            lpips\_val \= self.metrics.compute\_lpips(pred\_clipped, gt)

            

            results\['psnr'\].append(psnr)

            results\['ssim'\].append(ssim)

            results\['lpips'\].append(lpips\_val)

        

        \# Aggregate

        return {

            'psnr\_mean': np.mean(results\['psnr'\]),

            'psnr\_std': np.std(results\['psnr'\]),

            'ssim\_mean': np.mean(results\['ssim'\]),

            'ssim\_std': np.std(results\['ssim'\]),

            'lpips\_mean': np.mean(results\['lpips'\]),

            'lpips\_std': np.std(results\['lpips'\]),

        }

### Failure Analysis

\# scripts/analyze\_failures.py

def identify\_failures(predictions, targets, psnr\_threshold=24.0):

    """Identify images where model underperforms."""

    

    failures \= \[\]

    successes \= \[\]

    

    for i, (pred, target) in enumerate(zip(predictions, targets)):

        psnr \= peak\_signal\_noise\_ratio(target.squeeze(), pred.squeeze(), data\_range=1.0)

        

        if psnr \< psnr\_threshold:

            failures.append({'idx': i, 'psnr': psnr, 'pred': pred, 'target': target})

        else:

            successes.append({'idx': i, 'psnr': psnr})

    

    print(f"Failures (PSNR \< {psnr\_threshold}): {len(failures)} / {len(predictions)}")

    print(f"Failure rate: {len(failures) / len(predictions) \* 100:.1f}%")

    

    return failures, successes

---

## GPU OPTIMIZATION

### H100 Optimization Strategy

\# src/optimization/inference\_engine.py

import torch

import torch.nn as nn

import time

class OptimizedInferenceEngine:

    """Optimized inference for H100 GPU."""

    

    def \_\_init\_\_(self, model\_path, device='cuda', use\_compile=True):

        """

        Initialize inference engine.

        

        Args:

            model\_path: Path to saved model weights

            device: 'cuda' or 'cpu'

            use\_compile: Use torch.compile() if available (PyTorch 2.0+)

        """

        self.device \= device

        

        \# Load model

        from src.models.restorenet import RestoreNet

        self.model \= RestoreNet().to(device).eval()

        

        checkpoint \= torch.load(model\_path, map\_location=device)

        if 'model\_state' in checkpoint:

            self.model.load\_state\_dict(checkpoint\['model\_state'\])

        else:

            self.model.load\_state\_dict(checkpoint)

        

        \# Optional: torch.compile() (PyTorch 2.0+)

        if use\_compile and hasattr(torch, 'compile'):

            try:

                self.model \= torch.compile(self.model, mode='reduce-overhead')

                print("✓ Model compiled with torch.compile()")

            except Exception as e:

                print(f"⚠ torch.compile() failed: {e}. Using eager mode.")

        

        \# Pinned memory for faster transfers

        self.use\_pinned\_memory \= True

    

    def preprocess(self, noisylr\_path):

        """Load and preprocess image."""

        \# Load

        noisylr \= np.load(noisylr\_path).astype(np.float32)

        

        \# Add channel dimension

        noisylr \= noisylr\[np.newaxis, np.newaxis, ...\]  \# \[1, 1, H, W\]

        

        \# Convert to tensor (pinned for faster transfer)

        if self.use\_pinned\_memory:

            tensor \= torch.from\_numpy(noisylr).pin\_memory()

        else:

            tensor \= torch.from\_numpy(noisylr)

        

        return tensor

    

    def infer\_single(self, noisylr\_tensor):

        """Inference on single image."""

        noisylr\_tensor \= noisylr\_tensor.to(self.device, non\_blocking=True)

        

        with torch.inference\_mode():  \# Slightly faster than no\_grad()

            output \= self.model(noisylr\_tensor)

        

        \# Transfer back to CPU

        output \= output.cpu()

        

        return output

    

    def infer\_batch(self, noisylr\_tensors):

        """Batch inference."""

        \# Stack tensors

        batch \= torch.cat(noisylr\_tensors, dim=0)

        batch \= batch.to(self.device, non\_blocking=True)

        

        with torch.inference\_mode():

            outputs \= self.model(batch)

        

        outputs \= outputs.cpu()

        

        return torch.split(outputs, 1, dim=0)  \# Split back into single images

    

    def postprocess(self, output\_tensor):

        """Postprocess model output."""

        output\_np \= output\_tensor.squeeze().numpy()

        

        \# Clip to \[0,1\]

        output\_np \= np.clip(output\_np, 0, 1\)

        

        \# Convert to float32 for saving

        output\_np \= output\_np.astype(np.float32)

        

        return output\_np

    

    def benchmark(self, input\_dir, output\_dir, num\_images=None):

        """

        Benchmark inference speed.

        

        Measures:

        \- Disk I/O time

        \- Preprocessing time

        \- Model inference time

        \- Postprocessing time

        \- Total end-to-end time

        """

        import os

        from pathlib import Path

        

        image\_files \= sorted(Path(input\_dir).glob('\*.npy'))

        if num\_images:

            image\_files \= image\_files\[:num\_images\]

        

        timings \= {

            'disk\_io': \[\],

            'preprocess': \[\],

            'inference': \[\],

            'postprocess': \[\],

            'total': \[\]

        }

        

        Path(output\_dir).mkdir(parents=True, exist\_ok=True)

        

        \# Warmup

        print("Warming up model...")

        for \_ in range(3):

            dummy\_input \= torch.randn(1, 1, 256, 256).to(self.device)

            with torch.inference\_mode():

                \_ \= self.model(dummy\_input)

        

        \# Benchmark

        print(f"Benchmarking on {len(image\_files)} images...")

        for idx, image\_file in enumerate(image\_files):

            start\_total \= time.time()

            

            \# Disk I/O

            start \= time.time()

            noisylr\_tensor \= self.preprocess(str(image\_file))

            timings\['disk\_io'\].append(time.time() \- start)

            

            \# Preprocess (in this case, already done in preprocess())

            \# timings\['preprocess'\].append(0)

            

            \# Inference

            start \= time.time()

            output\_tensor \= self.infer\_single(noisylr\_tensor)

            timings\['inference'\].append(time.time() \- start)

            

            \# Postprocess

            start \= time.time()

            output\_np \= self.postprocess(output\_tensor)

            timings\['postprocess'\].append(time.time() \- start)

            

            \# Save

            output\_path \= Path(output\_dir) / image\_file.name

            np.save(output\_path, output\_np)

            

            timings\['total'\].append(time.time() \- start\_total)

            

            if (idx \+ 1\) % 20 \== 0:

                avg\_total \= np.mean(timings\['total'\]\[-20:\])

                print(f"  Processed {idx+1}/{len(image\_files)} images. "

                      f"Avg time: {avg\_total\*1000:.1f}ms/image")

        

        \# Report

        print("\\n" \+ "=" \* 80\)

        print("BENCHMARK RESULTS")

        print("=" \* 80\)

        print(f"Disk I/O:       {np.mean(timings\['disk\_io'\])\*1000:7.2f}ms (±{np.std(timings\['disk\_io'\])\*1000:.2f})")

        print(f"Inference:      {np.mean(timings\['inference'\])\*1000:7.2f}ms (±{np.std(timings\['inference'\])\*1000:.2f})")

        print(f"Postprocess:    {np.mean(timings\['postprocess'\])\*1000:7.2f}ms (±{np.std(timings\['postprocess'\])\*1000:.2f})")

        print(f"Total:          {np.mean(timings\['total'\])\*1000:7.2f}ms (±{np.std(timings\['total'\])\*1000:.2f})")

        print(f"Throughput:     {1/np.mean(timings\['total'\]):.1f} images/sec")

        print("=" \* 80\)

        

        return timings

### torch.compile() Evaluation

\# scripts/benchmark\_compile.py

def compare\_inference\_modes():

    """

    Compare inference speed with and without torch.compile().

    

    Note: torch.compile() is available in PyTorch 2.0+

    """

    import time

    

    model \= RestoreNet().cuda().eval()

    dummy\_input \= torch.randn(1, 1, 256, 256).cuda()

    

    \# Eager mode

    print("Benchmarking eager mode...")

    times\_eager \= \[\]

    for \_ in range(10):  \# Warmup

        with torch.inference\_mode():

            \_ \= model(dummy\_input)

    

    for \_ in range(100):

        start \= time.time()

        with torch.inference\_mode():

            \_ \= model(dummy\_input)

        times\_eager.append(time.time() \- start)

    

    \# Compiled mode

    if hasattr(torch, 'compile'):

        print("Benchmarking torch.compile() mode...")

        model\_compiled \= torch.compile(model, mode='reduce-overhead')

        

        times\_compiled \= \[\]

        for \_ in range(10):  \# Warmup

            with torch.inference\_mode():

                \_ \= model\_compiled(dummy\_input)

        

        for \_ in range(100):

            start \= time.time()

            with torch.inference\_mode():

                \_ \= model\_compiled(dummy\_input)

            times\_compiled.append(time.time() \- start)

        

        print(f"\\nEager:      {np.mean(times\_eager)\*1000:.2f}ms ± {np.std(times\_eager)\*1000:.2f}ms")

        print(f"Compiled:   {np.mean(times\_compiled)\*1000:.2f}ms ± {np.std(times\_compiled)\*1000:.2f}ms")

        print(f"Speedup:    {np.mean(times\_eager) / np.mean(times\_compiled):.2f}x")

    else:

        print("torch.compile() not available (requires PyTorch 2.0+)")

        print(f"Eager:      {np.mean(times\_eager)\*1000:.2f}ms ± {np.std(times\_eager)\*1000:.2f}ms")

### TensorRT Consideration (Optional)

**Only if torch.compile() doesn't achieve \<50ms:**

\# Export to ONNX

python scripts/export\_onnx.py \--model\_path checkpoints/best\_model.pt \--output onnx/model.onnx

\# Convert ONNX to TensorRT

trtexec \--onnx=onnx/model.onnx \--saveEngine=trt\_engines/model.engine \--fp16

\# Benchmark TensorRT

python scripts/benchmark\_tensorrt.py \--engine trt\_engines/model.engine

**Decision:** Use only if measurements show significant speedup over torch.compile().

---

## INFERENCE PIPELINE

### Standalone Inference Script

**MANDATORY:** Must accept `--input_dir` and `--output_dir` without code changes.

\# inference.py

\#\!/usr/bin/env python3

"""

Standalone inference script for KLA image restoration.

Usage:

    python inference.py \--input\_dir ./noisylr\_images \--output\_dir ./restored\_images

This script:

\- Loads all .npy files from input directory

\- Processes each with the trained model

\- Saves restored images to output directory

\- Requires NO manual code changes (fully configurable via CLI args)

"""

import argparse

import sys

from pathlib import Path

import numpy as np

import torch

import torch.nn as nn

from tqdm import tqdm

import time

\# Import model

sys.path.insert(0, str(Path(\_\_file\_\_).parent))

from src.models.restorenet import RestoreNet

from src.optimization.inference\_engine import OptimizedInferenceEngine

def parse\_args():

    parser \= argparse.ArgumentParser(

        description='Standalone inference for KLA image restoration'

    )

    parser.add\_argument('--input\_dir', type=str, required=True,

                       help='Input directory containing .npy files')

    parser.add\_argument('--output\_dir', type=str, required=True,

                       help='Output directory for restored images')

    parser.add\_argument('--model\_path', type=str, 

                       default='checkpoints/best\_model.pt',

                       help='Path to model checkpoint')

    parser.add\_argument('--device', type=str, default='cuda',

                       choices=\['cuda', 'cpu'\],

                       help='Device to use for inference')

    parser.add\_argument('--batch\_size', type=int, default=1,

                       help='Batch size for inference (1 for memory efficiency)')

    parser.add\_argument('--use\_compile', action='store\_true',

                       help='Use torch.compile() if available')

    parser.add\_argument('--save\_dtype', type=str, default='float32',

                       choices=\['float32', 'float64', 'uint8'\],

                       help='Output dtype')

    parser.add\_argument('--verbose', action='store\_true',

                       help='Print progress information')

    

    return parser.parse\_args()

def load\_model(model\_path, device='cuda'):

    """Load trained model."""

    model \= RestoreNet().to(device).eval()

    

    if Path(model\_path).exists():

        checkpoint \= torch.load(model\_path, map\_location=device)

        if isinstance(checkpoint, dict) and 'model\_state' in checkpoint:

            model.load\_state\_dict(checkpoint\['model\_state'\])

        else:

            model.load\_state\_dict(checkpoint)

        print(f"✓ Loaded model from {model\_path}")

    else:

        print(f"⚠ Model not found at {model\_path}. Using randomly initialized model.")

    

    return model

def infer\_image(model, image\_tensor, device='cuda'):

    """Inference on single image."""

    image\_tensor \= image\_tensor.to(device)

    

    with torch.inference\_mode():

        output \= model(image\_tensor)

    

    return output.cpu()

def save\_output(output\_array, output\_path, dtype='float32'):

    """Save output image."""

    output\_array \= np.clip(output\_array, 0, 1).astype(np.float32)

    

    if dtype \== 'uint8':

        output\_array \= (output\_array \* 255).astype(np.uint8)

    

    np.save(str(output\_path), output\_array)

def main():

    args \= parse\_args()

    

    \# Validate inputs

    input\_dir \= Path(args.input\_dir)

    if not input\_dir.exists():

        print(f"❌ Input directory not found: {input\_dir}")

        sys.exit(1)

    

    \# Create output directory

    output\_dir \= Path(args.output\_dir)

    output\_dir.mkdir(parents=True, exist\_ok=True)

    print(f"✓ Output directory: {output\_dir}")

    

    \# Find input files

    image\_files \= sorted(input\_dir.glob('\*.npy'))

    if not image\_files:

        print(f"❌ No .npy files found in {input\_dir}")

        sys.exit(1)

    

    print(f"✓ Found {len(image\_files)} images to process")

    

    \# Load model

    device \= args.device if torch.cuda.is\_available() else 'cpu'

    model \= load\_model(args.model\_path, device=device)

    

    if args.use\_compile and hasattr(torch, 'compile'):

        model \= torch.compile(model, mode='reduce-overhead')

        print("✓ Model compiled with torch.compile()")

    

    \# Inference loop

    start\_time \= time.time()

    timings \= \[\]

    

    for idx, image\_file in enumerate(image\_files):

        \# Load

        try:

            noisylr \= np.load(image\_file).astype(np.float32)

        except Exception as e:

            print(f"❌ Failed to load {image\_file}: {e}")

            continue

        

        \# Preprocess

        noisylr\_tensor \= torch.from\_numpy(noisylr\[np.newaxis, np.newaxis, ...\])

        

        \# Inference

        step\_start \= time.time()

        output\_tensor \= infer\_image(model, noisylr\_tensor, device=device)

        step\_time \= time.time() \- step\_start

        timings.append(step\_time)

        

        \# Postprocess

        output\_np \= output\_tensor.squeeze().numpy()

        

        \# Save

        output\_path \= output\_dir / image\_file.name

        try:

            save\_output(output\_np, output\_path, dtype=args.save\_dtype)

            status \= "✓"

        except Exception as e:

            print(f"❌ Failed to save {output\_path}: {e}")

            status \= "✗"

        

        if args.verbose or (idx \+ 1\) % 50 \== 0:

            print(f"{status} \[{idx+1}/{len(image\_files)}\] "

                  f"{image\_file.name} ({step\_time\*1000:.1f}ms)")

    

    \# Report

    total\_time \= time.time() \- start\_time

    print("\\n" \+ "=" \* 80\)

    print("INFERENCE SUMMARY")

    print("=" \* 80\)

    print(f"Total images:   {len(image\_files)}")

    print(f"Total time:     {total\_time:.2f}s")

    print(f"Avg time/image: {np.mean(timings)\*1000:.1f}ms (±{np.std(timings)\*1000:.2f}ms)")

    print(f"Throughput:     {len(image\_files)/total\_time:.1f} images/sec")

    print(f"Device:         {device}")

    print("=" \* 80\)

if \_\_name\_\_ \== '\_\_main\_\_':

    main()

**Usage:**

\# Basic

python inference.py \--input\_dir ./data/NoisyLR \--output\_dir ./results

\# With options

python inference.py \\

    \--input\_dir ./data/NoisyLR \\

    \--output\_dir ./results \\

    \--model\_path ./checkpoints/best\_model.pt \\

    \--device cuda \\

    \--batch\_size 8 \\

    \--use\_compile \\

    \--verbose

---

## REPOSITORY STRUCTURE

### Recommended Directory Layout

kla-image-restoration/

│

├─ README.md                          \# Main documentation

├─ LICENSE

├─ .gitignore

├─ requirements.txt                   \# Dependencies

├─ pyproject.toml                     \# Package metadata (optional)

│

├─ configs/                           \# Configuration files

│   ├─ base.yaml

│   ├─ train.yaml

│   ├─ model.yaml

│   ├─ inference.yaml

│   └─ benchmark.yaml

│

├─ src/                               \# Source code

│   ├─ \_\_init\_\_.py

│   ├─ data/

│   │   ├─ \_\_init\_\_.py

│   │   ├─ dataset.py                \# Data loading

│   │   ├─ augmentation.py           \# Synthetic degradation

│   │   ├─ split.py                  \# Train/val/test split

│   │   └─ loader.py                 \# DataLoader utilities

│   ├─ models/

│   │   ├─ \_\_init\_\_.py

│   │   ├─ restorenet.py             \# Main architecture

│   │   ├─ baseline.py               \# Baseline model

│   │   └─ blocks.py                 \# Reusable blocks

│   ├─ training/

│   │   ├─ \_\_init\_\_.py

│   │   ├─ trainer.py                \# Training loop

│   │   ├─ losses.py                 \# Loss functions

│   │   ├─ metrics.py                \# PSNR, SSIM, LPIPS

│   │   └─ validation.py             \# Validation logic

│   ├─ inference/

│   │   ├─ \_\_init\_\_.py

│   │   └─ engine.py                 \# Inference engine

│   ├─ optimization/

│   │   ├─ \_\_init\_\_.py

│   │   ├─ inference\_engine.py       \# H100 optimization

│   │   ├─ torchscript\_export.py     \# torch.compile

│   │   └─ tensorrt\_export.py        \# TensorRT (optional)

│   └─ utils/

│       ├─ \_\_init\_\_.py

│       ├─ logging.py

│       ├─ seed.py

│       └─ visualization.py

│

├─ scripts/                           \# Standalone scripts

│   ├─ inspect\_dataset.py            \# Data analysis

│   ├─ train.py                      \# Main training script

│   ├─ train\_baseline.py             \# Baseline training

│   ├─ evaluate.py                   \# Evaluation script

│   ├─ inference.py                  \# Standalone inference (CRITICAL)

│   ├─ benchmark.py                  \# Runtime benchmarking

│   ├─ generate\_synthetic\_pairs.py   \# Augmentation

│   ├─ export\_onnx.py               \# ONNX export (optional)

│   └─ analyze\_results.py           \# Post-processing

│

├─ tests/                            \# Unit tests

│   ├─ \_\_init\_\_.py

│   ├─ test\_dataset.py

│   ├─ test\_model.py

│   ├─ test\_losses.py

│   └─ test\_metrics.py

│

├─ notebooks/                        \# Jupyter notebooks (optional)

│   ├─ 01\_exploratory\_analysis.ipynb

│   ├─ 02\_model\_development.ipynb

│   └─ 03\_results\_visualization.ipynb

│

├─ data/                             \# Data directory (download separately)

│   ├─ GT/                           \# Ground truth images

│   ├─ NoisyLR/                      \# Degraded inputs

│   ├─ NoisyLR\_synth/               \# Synthetic degraded pairs

│   └─ test/                         \# Test set (if available)

│

├─ checkpoints/                      \# Trained models

│   ├─ baseline\_epoch\_50.pt

│   └─ best\_model.pt

│

├─ results/                          \# Results & outputs

│   ├─ metrics/

│   │   └─ results\_summary.json

│   ├─ visualizations/

│   │   ├─ success\_cases/

│   │   └─ failure\_cases/

│   ├─ benchmarks/

│   │   └─ h100\_benchmark.json

│   └─ inference\_outputs/

│

├─ logs/                             \# TensorBoard logs

│   └─ runs/

│

├─ solution\_presentation.pptx        \# CRITICAL: Final presentation

└─ Dockerfile                        \# Docker image (optional)

### Key Files to Prioritize

**CRITICAL (Must Have):**

1. `inference.py` — Standalone inference CLI (KLA requirement)  
2. `scripts/train.py` — Training script (reproducibility)  
3. `src/models/restorenet.py` — Model architecture  
4. `src/data/dataset.py` — Data loading  
5. `README.md` — Complete documentation  
6. `requirements.txt` — Dependencies with pinned versions  
7. `checkpoints/best_model.pt` — Trained weights  
8. `solution_presentation.pptx` — Final presentation

**IMPORTANT (Should Have):** 9\. `configs/train.yaml` — Training config 10\. `scripts/benchmark.py` — Runtime evaluation 11\. `src/training/losses.py` — Loss functions 12\. `tests/` — Unit tests

**NICE (Optional):** 13\. `notebooks/` — Exploratory analysis 14\. `Dockerfile` — Reproducibility 15\. `src/optimization/inference_engine.py` — H100 optimization

---

## IMPLEMENTATION GUIDE

### Step 0: Environment Setup

\# Clone repository

git clone https://github.com/CSNEHA20/kla-restoration.git

cd kla-restoration

\# Create virtual environment

python \-m venv venv

source venv/bin/activate  \# On Windows: venv\\Scripts\\activate

\# Install dependencies

pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \--index-url https://download.pytorch.org/whl/cu118

pip install \-r requirements.txt

\# Download data (from KLA Google Drive)

mkdir \-p data

\# Download GT.zip and NoisyLR.zip, extract to data/

\# Verify setup

python \-c "import torch; print(f'PyTorch {torch.\_\_version\_\_} on {torch.cuda.get\_device\_name()}')"

**requirements.txt:**

torch==2.1.0

torchvision==0.16.0

torchaudio==2.1.0

numpy==1.24.3

scipy==1.11.0

Pillow==10.0.0

PyYAML==6.0

tqdm==4.66.0

scikit-image==0.21.0

tensorboard==2.14.0

lpips==0.1.4

---

### Step 1: Dataset Inspection

\# Inspect official dataset

python scripts/inspect\_dataset.py \\

    \--gt\_dir data/GT \\

    \--noisylr\_dir data/NoisyLR \\

    \--sample\_size 100

\# Output: Statistics on image shapes, value ranges, pairing

**Expected output:**

\================================================================================

DATASET INSPECTION REPORT

\================================================================================

GT IMAGES:

  File count (sampled): 100

  Unique shapes: {(256, 256), (512, 512)}

  Value range: \[0.0001, 1.0000\]

  All values in \[0,1\]? True

NOISYLR IMAGES:

  File count (sampled): 100

  Unique shapes: {(256, 256), (512, 512)}

  Value range: \[-0.0159, 1.5406\]

  Outside \[0,1\]? True

  Negative values present? True

PAIR VALIDATION:

  Matched pairs: 100 / 100

\================================================================================

---

### Step 2: Generate Synthetic Degradation Pairs

\# Generate synthetic augmentation

python scripts/generate\_synthetic\_pairs.py \\

    \--gt\_dir data/GT \\

    \--output\_dir data/NoisyLR\_synth \\

    \--samples\_per\_image 2

**Output:**

Generated 600 synthetic degraded pairs (300 images × 2 samples each)

---

### Step 3: Train Baseline Model

\# Train baseline CNN

python scripts/train\_baseline.py \\

    \--config configs/baseline.yaml \\

    \--output\_dir checkpoints/baseline

\# Monitor with TensorBoard

tensorboard \--logdir logs/

**Expected result after 50 epochs:**

Epoch 50 complete. Avg Loss \= 0.0234

Validation PSNR: 25.3 dB | SSIM: 0.72

Saved checkpoint: checkpoints/baseline\_epoch\_50.pt

---

### Step 4: Sanity Checks

**4a. Overfit Single Image:**

\# src/sanity\_check.py

import torch

from src.models.restorenet import RestoreNet

from src.data.dataset import RestorationDataset

\# Load single image

dataset \= RestorationDataset('data/GT', 'data/NoisyLR')

noisylr, gt \= dataset\[0\]

\# Model

model \= RestoreNet().cuda()

optimizer \= torch.optim.Adam(model.parameters(), lr=0.01)

loss\_fn \= torch.nn.L1Loss()

\# Overfit loop

for epoch in range(100):

    pred \= model(noisylr.unsqueeze(0).cuda())

    loss \= loss\_fn(pred, gt.unsqueeze(0).cuda())

    

    optimizer.zero\_grad()

    loss.backward()

    optimizer.step()

    

    if (epoch \+ 1\) % 10 \== 0:

        print(f"Epoch {epoch+1}: Loss \= {loss.item():.6f}")

\# After 100 epochs, loss should approach 0 (overfitting proof)

**Expected:** Loss converges to near-zero, proving:

- Data loading works  
- Model forward pass works  
- Backprop works  
- GPU transfers work

---

### Step 5: Train Final Model

\# Full training

python scripts/train.py \\

    \--config configs/train.yaml \\

    \--exp\_name final\_model\_v1 \\

    \--seed 42

\# Monitor

tensorboard \--logdir logs/

\# Expected training time: 8–12 hours (single H100)

---

### Step 6: Evaluate

\# Comprehensive evaluation

python scripts/evaluate.py \\

    \--model\_path checkpoints/best\_model.pt \\

    \--val\_data\_dir data/NoisyLR \\

    \--gt\_data\_dir data/GT \\

    \--output\_dir results/

\# Expected output:

\# PSNR: 27.5 dB

\# SSIM: 0.78

\# LPIPS: 0.15

\# Runtime: 42 ms/image (H100)

---

### Step 7: Benchmark Inference

\# H100 benchmark

python scripts/benchmark.py \\

    \--model\_path checkpoints/best\_model.pt \\

    \--input\_dir data/NoisyLR \\

    \--output\_dir results/benchmarks \\

    \--num\_images 100 \\

    \--device cuda \\

    \--batch\_size 8

\# Expected:

\# End-to-end time: 42±3 ms/image

\# Throughput: 23.8 images/sec

---

### Step 8: Generate Presentation Visuals

\# Create before/after comparisons

python scripts/visualize\_results.py \\

    \--input\_dir data/NoisyLR \\

    \--pred\_dir results/inference\_outputs \\

    \--gt\_dir data/GT \\

    \--output\_dir results/visualizations \\

    \--num\_samples 20

---

### Step 9: Final Dry Run

**CRITICAL:** Simulate KLA evaluator environment.

\# Fresh directory (no hardcoded paths)

mkdir \-p eval\_test/input eval\_test/output

\# Copy test images

cp data/NoisyLR/\*.npy eval\_test/input/

\# Run inference (EXACTLY as KLA would)

python inference.py \\

    \--input\_dir eval\_test/input \\

    \--output\_dir eval\_test/output

\# Verify outputs

ls eval\_test/output/ | wc \-l  \# Should equal input count

file eval\_test/output/\*.npy     \# Verify .npy format

---

## TESTING & VERIFICATION

### Unit Tests

\# tests/test\_dataset.py

import pytest

import torch

from src.data.dataset import RestorationDataset

def test\_dataset\_loading():

    """Test dataset loads pairs correctly."""

    dataset \= RestorationDataset('data/GT', 'data/NoisyLR')

    

    assert len(dataset) \> 0, "Dataset is empty"

    

    noisylr, gt \= dataset\[0\]

    assert noisylr.shape \== gt.shape, "Shape mismatch"

    assert noisylr.dtype \== torch.float32

    assert gt.dtype \== torch.float32

def test\_dataset\_no\_clipping():

    """Test that NoisyLR values outside \[0,1\] are preserved."""

    dataset \= RestorationDataset('data/GT', 'data/NoisyLR', normalize=False)

    noisylr, \_ \= dataset\[0\]

    

    \# At least one value should be outside \[0,1\]

    assert (noisylr \< 0).any() or (noisylr \> 1).any(), \\

        "NoisyLR clipped; should preserve out-of-range values"

def test\_dataloader\_batching():

    """Test DataLoader batching."""

    from torch.utils.data import DataLoader

    

    dataset \= RestorationDataset('data/GT', 'data/NoisyLR')

    loader \= DataLoader(dataset, batch\_size=4, num\_workers=0)

    

    noisylr\_batch, gt\_batch \= next(iter(loader))

    assert noisylr\_batch.shape\[0\] \== 4, "Batch size mismatch"

    assert noisylr\_batch.shape\[1\] \== 1, "Should have 1 channel (grayscale)"

### Integration Tests

\# tests/test\_inference.py

def test\_inference\_pipeline():

    """Test full inference pipeline."""

    from src.models.restorenet import RestoreNet

    import torch

    

    model \= RestoreNet().cuda().eval()

    

    \# Dummy input \[B, 1, H, W\]

    dummy\_input \= torch.randn(1, 1, 256, 256).cuda()

    

    with torch.inference\_mode():

        output \= model(dummy\_input)

    

    \# Verify output properties

    assert output.shape \== dummy\_input.shape, "Shape mismatch"

    assert output.dtype \== torch.float32

    assert not torch.isnan(output).any(), "NaN in output"

    assert not torch.isinf(output).any(), "Inf in output"

### Reproducibility Tests

\# Run training twice with same seed; results should match

python scripts/train.py \--config configs/train.yaml \--seed 42 \--output\_dir exp1

python scripts/train.py \--config configs/train.yaml \--seed 42 \--output\_dir exp2

\# Compare final metrics

python \-c "

import json

with open('exp1/metrics.json') as f:

    m1 \= json.load(f)

with open('exp2/metrics.json') as f:

    m2 \= json.load(f)

    

assert abs(m1\['final\_psnr'\] \- m2\['final\_psnr'\]) \< 0.01, 'Results not reproducible'

print('✓ Reproducible results')

"

---

## DEPLOYMENT & DEMO

### Web-Based Demo (Optional)

\# demo/backend/app.py

from flask import Flask, request, jsonify, send\_file

import torch

from pathlib import Path

import numpy as np

from src.models.restorenet import RestoreNet

import io

import time

app \= Flask(\_\_name\_\_)

\# Load model once

model \= RestoreNet().cuda().eval()

checkpoint \= torch.load('checkpoints/best\_model.pt', map\_location='cuda')

model.load\_state\_dict(checkpoint\['model\_state'\] if isinstance(checkpoint, dict) else checkpoint)

@app.route('/api/restore', methods=\['POST'\])

def restore():

    """REST API for image restoration."""

    

    if 'image' not in request.files:

        return jsonify({'error': 'No image provided'}), 400

    

    file \= request.files\['image'\]

    

    try:

        \# Load image

        image\_data \= np.load(io.BytesIO(file.read()))

        image\_tensor \= torch.from\_numpy(image\_data\[np.newaxis, np.newaxis, ...\]).cuda()

        

        \# Inference

        start \= time.time()

        with torch.inference\_mode():

            output \= model(image\_tensor)

        inference\_time \= time.time() \- start

        

        \# Postprocess

        output\_np \= np.clip(output.cpu().squeeze().numpy(), 0, 1\)

        

        \# Return as bytes

        output\_bytes \= io.BytesIO()

        np.save(output\_bytes, output\_np)

        output\_bytes.seek(0)

        

        return send\_file(

            output\_bytes,

            mimetype='application/octet-stream',

            as\_attachment=True,

            download\_name='restored.npy'

        ), 200

    

    except Exception as e:

        return jsonify({'error': str(e)}), 500

@app.route('/api/info', methods=\['GET'\])

def info():

    """Model information."""

    return jsonify({

        'model': 'RestoreNet',

        'version': '1.0',

        'input\_format': 'uint8 or float32 grayscale',

        'input\_shape': '(H, W)',

        'output\_format': 'float32 \[0,1\]',

        'output\_shape': '(H, W)',

        'supported\_sizes': \['128×128', '256×256', '512×512'\],

        'expected\_runtime': '35–50 ms (H100)'

    })

if \_\_name\_\_ \== '\_\_main\_\_':

    app.run(host='0.0.0.0', port=5000, debug=False)

\<\!-- demo/frontend/index.html \--\>

\<\!DOCTYPE html\>

\<html\>

\<head\>

    \<title\>Image Restoration Demo\</title\>

    \<style\>

        body { font-family: Arial; margin: 20px; }

        .container { max-width: 1200px; margin: 0 auto; }

        .upload-area { border: 2px dashed \#ccc; padding: 20px; text-align: center; }

        .images { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 20px; }

        .image-wrapper { border: 1px solid \#ddd; padding: 10px; }

        img { width: 100%; max-height: 400px; }

    \</style\>

\</head\>

\<body\>

    \<div class="container"\>

        \<h1\>Image Restoration Demo\</h1\>

        

        \<div class="upload-area"\>

            \<input type="file" id="imageInput" accept=".npy" /\>

            \<button onclick="uploadImage()"\>Restore\</button\>

        \</div\>

        

        \<div class="images"\>

            \<div class="image-wrapper"\>

                \<h3\>Original (Degraded)\</h3\>

                \<img id="original" /\>

            \</div\>

            \<div class="image-wrapper"\>

                \<h3\>Restored\</h3\>

                \<img id="restored" /\>

            \</div\>

        \</div\>

        

        \<div id="metrics"\>\</div\>

    \</div\>

    

    \<script\>

        function uploadImage() {

            const file \= document.getElementById('imageInput').files\[0\];

            if (\!file) return;

            

            const formData \= new FormData();

            formData.append('image', file);

            

            fetch('/api/restore', { method: 'POST', body: formData })

                .then(r \=\> r.blob())

                .then(blob \=\> {

                    const url \= URL.createObjectURL(blob);

                    document.getElementById('restored').src \= url;

                    document.getElementById('metrics').innerHTML \= '\<p\>✓ Restored\</p\>';

                });

        }

    \</script\>

\</body\>

\</html\>

---

## PRESENTATION STRATEGY

### Recommended Slide Deck Structure (12–15 slides)

| Slide | Title | Key Points | Visual |
| :---- | :---- | :---- | :---- |
| 1 | Title \+ Team | Project name, team members, date | Title slide |
| 2 | Problem | Degradation model (speckle \+ Gaussian \+ downsample), impact on inspection | Diagram: Degradation |
| 3 | Dataset | KLA dataset specs, sample statistics, pairing, value ranges | Table: Dataset stats |
| 4 | Approach | End-to-end pipeline overview | Flow diagram |
| 5 | Architecture | RestoreNet: upsample \+ residual blocks \+ attention | Architecture diagram |
| 6 | Loss Function | Multi-term objective (L1 \+ SSIM \+ LPIPS) | Loss equation \+ weighting |
| 7 | Training | Data split, augmentation, synthetic pairs, schedule | Config summary |
| 8 | Baseline | Simple baseline performance (PSNR/SSIM/LPIPS) | Bar chart: Metrics |
| 9 | Results | Final model performance (ID \+ OOD) | Line plot: Metrics over epochs |
| 10 | Runtime | H100 benchmarking, end-to-end throughput | Bar chart: Latency breakdown |
| 11 | Visual Results | Before/after examples (success \+ failure) | Image grid (4 examples) |
| 12 | Ablation | Key design decisions (loss weights, blocks, attention) | Table: Ablation results |
| 13 | Limitations | Known failure modes, future work | Bullet points |
| 14 | Resources | External datasets/models used, licenses | Table: External resources |
| 15 | Conclusion | Key achievements, impact statement | Summary \+ repo link |

### Presentation Notes

**Slide 1 (Title):**

- Project name: "RestoreNet: Degradation-Aware Fidelity-First Restoration"  
- One-liner: "AI system for semiconductor image restoration that prioritizes fidelity over hallucination"  
- Team, institution, date

**Slide 2 (Problem):**

- Degradation: Speckle × Gaussian noise × downsampling (unknown order)  
- Impact: Hides fine structures, reduces downstream CV reliability  
- Example: Show side-by-side degraded/clean pair

**Slide 4 (Approach):**

NoisyLR → Bilinear Upsample → Residual CNN → Clipped Output → Restored GT

**Slide 5 (Architecture):**

Input \[B, 1, H, W\]

  ↓

Upsample ×scale\_factor

  ↓

Conv(1→64)

  ↓

×10 \[ResBlock \+ CA\]

  ↓

Conv(64→1)

  ↓

Add residual

  ↓

Output \[B, 1, H, W\]

**Slide 9 (Results \- KEY SLIDE):**

              Baseline    Final Model   Improvement

PSNR (dB):    25.3        27.8          \+2.5 dB ↑

SSIM:         0.72        0.81          \+0.09

LPIPS:        0.19        0.12          \-0.07 ↓ (better)

Runtime:      55 ms       42 ms         \-13 ms ↓ (faster)

Add separate ID/OOD metrics if tested.

**Slide 11 (Visual Results):**

- Grid layout: \[Input | GT | Pred | Diff\]  
- 2 success cases (high PSNR)  
- 2 failure cases (low PSNR) with candid discussion

**Slide 12 (Ablation):** | Component | PSNR Impact | Notes | |-----------|------------|-------| | L1 only | 25.8 | Baseline | | \+SSIM (λ=0.3) | 26.5 | \+0.7 dB | | \+LPIPS (λ=0.1) | 27.8 | \+0.6 dB | | \+Attention | 27.9 | \+0.1 dB | | \+Synthetic Aug | 28.1 | \+0.2 dB (if tested) |

---

## RISK REGISTER & MITIGATION

| Risk | Probability | Impact | Mitigation |
| :---- | :---- | :---- | :---- |
| **Training divergence** | Medium | High | Early stopping, gradient clipping, conservative LR (1e-3) |
| **OOD generalization gap** | Medium | High | Synthetic augmentation, validation on held-out set, conservative architecture |
| **Memory OOM** | Low | High | Reduce batch size, patch-based training, half-precision if needed |
| **Inference too slow** | Low | High | torch.compile, batch processing, profile hotspots |
| **Data leakage** | Low | High | Careful train/val split, no metadata overlap |
| **Numerical instability** | Low | Medium | Use Charbonnier loss, clip gradients, monitor for NaNs |
| **Model not converging** | Medium | High | Simpler architecture, stronger L1 term, warmup |
| **Reproducibility issues** | Low | High | Pin seed, document versions, freeze dependencies |
| **Submission deadline miss** | Low | High | Early checkpoint, fallback to baseline if final doesn't converge |

### Contingency Plans

**If final model doesn't converge:**

1. Fall back to baseline (already trained)  
2. Increase L1 weight (λ\_pixel → 2.0)  
3. Reduce learning rate (1e-3 → 5e-4)  
4. Train longer (100 → 150 epochs)

**If inference too slow:**

1. Reduce batch size (8 → 1\)  
2. Reduce image resolution for testing  
3. Use torch.compile() mode='reduce-overhead'  
4. Skip LPIPS computation during validation

**If PSNR/SSIM not competitive:**

1. Check for data loading bugs (out-of-range clipping?)  
2. Verify loss weights are correct  
3. Increase model capacity (num\_blocks: 10 → 15\)  
4. Longer training (100 → 200 epochs)

---

## DEVELOPMENT ROADMAP

### Week 1-2: Foundation

- [ ] Environment setup & dependencies  
- [ ] Data inspection & analysis  
- [ ] Baseline model training (50 epochs)  
- [ ] Baseline evaluation (PSNR/SSIM/LPIPS)  
- [ ] Sanity checks (single image overfit)

### Week 3: Model Development

- [ ] Final RestoreNet architecture implementation  
- [ ] Multi-term loss function  
- [ ] Synthetic degradation augmentation  
- [ ] Training pipeline (full dataset)  
- [ ] Validation metrics

### Week 4: Optimization & Evaluation

- [ ] H100 benchmarking  
- [ ] torch.compile() evaluation  
- [ ] OOD generalization testing  
- [ ] Failure analysis  
- [ ] Ablation studies

### Week 5: Presentation & Submission

- [ ] Final model checkpoint  
- [ ] Standalone inference script validation  
- [ ] Presentation slides (12–15)  
- [ ] README & documentation  
- [ ] GitHub repo setup  
- [ ] Final dry run (clean environment)  
- [ ] Submit 2 days before deadline

---

## BUILD ORDER (STEP-BY-STEP)

### Phase 0: SETUP (Day 1, 2 hours)

**Objective:** Reproducible environment ready

\# 0.1: Clone & setup

git clone ...

cd kla-restoration

python \-m venv venv

source venv/bin/activate

pip install \-r requirements.txt

\# 0.2: Verify CUDA

python \-c "import torch; print(torch.cuda.is\_available(), torch.cuda.get\_device\_name())"

\# 0.3: Download dataset (from KLA Google Drive)

mkdir data

\# \[Download GT.zip, NoisyLR.zip and extract\]

\# SUCCESS CRITERIA:

\# \- No import errors

\# \- CUDA available

\# \- Data files present (data/GT/\*.npy, data/NoisyLR/\*.npy)

---

### Phase 1: DATA INSPECTION (Day 1-2, 1 hour)

**Objective:** Understand dataset before training

\# 1.1: Inspect official dataset

python scripts/inspect\_dataset.py \\

    \--gt\_dir data/GT \\

    \--noisylr\_dir data/NoisyLR \\

    \--sample\_size 100

\# 1.2: Verify sanity

\# \- Check pair matching

\# \- Check value ranges (GT in \[0,1\], NoisyLR possibly outside)

\# \- Check unique shapes

\# SUCCESS CRITERIA:

\# \- All pairs matched

\# \- NoisyLR has values outside \[0,1\]

\# \- Consistent shapes per image size class

---

### Phase 2: BASELINE (Day 2-3, 8 hours)

**Objective:** Establish simple working solution

\# 2.1: Train baseline

python scripts/train\_baseline.py \\

    \--config configs/baseline.yaml \\

    \--output\_dir checkpoints/baseline \\

    2\>&1 | tee logs/baseline\_train.log

\# Monitor: tensorboard \--logdir logs/

\# 2.2: Evaluate baseline

python scripts/evaluate.py \\

    \--model\_path checkpoints/baseline/model\_epoch\_50.pt \\

    \--val\_data\_dir data/NoisyLR \\

    \--gt\_data\_dir data/GT \\

    \--output\_dir results/baseline/

\# SUCCESS CRITERIA:

\# \- PSNR ≥ 24 dB (on 256×256)

\# \- SSIM ≥ 0.70

\# \- No NaNs/Infs

\# \- Inference time \< 100ms/image

\# \- Training converges (loss decreases)

---

### Phase 3: FINAL MODEL (Day 4-6, 18 hours)

**Objective:** Train production-ready model

\# 3.1: Generate synthetic data

python scripts/generate\_synthetic\_pairs.py \\

    \--gt\_dir data/GT \\

    \--output\_dir data/NoisyLR\_synth \\

    \--samples\_per\_image 2

\# 3.2: Train final model

python scripts/train.py \\

    \--config configs/train.yaml \\

    \--exp\_name final\_v1 \\

    \--seed 42 \\

    2\>&1 | tee logs/final\_train.log

\# \[Let run \~12 hours\]

\# Monitor: tensorboard \--logdir logs/

\# 3.3: Evaluate final model

python scripts/evaluate.py \\

    \--model\_path checkpoints/best\_model.pt \\

    \--val\_data\_dir data/NoisyLR \\

    \--gt\_data\_dir data/GT \\

    \--output\_dir results/final/

\# SUCCESS CRITERIA:

\# \- PSNR ≥ 27 dB (\>2.5 dB improvement over baseline)

\# \- SSIM ≥ 0.78

\# \- LPIPS ≤ 0.15

\# \- Training converged (plateau in loss)

\# \- No NaNs/Infs

---

### Phase 4: OPTIMIZATION (Day 6-7, 6 hours)

**Objective:** Ensure fast inference

\# 4.1: Profile model

python \-c "

from src.models.restorenet import RestoreNet

import torch

model \= RestoreNet().cuda()

model.eval()

dummy \= torch.randn(1, 1, 256, 256).cuda()

with torch.profiler.profile() as prof:

    with torch.inference\_mode():

        for \_ in range(10):

            model(dummy)

print(prof.key\_averages().table(sort\_by='cpu\_time\_total'))

"

\# 4.2: Benchmark with torch.compile()

python scripts/benchmark\_compile.py

\# 4.3: Full inference benchmark

python scripts/benchmark.py \\

    \--model\_path checkpoints/best\_model.pt \\

    \--input\_dir data/NoisyLR \\

    \--output\_dir results/benchmarks \\

    \--num\_images 100 \\

    \--device cuda

\# SUCCESS CRITERIA:

\# \- End-to-end time \< 50ms/image (H100)

\# \- torch.compile() speeds up or no regression

\# \- Throughput ≥ 20 images/sec

---

### Phase 5: DRY RUN (Day 7, 2 hours)

**Objective:** Simulate KLA evaluator environment

\# 5.1: Create clean test environment

mkdir \-p dry\_run/input dry\_run/output

cp data/NoisyLR/\*.npy dry\_run/input/  \# First 50 for speed

\# 5.2: Run inference (as KLA would)

python inference.py \\

    \--input\_dir dry\_run/input \\

    \--output\_dir dry\_run/output \\

    \--model\_path checkpoints/best\_model.pt \\

    \--device cuda

\# 5.3: Verify outputs

ls \-lh dry\_run/output/\*.npy | wc \-l  \# Should match input count

file dry\_run/output/\*.npy              \# All should be numpy format

\# 5.4: Quick manual metric check

python \-c "

import numpy as np

from skimage.metrics import peak\_signal\_noise\_ratio

input\_files \= sorted(glob('data/NoisyLR/\*.npy'))\[:5\]

gt\_files \= sorted(glob('data/GT/\*.npy'))\[:5\]

pred\_files \= sorted(glob('dry\_run/output/\*.npy'))\[:5\]

psnrs \= \[\]

for pred\_f, gt\_f in zip(pred\_files, gt\_files):

    pred \= np.load(pred\_f)

    gt \= np.load(gt\_f)

    psnr \= peak\_signal\_noise\_ratio(gt, pred, data\_range=1.0)

    psnrs.append(psnr)

print(f'Spot check PSNR: {np.mean(psnrs):.2f} ± {np.std(psnrs):.2f} dB')

"

\# SUCCESS CRITERIA:

\# \- Number of outputs matches inputs

\# \- All files are .npy format

\# \- Output PSNR reasonable (\~27 dB)

\# \- No errors during inference

\# \- Inference completes in \<5 min for 50 images

---

### Phase 6: PRESENTATION (Day 8, 4 hours)

**Objective:** Create compelling presentation

\# 6.1: Collect results

cp results/final/\*.json results/

mkdir \-p results/visualizations

\# 6.2: Generate visuals

python scripts/visualize\_results.py \\

    \--input\_dir data/NoisyLR \\

    \--pred\_dir dry\_run/output \\

    \--gt\_dir data/GT \\

    \--output\_dir results/visualizations \\

    \--num\_samples 8

\# 6.3: Create presentation

\# \[Use PowerPoint/Google Slides template\]

\# \[Follow 12-15 slide structure from Presentation Strategy\]

\# \[Include generated visualizations and metrics\]

\# 6.4: Proofread

\# \- Check all links work (GitHub, datasets)

\# \- Verify metrics are correct

\# \- Spell check

\# \- Test slide transitions

\# SUCCESS CRITERIA:

\# \- 12–15 professional slides

\# \- All visuals high-quality

\# \- All metrics accurate

\# \- No typos

\# \- Fits 5–7 min presentation

---

### Phase 7: SUBMISSION (Day 9, 2 hours)

**Objective:** Package and submit

\# 7.1: Final repo checks

git status  \# Nothing uncommitted

git log     \# Commit history clean

\# 7.2: Verify all required files present

ls README.md inference.py scripts/train.py src/ configs/ checkpoints/best\_model.pt

file solution\_presentation.pptx

\# 7.3: Create submission archive

tar \-czf kla\_submission\_final.tar.gz \\

    README.md \\

    inference.py \\

    scripts/ \\

    src/ \\

    configs/ \\

    checkpoints/best\_model.pt \\

    requirements.txt \\

    solution\_presentation.pptx

\# 7.4: Final checks

python \-m py\_compile inference.py src/\*\*/\*.py  \# No syntax errors

\# 7.5: Submit

\# \[Upload to KLA hackathon portal before deadline\]

\# \[Verify receipt email received\]

\# SUCCESS CRITERIA:

\# \- All required files included

\# \- No syntax errors

\# \- File size reasonable (\~100MB with model)

\# \- Submission confirmed by portal

---

### Critical Path Dependency

SETUP (2h)

  ↓

DATA INSPECTION (1h)

  ↓

BASELINE (8h) ← Can parallelize with infrastructure

  ↓

FINAL MODEL (18h) ← Critical path, must start early

  ↓

OPTIMIZATION (6h)

  ↓

DRY RUN (2h)

  ↓

PRESENTATION (4h)

  ↓

SUBMISSION (2h)

Total: \~43 hours active work \+ 12-18 hours passive training

**Key:** Start training (Phase 3\) as soon as possible. Don't wait for perfect baseline.

---

## FINAL CHECKLIST

### Pre-Submission (48 hours before deadline)

- [ ] All dependencies frozen in requirements.txt  
- [ ] Inference script tested (no manual edits needed)  
- [ ] Model weights downloadable/included  
- [ ] README complete with exact commands  
- [ ] All external resources disclosed  
- [ ] No hardcoded paths in code  
- [ ] GitHub repo public & accessible  
- [ ] Presentation slides complete (12–15)  
- [ ] PSNR, SSIM, LPIPS reported  
- [ ] Baseline included for comparison  
- [ ] Runtime measurements on H100 (or equivalent GPU)  
- [ ] OOD evaluation (if attempted)  
- [ ] Failure analysis included  
- [ ] No confidential data in repo

### Day-of-Submission

- [ ] Final push to GitHub  
- [ ] Download & test inference script in clean environment  
- [ ] Verify presentation PDFs render correctly  
- [ ] Double-check submission portal requirements  
- [ ] Submit with 1 hour to spare  
- [ ] Verify confirmation email received

---

## CONCLUSION

This implementation guide provides a **complete roadmap for building a competitive KLA hackathon submission**.

**Core Philosophy:**

> **Build simple, reproducible, empirically-grounded systems that prioritize fidelity over cleverness.**

**Key Strengths of This Approach:**

1. **Engineering rigor:** Clean code, reproducible, tested  
2. **Practical efficiency:** Optimizes full pipeline, not just model  
3. **Conservative risk:** Baseline as fallback, proven architectures  
4. **Fidelity-first:** Avoids hallucination; preserves observed information  
5. **OOD awareness:** Tests on unfamiliar content  
6. **Production-ready:** Standalone inference, no code editing

**Expected Outcomes:**

- PSNR: 27–28 dB (\>2.5 dB over baseline)  
- SSIM: 0.78–0.80  
- LPIPS: 0.12–0.15  
- Runtime: 40–50 ms/image (H100)  
- Reproducibility: ✓ (documented, seeded, tested)

**Timeline:** 4–5 weeks from start to submission

**Success Criteria:** Beat baseline significantly, demonstrate engineering discipline, run cleanly on evaluators' hardware.

---

**Document Version:** 1.0  
**Last Updated:** August 2026  
**Prepared for:** SEMICON India Hackathon 2026 / KLA Problem Statement  
**Status:** Ready for implementation  
