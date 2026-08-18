# RestoreNet — Completion & Production Prompts
## KLA SEMICON India Hackathon 2026

---

## 🔍 AUDIT REPORT — What Exists vs What's Missing

### ✅ COMPLETE & SOLID
| File | Status |
|---|---|
| `src/models/restorenet.py` | ✅ Full, clean, correct |
| `src/models/baseline.py` | ✅ Full |
| `src/models/blocks.py` | ✅ Full |
| `src/training/losses.py` | ✅ SSIM + Charbonnier + RestorationLoss with LPIPS fallback |
| `src/training/metrics.py` | ✅ PSNR, SSIM, LPIPS, MetricsTracker |
| `src/training/trainer.py` | ✅ Full AMP training loop, early stopping, checkpointing, TensorBoard |
| `src/utils/visualization.py` | ✅ Full |
| `src/utils/seed.py` | ✅ |
| `src/utils/logging.py` | ✅ |
| `src/optimization/inference_engine.py` | ✅ |
| `src/optimization/torchscript_export.py` | ✅ |
| `inference.py` (root) | ✅ KLA-ready CLI |
| `kla-image-restoration/inference.py` | ✅ |
| `configs/*.yaml` | ✅ All 5 configs |
| `tests/test_*.py` | ✅ 5 test files exist |
| `README.md` | ✅ Good |
| `requirements.txt` | ✅ |

### ❌ CRITICAL — MISSING (Breaks Everything)
| What | Why It's Critical |
|---|---|
| `src/data/` — **ENTIRE DIRECTORY MISSING** | `dataset.py`, `augmentation.py`, `split.py`, `loader.py` are all absent. Every script that imports them will crash immediately. This is the #1 blocker. |
| `src/training/validation.py` | File is empty (only has `# Validation loop` comment). |
| `checkpoints/best_model.pt` | No trained weights exist. Inference runs but produces random noise (PSNR ~2.9 dB in results JSON confirms this). |

### ⚠️ INCOMPLETE — Partially Done
| What | Issue |
|---|---|
| Results metrics | PSNR 2.9 dB / SSIM 0.19 / LPIPS 0.95 → untrained random-weight outputs. Not submission-ready. |
| `scripts/dry_run.py` (root) | Checks for `src/data/dataset.py` but it doesn't exist — will always fail check #1. |
| No frontend / demo UI | KLA wants a demo. No web dashboard exists. |
| No `solution_presentation.pptx` | Required by KLA guidelines. `slide_content.md` exists but no actual .pptx. |
| `src/inference/engine.py` | Exists but is a thin wrapper; not wired to `inference.py`. |
| Benchmark results | Shows 102ms on CPU (no GPU). Target is <100ms on H100. |

---

## 🚀 COMPLETION PROMPTS — Run These In Order

---

### PROMPT C-1 — Create the Entire `src/data/` Package (CRITICAL BLOCKER)

```
The entire `src/data/` directory is missing from the project. Without it, every training, evaluation,
and dry-run script crashes. Create the following four files inside `kla-image-restoration/src/data/`:

--- FILE: src/data/__init__.py ---
(empty, just the package marker)

--- FILE: src/data/dataset.py ---
Write RestorationDataset(torch.utils.data.Dataset):
- __init__(self, gt_dir: str, noisylr_dir: str, normalize: bool = False, augment: bool = False)
- Pairs GT and NoisyLR files by matching stem names (.npy files).
- Raises ValueError if no pairs found.
- Prints "Loaded N GT/NoisyLR pairs" at init.
- __len__: returns number of pairs.
- __getitem__(idx):
  1. np.load both as float32.
  2. Assert shapes are equal OR gt is exactly 2× noisylr in H and W.
  3. DO NOT clip on load.
  4. If normalize=True: per-image normalize noisylr only: (x - mean) / (std + 1e-6).
  5. If augment=True: call _augment(gt, noisylr).
  6. Add channel dim: [1, H, W] for both. Convert to torch.float32 tensors.
  7. Return (noisylr_tensor, gt_tensor).
- _augment: random h-flip (p=0.5), v-flip (p=0.5), random rot90 (k in 0-3). Same transform for both.

--- FILE: src/data/split.py ---
Write two functions:

create_train_val_split(dataset, train_ratio=0.70, val_ratio=0.20, seed=42):
- Sets np and torch seeds.
- Returns (train_Subset, val_Subset, holdout_Subset).
- Prints split sizes.

get_dataloaders(train_ds, val_ds, batch_size=8, num_workers=4):
- Returns (train_loader, val_loader).
- train: shuffle=True, pin_memory=True, prefetch_factor=2, drop_last=False.
- val: shuffle=False, pin_memory=True.

--- FILE: src/data/augmentation.py ---
Write SyntheticDegradationAugmentor:
- __init__(self, noise_std_range=(0.01, 0.05), speckle_range=(0.5, 1.5), downsample_factors=(2, 3, 4))
- apply_speckle(img, strength): img * uniform(1/strength, strength). DO NOT clip.
- apply_gaussian(img, std): img + normal(0, std). DO NOT clip.
- apply_downsample(img, factor): scipy.ndimage.zoom(img, 1/factor, order=1). If factor==1 return img unchanged.
- generate_synthetic_pair(gt: np.ndarray) -> np.ndarray:
  Random permutation of [speckle, gaussian, downsample] applied in random order.
- augment_dataset(gt_dir: str, output_dir: str, samples_per_image: int = 2) -> int:
  Generates synthetic NoisyLR and saves as {stem}_synth_{i}.npy. Returns count.

Also write SyntheticRestorationDataset(torch.utils.data.Dataset):
- __init__(self, gt_dir: str, augmentor: SyntheticDegradationAugmentor = None, augment: bool = True)
- Generates NoisyLR on-the-fly in __getitem__ using augmentor.generate_synthetic_pair.
- Returns (noisy_tensor [1,H,W], gt_tensor [1,H,W]) as float32.

--- FILE: src/data/loader.py ---
Write a single function build_combined_dataloader(gt_dir, noisylr_dir, config) -> (DataLoader, DataLoader, DataLoader):
- Creates official RestorationDataset.
- If config['data']['include_synthetic'] is True, also creates SyntheticRestorationDataset.
- Combines them with ConcatDataset.
- Calls create_train_val_split then get_dataloaders.
- Returns train_loader, val_loader, holdout_loader.

After writing all files, verify by running:
python -c "from src.data.dataset import RestorationDataset; print('OK')"
```

**Success check:** `python -c "from src.data.dataset import RestorationDataset; from src.data.split import create_train_val_split; from src.data.augmentation import SyntheticDegradationAugmentor; print('All data imports OK')"` prints OK.

---

### PROMPT C-2 — Fix `src/training/validation.py` (Empty File)

```
The file src/training/validation.py currently contains only a comment: "# Validation loop".
Fill it with a complete standalone validation module.

Write a validate(model, val_loader, device, verbose=False) function that:
- Puts model in eval mode.
- Iterates val_loader with torch.no_grad().
- For each batch: runs forward pass, computes per-image PSNR and SSIM.
- Returns a dict: {'val_psnr': mean_psnr, 'val_ssim': mean_ssim, 'n_images': count}
- Prints a clean summary line.

Also write validate_with_lpips(model, val_loader, device, max_batches=10) -> dict:
- Same as above but also computes LPIPS (limited to max_batches to save time).
- Returns {'val_psnr', 'val_ssim', 'val_lpips'}.

Both functions should import compute_psnr, compute_ssim from src.training.metrics.
```

**Success check:** `from src.training.validation import validate` imports cleanly.

---

### PROMPT C-3 — Wire `scripts/train.py` Complete `main()` Function

```
The file scripts/train.py has a partial main() — the function signature exists but the body is
missing or incomplete. Complete main() with the following logic:

1. parse_args() — already defined.
2. load_config(args.config) → config dict.
3. Override config with CLI args (if --epochs, --batch_size, --device are passed, override config values).
4. Set config device: if args.device, use it; else use config.get('device', 'cuda').
5. print_system_info(device) — already defined.
6. set_seed(config.get('seed', 42)).
7. pprint the config for reproducibility.

8. Determine gt_dir and noisylr_dir:
   - Use args.gt_dir if provided.
   - Else fall back to: os.path.join(config.get('data_root', './data'), 'GT') and .../NoisyLR.
   - If neither directory exists, print error and sys.exit(1).

9. Build dataset:
   - official_dataset = RestorationDataset(gt_dir, noisylr_dir, normalize=False, augment=True)
   - If args.max_samples: slice dataset.pairs list to [:max_samples].
   - If config['data']['include_synthetic']:
       - augmentor = SyntheticDegradationAugmentor()
       - synth_dir = os.path.join(config.get('data_root','./data'), 'NoisyLR_synth')
       - augmentor.augment_dataset(gt_dir, synth_dir, config['data']['synthetic_samples_per_image'])
       - synth_dataset = SyntheticRestorationDataset(gt_dir, augmentor)
       - combined = ConcatDataset([official_dataset, synth_dataset])
     else:
       - combined = official_dataset

10. create_train_val_split → get_dataloaders using config batch_size and num_workers=4.

11. Build RestoreNet from config model section:
    - model = RestoreNet(scale_factor=cfg.model.scale_factor, ...).

12. Print parameter count.

13. Instantiate Trainer(model, train_loader, val_loader, config, device).

14. If args.resume: trainer.load_checkpoint(args.resume).

15. trainer.fit().

16. Print "Training complete. Best model saved at checkpoints/best_model.pt".

At the top of main(), call print_system_info.
```

**Success check:** `python scripts/train.py --help` shows all args. `python scripts/train.py --gt_dir data/GT --noisylr_dir data/NoisyLR --epochs 2` starts training.

---

### PROMPT C-4 — Real Training Run with the KLA Dataset

```
The model has never been trained — the results JSON shows PSNR ~2.9 dB which is random-weight output.
You need to run training with the actual KLA dataset now.

First, extract the dataset:
cd kla-image-restoration
unzip dataset/train.zip -d data/
unzip dataset/Test_NoisyLR.zip -d data/

Verify the structure: data/GT/ and data/NoisyLR/ should each have .npy files.
Run: python scripts/inspect_dataset.py --gt_dir data/GT --noisylr_dir data/NoisyLR

Then train the BASELINE first (fast, 50 epochs):
python scripts/train_baseline.py \
  --gt_dir data/GT \
  --noisylr_dir data/NoisyLR \
  --config configs/baseline.yaml \
  --output_dir checkpoints/

Then train RestoreNet (full, 100 epochs):
python scripts/train.py \
  --gt_dir data/GT \
  --noisylr_dir data/NoisyLR \
  --config configs/train.yaml

After training completes, run inference on the test set:
python inference.py \
  --input_dir data/test/ \
  --output_dir results/inference_outputs/ \
  --model_path checkpoints/best_model.pt \
  --device cuda \
  --verbose

Then evaluate if you have GT for validation:
python scripts/evaluate.py \
  --gt_dir data/GT \
  --pred_dir results/inference_outputs/ \
  --output_json results/metrics/results_summary.json \
  --verbose

Target metrics after training:
- PSNR > 27 dB
- SSIM > 0.78
- LPIPS < 0.15

If PSNR is stuck below 25 dB after 20 epochs, stop and investigate the data loading (check if
NoisyLR and GT spatial sizes are consistent).
```

**Success check:** Checkpoint `checkpoints/best_model.pt` saved and PSNR > 20 dB after 20 epochs.

---

### PROMPT C-5 — Fix Dry Run Script to Match Actual File Layout

```
The file scripts/dry_run.py at the project root currently checks for files that don't exist yet
(like src/data/dataset.py which was missing). Now that src/data/ is complete, update the dry_run
to also:

1. Add src/data/dataset.py, src/data/augmentation.py, src/data/split.py to the required files list.
2. Add a check that src/training/validation.py is NOT empty (size > 50 bytes).
3. Add a check that checkpoints/best_model.pt exists; if not, print a WARNING (not a failure) that
   training hasn't been done yet.
4. When running inference smoke test, use --device cpu so it works without GPU in CI.
5. After all checks, also print the actual current metric results from
   results/metrics/results_summary.json if it exists: show mean PSNR, SSIM, LPIPS.
6. The final output should show two categories:
   ✅ READY: (things that are working)
   ⚠️  PENDING: (things that still need action, like training)

Run it: python scripts/dry_run.py
```

**Success check:** Script runs end-to-end and prints a categorized checklist with no Python errors.

---

### PROMPT C-6 — Generate the Submission Presentation (PPTX)

```
The KLA guidelines require solution_presentation.pptx (12-15 slides).
The slide_content.md outline already exists at results/slide_content.md.

Read the skill at /mnt/skills/public/pptx/SKILL.md first.

Then create a professional 14-slide PowerPoint at the repo root: solution_presentation.pptx

Slide content:
1. TITLE: "RestoreNet: Degradation-Aware Fidelity-First Image Restoration" | KLA SEMICON India 
   Hackathon 2026 | Team: [Your team name]

2. PROBLEM STATEMENT: Three degradations (Gaussian noise, Speckle noise, Downsampling) in unknown 
   order. Input: NoisyLR ∈ [-0.05, 1.4]. Output: GT ∈ [0, 1]. Challenge: single model must 
   handle all 6 possible orderings.

3. KEY INSIGHT: "Order-Agnostic Unified Restoration" — a single forward pass learns the inverse 
   of the composite degradation without explicitly identifying degradation order.

4. ARCHITECTURE DIAGRAM: Draw a text-based pipeline chart:
   NoisyLR [B,1,128,128]
   → Bilinear Upsample ×2 [B,1,256,256]
   → Conv2d (1→64)
   → 10× ResidualBlock + ChannelAttention (every 5th)
   → Conv2d (64→1) [residual]
   + Upsampled Input (global skip connection)
   = Restored Output [B,1,256,256]
   Parameters: ~1.6M | Inference: <100ms

5. FIDELITY-FIRST PHILOSOPHY: Three principles: (1) No clipping on load, (2) Residual learning = 
   learn only the correction, not the full image, (3) Conservative loss weights avoid hallucination.

6. DATA PIPELINE: Official pairs + 2× synthetic augmentation (random degradation order). 
   70/20/10 train/val/holdout split. Seed 42 for reproducibility.

7. LOSS FUNCTION: L_total = 1.0×L1 + 0.3×(1-SSIM) + 0.1×LPIPS
   Table: Term | Weight | Purpose
   L1 Pixel | 1.0 | Reconstruction fidelity
   1-SSIM   | 0.3 | Structural consistency  
   LPIPS    | 0.1 | Perceptual alignment

8. TRAINING STRATEGY: Adam (lr=1e-3) + CosineAnnealingLR. AMP mixed precision. 
   Early stopping (patience=20). Gradient clipping (max_norm=1.0). 100 epochs.

9. QUANTITATIVE RESULTS: Table showing:
   | Model | PSNR (dB) | SSIM | LPIPS | Runtime |
   | Baseline CNN (L1) | XX.XX | 0.XXX | 0.XXX | XXms |
   | RestoreNet (Full) | XX.XX | 0.XXX | 0.XXX | XXms |
   | Improvement | +X.XX | +0.XXX | -0.XXX | — |
   (Fill these from actual trained results)

10. ABLATION STUDY: Table showing contribution of each loss term and attention blocks.
    (Sourced from results/metrics/ablation_results.json)

11. VISUAL COMPARISONS: Include comparison_grid.png from results/visualizations/ if available.
    Caption: "NoisyLR | RestoreNet Prediction | Ground Truth"

12. OOD ROBUSTNESS: Model generalizes to unseen content. OOD delta PSNR vs in-distribution.
    (Sourced from results/metrics/ood_results.json)

13. RUNTIME PERFORMANCE: H100-targeted inference pipeline.
    Table: Mode | Mean ms | P95ms | Throughput
    Eager | XX | XX | XX img/s
    torch.compile | XX | XX | XX img/s
    Include: end-to-end latency including disk I/O.

14. CONCLUSION & FUTURE WORK:
    ✅ PSNR XX.XX dB | SSIM 0.XXX | LPIPS 0.XXX | Runtime XXms
    Engineering wins: Reproducible, tested, no hardcoded paths, standalone inference.
    Future: Diffusion refinement, ensemble, multi-scale cascade.

Save as: solution_presentation.pptx at the repository root.
```

**Success check:** File `solution_presentation.pptx` exists at repo root and is openable.

---

## 🎨 FRONTEND / DEMO UI PROMPTS

> KLA hackathons award extra points for demos. Below are two prompts:
> one for the Stitch design tool, and one for building it with Antigravity.

---

### PROMPT D-1 — Stitch Design Prompt (Visual UI Design)

```
Design a sleek, professional web-based demo interface for "RestoreNet" — an AI image restoration
system for semiconductor inspection. The design should look like a real production engineering tool,
NOT a consumer app.

DESIGN DIRECTION:
- Dark theme: Background #0A0E1A (deep navy). Surface cards: #111827. Accent: #00D4FF (KLA cyan).
- Typography: Inter or JetBrains Mono for labels. Professional, clean, no decorative elements.
- Brand feel: Think NVIDIA DIGITS / industrial vision software, not consumer SaaS.

LAYOUT — Single-page app with 3 sections:

SECTION 1 — HEADER (top bar, full width):
- Left: "RestoreNet" logo with circuit-board icon. Subtitle: "KLA Semiconductor Image Restoration"
- Right: Status pill "Model Ready" (green dot) or "Loading..." (orange dot)
- Background: #0D1117 with a subtle cyan bottom border

SECTION 2 — MAIN PANEL (two-column grid):
LEFT COLUMN — Upload & Controls:
- Large dashed upload zone: "Drop .npy file here or click to upload"
  Icon: upload arrow. Subtitle: "Accepts NumPy .npy float32 arrays"
- Below upload zone: a compact "Image Info" card showing:
  - Shape: 128×128
  - Value Range: [-0.02, 1.54]
  - Dtype: float32
  - Status: ● Degraded (or ● Clean)
- "Restore Image" button: large, full-width, cyan gradient (#00D4FF → #0096FF), 
  with a loading spinner state when processing.
- Advanced Options accordion (collapsed by default):
  - Scale Factor slider: 1×, 2×, 4×
  - Device: CPU / GPU toggle
  - Use torch.compile: toggle switch

RIGHT COLUMN — Results Display:
- Three side-by-side image panels (equal width):
  Panel 1: "Input (NoisyLR)" — grayscale image preview
  Panel 2: "Restored" — grayscale image preview with cyan border
  Panel 3: "Difference Map" — heatmap visualization (red = large error)
- Each panel has: label header, image area (aspect-ratio square), and bottom bar showing
  min/max/mean value.
- Below panels: Metrics bar (horizontal, 3 metric cards):
  PSNR: XX.XX dB  |  SSIM: 0.XXX  |  LPIPS: 0.XXX  |  Time: XXms

SECTION 3 — BOTTOM PANEL (full width, collapsed):
- Title: "System Info"
- Two sub-cards:
  Left: Model Stats table (Parameters, Architecture, Scale Factor, Loss)
  Right: Runtime Chart (simple bar chart: Disk I/O | Inference | Postprocess | Total)

COLOR TOKENS:
- bg-primary: #0A0E1A
- bg-surface: #111827
- bg-card: #1A2235
- accent-primary: #00D4FF
- accent-secondary: #0096FF
- text-primary: #E8EAF0
- text-muted: #6B7280
- success: #10B981
- warning: #F59E0B
- border: #1E2D45

COMPONENT DETAILS:
- All cards: 8px border radius, 1px border (#1E2D45), no drop shadow
- Buttons: uppercase letter-spacing, 600 weight, no border radius rounding > 6px
- Upload zone: 2px dashed border (#1E2D45), hover state turns cyan border
- Status pills: small (6px×6px) colored dot + label text
- Metric cards: dark bg #1A2235, center-aligned, large bold number in cyan, label below in muted gray

OUTPUT: Full Figma-ready wireframe / mockup design for the above layout.
```

---

### PROMPT D-2 — Antigravity Build Prompt (Full Working Frontend)

```
Build a complete single-page React application for the RestoreNet demo dashboard.
This is a production-quality demo for the KLA SEMICON India Hackathon 2026.

TECH STACK: React + Tailwind CSS (using only pre-defined base classes, no JIT)
Use lucide-react for icons. Use recharts for the runtime bar chart.
No external API calls during build — use mock data and simulate processing locally.

FULL COMPONENT STRUCTURE:

1. App.jsx — Root component with state management:
   States: uploadedFile, imageInfo, inputImageData, restoredImageData, 
           diffMapData, metrics, processingStatus, showAdvanced

2. Header component:
   - "RestoreNet" title with a Circuit icon (lucide)
   - Subtitle: "KLA Semiconductor Image Restoration · Phase 2 Demo"
   - Status badge: dynamic (Idle / Processing / Ready / Error)
   - All on dark bg, cyan accent color (use bg-slate-950, text-cyan-400)

3. UploadZone component:
   - Drag-and-drop zone for .npy files
   - On file drop/select: parse as ArrayBuffer → read as Float32Array → 
     detect shape as sqrt(length) × sqrt(length) for square images
   - Display Image Info card: Shape, Value Range (min/max), Dtype
   - Show a canvas preview of the raw image (grayscale, normalized for display)

4. ControlPanel component:
   - "Restore Image" button (cyan gradient style with Zap icon)
   - Loading spinner state during "processing"
   - Collapsible Advanced Options: scale factor (2x hardcoded default), device toggle (CPU/GPU)

5. ImageDisplay component with 3 panels:
   - Panel 1 (Input): render uploaded .npy data as grayscale canvas
   - Panel 2 (Restored): simulate restoration by applying a simple mock filter:
       For demo: convolve a box blur + brightness boost on the raw Float32Array
       to visually distinguish it from input. In production this calls the backend API.
   - Panel 3 (Diff Map): compute |input - restored| per pixel, render as heatmap 
       (low=blue, high=red) using canvas pixel manipulation
   - Each panel: label header, canvas element (200×200 fixed), value range footer

6. MetricsBar component:
   4 metric cards in a horizontal row: PSNR | SSIM | LPIPS | Time
   Values: mock-generated on "restore" (e.g. PSNR: 27.4 dB, SSIM: 0.812, LPIPS: 0.134, 38ms)
   Cards: bg-slate-800, large cyan number, small gray label

7. RuntimeChart component (recharts BarChart):
   Shows breakdown of inference pipeline timing:
   bars: Disk I/O (12ms), Preprocess (3ms), GPU Transfer (2ms), 
         Inference (18ms), Postprocess (2ms), Save (3ms)
   Color: cyan bars, dark background

8. SystemInfoPanel component (collapsible bottom section):
   Two sub-cards:
   - Model Stats table (static data about RestoreNet)
   - Runtime Chart (the BarChart from above)

CANVAS RENDERING LOGIC:
For grayscale .npy → canvas:
  pixels are Float32 in [0,1] range (for GT) or [-0.1, 1.5] for NoisyLR
  Normalize for display: val_display = clip((val - min)/(max - min), 0, 1)
  Set ImageData: R = G = B = Math.round(val_display * 255), A = 255

DIFF MAP HEATMAP:
  diff = |input_norm - restored_norm| per pixel (both normalized to [0,1] for display)
  Map diff to color: 
    if diff < 0.1: blue (0,0,255)
    if diff < 0.3: green (0,255,0)
    if diff >= 0.3: red (255,0,0)
  Interpolate smoothly between these.

DARK THEME CLASSES (Tailwind):
  bg: bg-slate-950 (main), bg-slate-900 (cards), bg-slate-800 (inner cards)
  text: text-slate-100 (primary), text-slate-400 (muted), text-cyan-400 (accent)
  border: border-slate-700
  button: bg-gradient-to-r from-cyan-500 to-blue-600

STATE MACHINE for processingStatus:
  'idle' → 'uploading' (on file select) → 'ready' (file parsed) → 
  'processing' (on Restore click, 2 second fake delay) → 'complete' (show results)

IMPORTANT: Use only React useState and useRef. No external state libraries.
Use useRef for canvas elements and draw to them after state updates in useEffect.
No localStorage. All data in memory only.

The entire app must be in a SINGLE .jsx file for artifact rendering.
Include the full implementation — no placeholders or "// TODO" comments.
```

---

## 📋 FINAL SUBMISSION CHECKLIST PROMPTS

---

### PROMPT C-7 — Update README with Actual Trained Metrics

```
After training completes, update kla-image-restoration/README.md and the root README.md
to replace all placeholder XX.XX metric values with actual results from:
  results/metrics/results_summary.json

Specifically update:
1. The "Target Performance" table — add a "Achieved" column with actual values.
2. Add a "Results" section with the actual PSNR / SSIM / LPIPS from results_summary.json.
3. Add the runtime from results/benchmarks/benchmark_results.json.
4. Add a "Comparison" table: Baseline CNN vs RestoreNet.
5. Update the inference command to use the correct default model_path.

Also verify that this command in the README works exactly as written:
python inference.py --input_dir ./data/NoisyLR --output_dir ./results --model_path ./checkpoints/best_model.pt

Ensure GitHub repo name matches in the clone command.
```

---

### PROMPT C-8 — Final `create_submission.py` and Package Validation

```
The file scripts/create_submission.py exists but may reference paths that have changed now that
src/data/ exists. Update it to:

1. Verify all required files exist:
   - inference.py (root)
   - kla-image-restoration/inference.py
   - checkpoints/best_model.pt  ← CRITICAL
   - solution_presentation.pptx
   - requirements.txt
   - README.md
   - src/data/dataset.py
   - src/data/augmentation.py
   - configs/train.yaml
   - results/metrics/results_summary.json

2. Run a quick import check: subprocess.run(['python', '-c', 'from src.data.dataset import RestorationDataset; from src.models.restorenet import RestoreNet; print("OK")'])

3. Run inference smoke test on 3 dummy images (auto-generated, no manual setup).

4. Create the archive: kla_submission_YYYYMMDD_HHMM.tar.gz containing:
   - inference.py
   - kla-image-restoration/ (excluding data/, checkpoints/checkpoint_epoch_*.pt, logs/, __pycache__)
   - checkpoints/best_model.pt
   - solution_presentation.pptx
   - README.md
   - requirements.txt

5. Print archive size. Warn if > 200MB.

6. Print final summary:
   ================================================================
   SUBMISSION PACKAGE READY
   ================================================================
   Archive: kla_submission_YYYYMMDD_HHMM.tar.gz
   Size: XX.X MB
   PSNR: XX.XX dB | SSIM: 0.XXX | LPIPS: 0.XXX
   Runtime: XX.X ms/image
   ================================================================
   Next: Upload to KLA hackathon portal before deadline.

Run: python scripts/create_submission.py
```

---

## ⚡ PRIORITY ORDER (Do These First)

```
Day 1 (Today):
  1. Run Prompt C-1 → Creates src/data/ (unblocks everything)
  2. Run Prompt C-2 → Fixes validation.py
  3. Run Prompt C-3 → Completes scripts/train.py main()
  4. Extract dataset zips from dataset/ folder
  5. Run Prompt C-4 → START TRAINING (let it run overnight)

Day 2 (Training should be done):
  6. Run Prompt C-5 → Fix dry_run.py
  7. Run Prompt C-7 → Update README with real metrics
  8. Run Prompt D-2 → Build the Antigravity frontend demo
  9. Run Prompt C-6 → Generate solution_presentation.pptx with real metrics

Day 3 (Final):
  10. Run Prompt C-8 → Package validation and submission archive
  11. Upload to KLA portal

Stitch Design (Prompt D-1) → Run FIRST on Stitch tool before Day 2 so you have the 
visual reference ready when you build the Antigravity frontend.
```

---

*Total prompts: 8 completion + 2 frontend + 1 checklist = 11 prompts to full production readiness*
*Root cause of current failures: src/data/ directory entirely missing → all scripts fail on import*
