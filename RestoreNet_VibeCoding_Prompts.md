# RestoreNet — Vibe-Coding Prompt Sequence
## KLA SEMICON India Hackathon 2026 · AI-Based Image Restoration

> **How to use this file**
> Copy each numbered prompt exactly into your AI coding assistant (Cursor, Windsurf, Claude Code, etc.).
> Complete each prompt fully and verify its success criteria before moving to the next.
> Prompts are ordered to match the critical path: Setup → Data → Baseline → Full Model → Inference → Optimization → Tests → Presentation.

---

## PHASE 0 — PROJECT SCAFFOLD

### Prompt 0.1 — Initialize Repository & Environment

```
Create a Python project called `kla-image-restoration` with the following exact directory structure:

kla-image-restoration/
├── README.md
├── .gitignore
├── requirements.txt
├── pyproject.toml
├── configs/
│   ├── base.yaml
│   ├── train.yaml
│   ├── baseline.yaml
│   ├── inference.yaml
│   └── benchmark.yaml
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   ├── augmentation.py
│   │   ├── split.py
│   │   └── loader.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── baseline.py
│   │   ├── restorenet.py
│   │   └── blocks.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   ├── losses.py
│   │   ├── metrics.py
│   │   └── validation.py
│   ├── inference/
│   │   ├── __init__.py
│   │   └── engine.py
│   ├── optimization/
│   │   ├── __init__.py
│   │   ├── inference_engine.py
│   │   └── torchscript_export.py
│   └── utils/
│       ├── __init__.py
│       ├── logging.py
│       ├── seed.py
│       └── visualization.py
├── scripts/
│   ├── inspect_dataset.py
│   ├── train.py
│   ├── train_baseline.py
│   ├── evaluate.py
│   ├── benchmark.py
│   └── generate_synthetic_pairs.py
├── tests/
│   ├── __init__.py
│   ├── test_dataset.py
│   ├── test_model.py
│   ├── test_losses.py
│   └── test_metrics.py
├── data/
│   ├── GT/
│   ├── NoisyLR/
│   └── NoisyLR_synth/
├── checkpoints/
├── results/
│   ├── metrics/
│   ├── visualizations/
│   └── benchmarks/
└── logs/

Populate requirements.txt with:
torch==2.1.0
torchvision==0.16.0
numpy>=1.24
scipy>=1.10
scikit-image>=0.21
lpips>=0.1.4
pyyaml>=6.0
tqdm>=4.65
tensorboard>=2.14
matplotlib>=3.7
Pillow>=10.0

Populate .gitignore to exclude data/, checkpoints/, logs/, __pycache__, *.pyc, .env, venv/, *.egg-info.

In README.md write a placeholder structure with sections: Overview, Setup, Dataset, Training, Inference, Evaluation, Results, License.

In src/utils/seed.py write a set_seed(seed=42) function that sets seeds for random, numpy, torch, and torch.cuda, and also sets torch.backends.cudnn.deterministic = True.
```

**Success check:** Directory tree matches exactly. `python -c "from src.utils.seed import set_seed; set_seed(42)"` runs without errors.

---

### Prompt 0.2 — Base & Training YAML Configs

```
Populate the YAML config files as follows.

configs/base.yaml:
- seed: 42
- device: cuda
- data_root: ./data
- output_dir: ./results
- checkpoint_dir: ./checkpoints
- log_dir: ./logs

configs/train.yaml (inherits base, adds):
model:
  name: restorenet
  scale_factor: 2
  num_features: 64
  num_blocks: 10
training:
  epochs: 100
  batch_size: 8
  learning_rate: 0.001
  optimizer: Adam
  betas: [0.9, 0.999]
  weight_decay: 0
  warmup_epochs: 5
  gradient_clip: 1.0
  scheduler:
    name: CosineAnnealingLR
    T_max: 100
    eta_min: 0.000001
loss:
  lambda_pixel: 1.0
  lambda_ssim: 0.3
  lambda_lpips: 0.1
mixed_precision:
  enabled: true
data:
  train_ratio: 0.70
  val_ratio: 0.20
  include_synthetic: true
  synthetic_samples_per_image: 2
checkpointing:
  save_every_n_epochs: 5
  keep_last_n: 3
  best_metric: val_psnr
  patience: 20
logging:
  tensorboard: true
  log_every_n_batches: 50

configs/baseline.yaml:
model:
  name: baseline
  scale_factor: 2
  num_features: 64
  num_blocks: 3
training:
  epochs: 50
  batch_size: 8
  learning_rate: 0.001
  loss: L1
  scheduler:
    name: CosineAnnealingLR
    T_max: 50

configs/inference.yaml:
model_path: ./checkpoints/best_model.pt
device: cuda
batch_size: 1
use_compile: false
save_dtype: float32

configs/benchmark.yaml:
warmup_runs: 10
benchmark_runs: 100
batch_sizes: [1, 4, 8]
use_compile: true

Also write a small helper in src/utils/logging.py that loads a YAML file and returns a dict, with a get_config(path) function.
```

**Success check:** `python -c "from src.utils.logging import get_config; c = get_config('configs/train.yaml'); print(c)"` prints the dict.

---

## PHASE 1 — DATA PIPELINE

### Prompt 1.1 — Dataset Inspection Script

```
Write scripts/inspect_dataset.py as a standalone CLI script that performs the following analysis on the GT and NoisyLR directories:

Arguments: --gt_dir, --noisylr_dir, --sample_size (default 50)

Analysis steps:
1. List all .npy files in each directory, count them, check they are equal in number.
2. For a random sample of `sample_size` pairs (matched by filename stem):
   a. Load each file as float32.
   b. Record shape, dtype, min, max, mean, std.
3. Print a structured report with sections:
   - GT IMAGES: count, unique shapes, value range, all-in-[0,1] flag.
   - NOISYLR IMAGES: count, unique shapes, value range, has-negative-values flag, has-values-above-1 flag.
   - PAIR VALIDATION: number of matched pairs vs total files.
4. Exit with code 1 if no pairs are found; exit 0 on success.

Key constraint: Do NOT clip any values during loading. Preserve the out-of-range nature of NoisyLR.

Usage: python scripts/inspect_dataset.py --gt_dir data/GT --noisylr_dir data/NoisyLR
```

**Success check:** Script runs with synthetic dummy data and prints a clean report.

---

### Prompt 1.2 — Dataset Loader (`src/data/dataset.py`)

```
Write the RestorationDataset class in src/data/dataset.py.

Requirements:
- Inherits from torch.utils.data.Dataset.
- __init__ takes: gt_dir (str), noisylr_dir (str), normalize (bool=False), augment (bool=False).
- Pair matching: for each .npy file in gt_dir, look for a file with the same name stem in noisylr_dir. Store matched pairs in self.pairs list.
- Raise ValueError if no pairs found.
- __len__ returns number of pairs.
- __getitem__(idx):
  1. Load gt and noisylr as float32 numpy arrays using np.load.
  2. Do NOT clip any values.
  3. If normalize=True, apply per-image normalization to noisylr: (x - mean) / (std + 1e-6). Do NOT normalize gt.
  4. If augment=True, call self._augment(gt, noisylr).
  5. Add channel dimension: both become shape [1, H, W].
  6. Convert to torch.Tensor (float32).
  7. Return (noisylr_tensor, gt_tensor).
- _augment method: random horizontal flip (50%), random vertical flip (50%), random 90° rotation (k in 0–3). Apply the same transform to both gt and noisylr arrays.
- Include an assert that checks gt.shape == noisylr.shape OR gt is exactly 2× the size of noisylr in spatial dims.
- Print "Loaded N GT/NoisyLR pairs" at init.
```

**Success check:** `from src.data.dataset import RestorationDataset` imports cleanly and can be instantiated with dummy directories containing at least one matching .npy pair.

---

### Prompt 1.3 — Train/Val/Holdout Split (`src/data/split.py`)

```
Write src/data/split.py with one function:

create_train_val_split(dataset, train_ratio=0.70, val_ratio=0.20, seed=42)

- Sets numpy and torch random seeds from the seed argument.
- Computes n = len(dataset).
- Creates a random permutation of indices.
- Splits into train (first 70%), val (next 20%), holdout (remaining 10%).
- Returns three torch.utils.data.Subset objects: train_dataset, val_dataset, holdout_dataset.
- Prints a summary: "Split: N_train train | N_val val | N_holdout holdout"

Also write a get_dataloaders(train_ds, val_ds, batch_size=8, num_workers=4) function that returns (train_loader, val_loader) as DataLoader objects with:
- train: shuffle=True, pin_memory=True, prefetch_factor=2
- val: shuffle=False, pin_memory=True
```

**Success check:** Function runs without error on a dummy dataset of 100 items and returns subsets of sizes 70, 20, 10.

---

### Prompt 1.4 — Synthetic Degradation Augmentor (`src/data/augmentation.py`)

```
Write the SyntheticDegradationAugmentor class in src/data/augmentation.py.

__init__(self, noise_std_range=(0.01, 0.05), speckle_range=(0.5, 1.5), downsample_factors=(2, 3, 4))

Methods:
- apply_speckle(img: np.ndarray, strength: float) -> np.ndarray
  Multiplicative speckle: img * uniform(1/strength, strength) element-wise. Return result WITHOUT clipping (preserve out-of-range).
  
- apply_gaussian(img: np.ndarray, std: float) -> np.ndarray
  Additive Gaussian: img + np.random.normal(0, std, img.shape). Do NOT clip.

- apply_downsample(img: np.ndarray, factor: int) -> np.ndarray
  Use scipy.ndimage.zoom with 1/factor scale and order=1 (bilinear).

- generate_synthetic_pair(gt: np.ndarray) -> np.ndarray
  Applies all three degradations in a RANDOM order (np.random.permutation([0,1,2])).
  Returns the degraded image.

- augment_dataset(gt_dir: str, output_dir: str, samples_per_image: int = 2)
  For each .npy in gt_dir, generate `samples_per_image` synthetic NoisyLR images.
  Save each as {stem}_synth_{i}.npy in output_dir (create if not exists).
  Print "Generated N synthetic degraded pairs" at end.
  Return total count.

Also write a corresponding SyntheticRestorationDataset class that accepts a gt_dir and generates synthetic pairs on-the-fly in __getitem__ (instead of reading from disk NoisyLR). This allows dynamic augmentation during training.
```

**Success check:** Running `augment_dataset` on a directory with 5 dummy GT images produces 10 synthetic NoisyLR files.

---

### Prompt 1.5 — Verify Full Data Pipeline

```
Write a short self-contained test script at tests/test_dataset.py that:

1. Creates 10 dummy GT .npy files (128×128, float32, uniform [0,1]) and 10 matching NoisyLR .npy files (128×128, float32, uniform [-0.1, 1.6]) in a temp directory.
2. Instantiates RestorationDataset with normalize=False, augment=True.
3. Asserts len(dataset) == 10.
4. Fetches item 0; asserts noisylr.shape == (1, 128, 128) and gt.shape == (1, 128, 128).
5. Asserts noisylr.dtype == torch.float32.
6. Asserts that clipping has NOT been applied: noisylr.min() can be < 0 or noisylr.max() can be > 1.
7. Calls create_train_val_split and asserts sizes sum to 10.
8. Instantiates SyntheticDegradationAugmentor and calls generate_synthetic_pair on a dummy image; asserts output shape matches input.
9. Prints "ALL DATA PIPELINE TESTS PASSED" on success.

Run using: python -m pytest tests/test_dataset.py -v
```

**Success check:** All 9 assertions pass.

---

## PHASE 2 — BASELINE MODEL

### Prompt 2.1 — Reusable Blocks (`src/models/blocks.py`)

```
Write src/models/blocks.py with these classes:

1. ResidualBlock(nn.Module)
   - __init__(self, channels, kernel_size=3, padding=1)
   - Two Conv2d layers with ReLU between them.
   - forward: return x + conv2(relu(conv1(x)))

2. ChannelAttention(nn.Module) — Squeeze-and-Excitation
   - __init__(self, channels, reduction=16)
   - fc1: Conv2d(channels, channels//reduction, kernel_size=1)
   - fc2: Conv2d(channels//reduction, channels, kernel_size=1)
   - forward:
     1. Global average pool to [B, C, 1, 1]
     2. fc1 → ReLU → fc2 → sigmoid
     3. Return x * attention_weights

3. UpsampleBlock(nn.Module)
   - __init__(self, scale_factor=2, mode='bilinear')
   - forward: F.interpolate(x, scale_factor=scale_factor, mode=mode, align_corners=False)

4. PixelShuffleBlock(nn.Module) — alternative learned upsampling
   - __init__(self, in_channels, scale_factor=2)
   - Conv2d(in_channels, in_channels * scale_factor**2, 3, padding=1) followed by nn.PixelShuffle(scale_factor)
   - forward: applies conv then pixel shuffle

Each class should have a brief docstring and a shape comment showing input/output shapes.
```

**Success check:** `from src.models.blocks import ResidualBlock, ChannelAttention, UpsampleBlock` imports cleanly. Quick shape test: `ResidualBlock(64)(torch.randn(2, 64, 32, 32)).shape == (2, 64, 32, 32)`.

---

### Prompt 2.2 — Baseline CNN (`src/models/baseline.py`)

```
Write the BaselineRestorationCNN class in src/models/baseline.py.

Architecture (in order):
1. Bilinear upsample by scale_factor (nn.Upsample, align_corners=False).
2. Conv2d(1, num_features, 3, padding=1) — initial feature extraction.
3. N residual blocks (use ResidualBlock from blocks.py).
4. Conv2d(num_features, 1, 3, padding=1) — output conv.
5. Add a global residual connection: final output = conv_out(features) + upsampled_input.

__init__(self, scale_factor=2, num_features=64, num_blocks=3)

forward(self, x):
  upsampled = self.upsample(x)
  feat = self.conv_in(upsampled)
  for block in self.res_blocks:
      feat = feat + block(feat)
  out = self.conv_out(feat) + upsampled
  return out

Add a count_parameters(model) helper function at module level.
Add a __main__ block that instantiates the model, prints parameter count, and runs a forward pass with a dummy tensor [1, 1, 128, 128] to verify shapes.

Expected: ~0.4M parameters, output shape [1, 1, 256, 256] for scale_factor=2.
```

**Success check:** `python src/models/baseline.py` prints parameter count and "Output shape: (1, 1, 256, 256)" without errors.

---

### Prompt 2.3 — Baseline Training Script (`scripts/train_baseline.py`)

```
Write scripts/train_baseline.py — a standalone training script for the baseline model.

It should:
1. Accept CLI arguments: --gt_dir, --noisylr_dir, --config (default configs/baseline.yaml), --output_dir (default checkpoints/).
2. Load config from YAML.
3. Call set_seed(42).
4. Instantiate RestorationDataset with augment=True.
5. Create train/val split (70/20/10).
6. Instantiate BaselineRestorationCNN from config.
7. Use nn.L1Loss() as criterion.
8. Use Adam optimizer with lr from config.
9. Use CosineAnnealingLR scheduler.
10. Training loop (for N epochs from config):
    a. Train one epoch: forward pass, loss, backward, clip_grad_norm_ (max_norm=1.0), optimizer step.
    b. Validate: compute average PSNR (using skimage.metrics.peak_signal_noise_ratio with data_range=1.0 on clipped [0,1] outputs).
    c. Save checkpoint if val_psnr is the best seen so far as `{output_dir}/baseline_best.pt`.
    d. Print: "Epoch N/total | Train Loss: X.XXXX | Val PSNR: XX.XX dB"
11. Save final checkpoint as `{output_dir}/baseline_final.pt`.
12. Print total training time.

The checkpoint dict should contain: {'model_state': state_dict, 'epoch': epoch, 'val_psnr': best_psnr, 'config': config}
```

**Success check:** Script runs for 2 epochs on dummy data without errors and saves a checkpoint file.

---

## PHASE 3 — MAIN MODEL (RestoreNet)

### Prompt 3.1 — RestoreNet Architecture (`src/models/restorenet.py`)

```
Write the full RestoreNet class in src/models/restorenet.py.

Use ResidualBlock and ChannelAttention from src/models/blocks.py.

Architecture:
- __init__(self, scale_factor=2, num_features=64, num_blocks=10)
- self.upsample: nn.Upsample(scale_factor, mode='bilinear', align_corners=False)
- self.conv_in: Conv2d(1, num_features, 3, padding=1)
- self.res_blocks: ModuleList of `num_blocks` ResidualBlock(num_features) instances
- self.attention_blocks: ModuleList of (num_blocks // 5) ChannelAttention(num_features) instances
- self.conv_mid: Conv2d(num_features, num_features, 3, padding=1)
- self.conv_out: Conv2d(num_features, 1, 3, padding=1)

forward(self, x):
  upsampled = self.upsample(x)
  feat = self.conv_in(upsampled)
  attn_idx = 0
  for i, block in enumerate(self.res_blocks):
      feat = block(feat)
      if (i+1) % 5 == 0 and attn_idx < len(self.attention_blocks):
          feat = self.attention_blocks[attn_idx](feat)
          attn_idx += 1
  feat = self.conv_mid(feat)
  residual = self.conv_out(feat)
  return upsampled + residual

Add a __main__ block that:
- Instantiates RestoreNet(scale_factor=2, num_features=64, num_blocks=10)
- Prints parameter count (expected ~1.6M)
- Runs forward pass on [1, 1, 128, 128] dummy input
- Prints output shape (expected [1, 1, 256, 256])
- Confirms no NaN in output
```

**Success check:** `python src/models/restorenet.py` runs cleanly and reports ~1.6M parameters.

---

### Prompt 3.2 — Metrics (`src/training/metrics.py`)

```
Write src/training/metrics.py with the following functions, all operating on numpy arrays in [0,1]:

1. compute_psnr(pred: np.ndarray, target: np.ndarray) -> float
   Uses skimage.metrics.peak_signal_noise_ratio(target, pred, data_range=1.0).
   Both inputs are clipped to [0,1] before computation.

2. compute_ssim(pred: np.ndarray, target: np.ndarray) -> float
   Uses skimage.metrics.structural_similarity(target, pred, data_range=1.0).
   Both inputs clipped to [0,1].

3. compute_lpips(pred: torch.Tensor, target: torch.Tensor, device='cuda') -> float
   Uses the lpips library. Inputs are [B,1,H,W] tensors in [0,1].
   Converts grayscale to 3-channel by repeating. Returns mean LPIPS as a float.

4. compute_all_metrics(pred_batch, gt_batch, device='cuda') -> dict
   Accepts torch tensors [B,1,H,W]. Returns dict with keys:
   {'psnr': float, 'ssim': float, 'lpips': float}
   Computes average across the batch.

5. MetricsTracker class:
   - reset(): clears accumulated values
   - update(psnr, ssim, lpips): appends to internal lists
   - summary() -> dict: returns mean of each metric across all updates
   - log_string() -> str: returns "PSNR: XX.XX | SSIM: 0.XXX | LPIPS: 0.XXX"
```

**Success check:** All functions importable. Unit test with random arrays: compute_psnr of a tensor against itself returns ~100 dB (or infinity); compute_ssim returns 1.0.

---

### Prompt 3.3 — Loss Functions (`src/training/losses.py`)

```
Write src/training/losses.py with a multi-term RestorationLoss class.

class RestorationLoss(nn.Module):
  __init__(self, lambda_pixel=1.0, lambda_ssim=0.3, lambda_lpips=0.1, device='cuda')
  
  Internal components:
  - self.l1 = nn.L1Loss()
  - self.ssim_fn = SSIM() (implement inline below)
  - self.lpips_fn: use the lpips library's LPIPS(net='alex') in eval mode with frozen params

  forward(self, pred, target):
    pred_clipped = torch.clamp(pred, 0, 1)
    l1_loss = self.l1(pred_clipped, target)
    ssim_loss = 1 - self.ssim_fn(pred_clipped, target)
    with torch.no_grad():
        lpips_loss = self.lpips_fn(
            pred_clipped.repeat(1,3,1,1) * 2 - 1,
            target.repeat(1,3,1,1) * 2 - 1
        ).mean()
    total = lambda_pixel*l1_loss + lambda_ssim*ssim_loss + lambda_lpips*lpips_loss
    return total, {'l1': l1_loss.item(), 'ssim_loss': ssim_loss.item(), 'lpips': lpips_loss.item()}

Also implement class SSIM(nn.Module) in the same file:
  - Window-based SSIM using an 11×11 Gaussian window (sigma=1.5).
  - Constants C1=0.01**2, C2=0.03**2.
  - forward(x, y) returns scalar SSIM value (average over spatial map and batch).
  - Use F.conv2d for local statistics.

Add a CharbonnierLoss(nn.Module):
  - forward(pred, target, eps=0.01): returns mean of sqrt((pred-target)**2 + eps**2)
  - This is an L1-smooth approximation useful as a drop-in replacement for L1.
```

**Success check:** `RestorationLoss()(torch.rand(2,1,64,64), torch.rand(2,1,64,64))` returns a scalar tensor without error.

---

### Prompt 3.4 — Full Training Loop (`src/training/trainer.py`)

```
Write src/training/trainer.py implementing a Trainer class.

class Trainer:
  __init__(self, model, train_loader, val_loader, config, device='cuda'):
    - Stores all references.
    - Instantiates RestorationLoss from config loss lambdas.
    - Instantiates Adam optimizer (lr, betas, weight_decay from config).
    - Instantiates CosineAnnealingLR scheduler (T_max, eta_min from config).
    - Sets up AMP GradScaler if config.mixed_precision.enabled is True.
    - Initializes MetricsTracker.
    - Creates checkpoint_dir and log_dir if they don't exist.
    - Initializes TensorBoard SummaryWriter if logging.tensorboard is True.

  train_epoch(self, epoch) -> float:
    - model.train()
    - For each batch (noisylr, gt): forward, loss, backward with gradient clipping (max_norm=1.0), optimizer step.
    - With AMP: use autocast and GradScaler.
    - Every log_every_n_batches: print batch loss.
    - Return average epoch loss.

  val_epoch(self, epoch) -> dict:
    - model.eval() with torch.no_grad()
    - For each val batch: forward pass, compute PSNR and SSIM per image.
    - Return {'val_psnr': mean_psnr, 'val_ssim': mean_ssim}

  save_checkpoint(self, epoch, metrics, is_best=False):
    - Saves dict: {model_state, optimizer_state, epoch, metrics, config}
    - If is_best: saves as best_model.pt
    - Also saves as checkpoint_epoch_{N}.pt
    - Keeps only the last keep_last_n checkpoints (delete older ones).

  load_checkpoint(self, path) -> int: loads and returns epoch number.

  fit(self):
    - Main loop: for epoch in range(epochs):
        train_loss = self.train_epoch(epoch)
        val_metrics = self.val_epoch(epoch)
        scheduler.step()
        if val_psnr > best_psnr: save best checkpoint
        if epoch % save_every_n_epochs == 0: save periodic checkpoint
        if no improvement for `patience` epochs: early stop
    - Print final summary with best epoch and best val_psnr.
```

**Success check:** Trainer can be instantiated with a tiny model and dummy loaders and runs 2 epochs without error.

---

### Prompt 3.5 — Main Training Script (`scripts/train.py`)

```
Write scripts/train.py as the production training entry point.

CLI args:
  --gt_dir (required)
  --noisylr_dir (required)
  --config (default: configs/train.yaml)
  --resume (optional: path to checkpoint to resume from)
  --device (default: cuda)

Logic:
1. Load config from YAML.
2. set_seed(config.seed).
3. Print full config for reproducibility.
4. Build RestorationDataset (augment=True for training).
5. If config.data.include_synthetic is True:
   - Generate synthetic NoisyLR using SyntheticDegradationAugmentor into data/NoisyLR_synth/.
   - Create a second RestorationDataset for synthetic pairs.
   - Use ConcatDataset to combine official and synthetic datasets.
6. create_train_val_split → get_dataloaders.
7. Instantiate RestoreNet from config.
8. Instantiate Trainer.
9. If --resume is provided, call trainer.load_checkpoint(resume).
10. Call trainer.fit().
11. Print "Training complete. Best model saved at checkpoints/best_model.pt"

At the top, print GPU info: torch.cuda.get_device_name() and total VRAM.
```

**Success check:** `python scripts/train.py --gt_dir data/GT --noisylr_dir data/NoisyLR` begins training and writes a checkpoint after epoch 1.

---

## PHASE 4 — INFERENCE PIPELINE

### Prompt 4.1 — Standalone Inference Script (`inference.py`)

```
Write inference.py at the repo root — this is the CRITICAL KLA submission file.

CLI args (all configurable, NO hardcoded paths):
  --input_dir (required): directory of .npy NoisyLR files
  --output_dir (required): directory to save restored .npy files
  --model_path (default: checkpoints/best_model.pt)
  --device (default: cuda, choices: [cuda, cpu])
  --batch_size (default: 1)
  --use_compile (flag, bool): if set and torch.compile exists, compile the model
  --save_dtype (default: float32, choices: [float32, uint8])
  --verbose (flag)

Logic:
1. Validate --input_dir exists; create --output_dir if not exists.
2. Glob all *.npy files from input_dir; sort them.
3. Exit with error if zero files found.
4. Load RestoreNet model from --model_path.
   - If checkpoint is a dict with 'model_state', load that key.
   - If model file not found, print a warning and use random weights (for testing).
5. If --use_compile and PyTorch >= 2.0: apply torch.compile(model, mode='reduce-overhead').
6. For each file:
   a. np.load → float32 (DO NOT clip on load)
   b. Add batch and channel dims: [1, 1, H, W] tensor
   c. .to(device)
   d. torch.inference_mode() forward pass
   e. .cpu().squeeze().numpy()
   f. np.clip(output, 0, 1).astype(float32) — clip ONLY at save time
   g. np.save(output_dir / filename, output)
   h. Append per-image timing.
7. After all images: print summary table:
   - Total images, total time, avg ms/image, std ms/image, throughput (images/sec).
8. Exit code 0 on success.

IMPORTANT: No code changes should be needed by the evaluator. All paths via CLI args.
```

**Success check:** `python inference.py --input_dir data/NoisyLR --output_dir results/inference_outputs` runs end-to-end and saves .npy files matching input filenames.

---

### Prompt 4.2 — Optimized Inference Engine (`src/optimization/inference_engine.py`)

```
Write src/optimization/inference_engine.py with an OptimizedInferenceEngine class.

class OptimizedInferenceEngine:
  __init__(self, model, device='cuda', use_compile=False, batch_size=4):
    - Puts model in eval mode.
    - If use_compile and torch version >= 2.0: torch.compile(model, mode='reduce-overhead').
    - Stores batch_size.
    - Warmup: run 5 forward passes on dummy input [1,1,256,256] to trigger CUDA caching.

  preprocess(self, npy_path: str) -> torch.Tensor:
    - Loads .npy as float32 (no clipping).
    - Returns [1, 1, H, W] tensor on CPU.

  infer_single(self, tensor: torch.Tensor) -> torch.Tensor:
    - Moves tensor to device.
    - torch.inference_mode() forward.
    - Returns output on CPU.

  postprocess(self, tensor: torch.Tensor) -> np.ndarray:
    - squeeze → numpy → clip [0,1] → float32.

  process_directory(self, input_dir: str, output_dir: str) -> dict:
    - Processes all .npy files. Records timing breakdown:
      {disk_io, inference, postprocess, total} as lists.
    - Prints benchmark table after completion.
    - Returns timings dict.

  benchmark(self, n_images=50, image_size=(256,256)) -> dict:
    - Benchmarks with dummy random images of given size.
    - Returns {mean_ms, std_ms, throughput} for warmup-excluded runs.
```

**Success check:** Engine instantiates, runs benchmark on CPU with dummy data, prints timing table.

---

## PHASE 5 — EVALUATION

### Prompt 5.1 — Evaluation Script (`scripts/evaluate.py`)

```
Write scripts/evaluate.py — computes metrics against ground truth.

CLI args:
  --gt_dir (required)
  --pred_dir (required): directory of restored .npy predictions
  --output_json (default: results/metrics/results_summary.json)
  --verbose (flag)

Logic:
1. Match files by stem between gt_dir and pred_dir.
2. For each matched pair:
   a. Load gt (clip to [0,1]) and pred (clip to [0,1]) as float32.
   b. Compute PSNR, SSIM via skimage.
3. Compute LPIPS once for all matched pairs in batches of 16.
4. Print per-image results if --verbose.
5. Print summary table:
   ┌─────────────────────────────────────┐
   │         EVALUATION RESULTS         │
   ├──────────┬──────────────────────────┤
   │  Metric  │   Mean    ±  Std         │
   ├──────────┼──────────────────────────┤
   │  PSNR    │  XX.XX dB ± X.XX         │
   │  SSIM    │  0.XXX    ± 0.XXX        │
   │  LPIPS   │  0.XXX    ± 0.XXX        │
   └──────────┴──────────────────────────┘
6. Save results JSON with: {per_image: [...], summary: {psnr_mean, ssim_mean, lpips_mean, ...}}.
7. Also include BASELINE comparison if baseline outputs exist at results/baseline_outputs/.
```

**Success check:** Script runs on two directories of dummy .npy files and produces a valid JSON.

---

### Prompt 5.2 — Visualization Script (`src/utils/visualization.py`)

```
Write src/utils/visualization.py with these functions:

1. visualize_restoration(noisylr, pred, gt, title='', save_path=None)
   - Creates a 3-panel matplotlib figure: NoisyLR | Predicted | GT
   - Below each panel: min/max range and PSNR vs GT
   - Saves to save_path if provided; otherwise plt.show()

2. create_comparison_grid(noisylr_dir, pred_dir, gt_dir, output_dir, num_samples=8)
   - Randomly samples `num_samples` triplets.
   - For each: loads .npy files, calls visualize_restoration, saves to output_dir.
   - Also saves a combined grid image showing all pairs.

3. plot_training_curves(log_dir, output_path)
   - Reads TensorBoard event files from log_dir using tensorboard.
   - Plots train loss, val PSNR, val SSIM over epochs.
   - Saves figure to output_path.

4. plot_metric_histogram(metrics_json_path, output_path)
   - Loads results JSON, plots histogram of per-image PSNR distribution.
   - Annotates mean and std.

All functions should gracefully handle missing files with a print warning.
```

**Success check:** `visualize_restoration` creates a 3-panel figure with dummy numpy arrays and saves it as a .png.

---

## PHASE 6 — RUNTIME OPTIMIZATION

### Prompt 6.1 — Benchmarking Script (`scripts/benchmark.py`)

```
Write scripts/benchmark.py to benchmark inference speed on the evaluation hardware.

CLI args:
  --model_path (default: checkpoints/best_model.pt)
  --device (default: cuda)
  --image_size (default: 256, the square image size)
  --warmup_runs (default: 10)
  --benchmark_runs (default: 100)
  --output_json (default: results/benchmarks/benchmark_results.json)

Benchmark modes (run all automatically):
1. Eager mode: standard forward pass
2. torch.compile() mode (if PyTorch >= 2.0): mode='reduce-overhead'
3. torch.inference_mode() vs torch.no_grad()

For each mode:
  - Run warmup_runs silent forward passes.
  - Time benchmark_runs passes using time.perf_counter().
  - Record: mean_ms, std_ms, min_ms, max_ms, p95_ms (95th percentile), throughput (images/sec).

Print a comparison table:
  Mode           | Mean (ms) | Std   | P95   | Throughput
  Eager          |   XX.X    | X.X   | XX.X  | XX.X img/s
  torch.compile  |   XX.X    | X.X   | XX.X  | XX.X img/s

Save all results to JSON.
Print: "✓ Target: <100ms end-to-end. Current best: XX.X ms"
```

**Success check:** Script runs on CPU with a dummy model and produces a results JSON.

---

### Prompt 6.2 — torch.compile Export (`src/optimization/torchscript_export.py`)

```
Write src/optimization/torchscript_export.py with:

1. compile_model(model, mode='reduce-overhead') -> compiled_model
   - Checks torch version >= 2.0; raises RuntimeError if not.
   - Applies torch.compile(model, mode=mode).
   - Runs one warmup forward pass on [1,1,256,256] dummy input.
   - Returns compiled model.

2. export_torchscript(model, save_path, example_input_shape=(1,1,256,256))
   - Puts model in eval mode.
   - Runs torch.jit.trace on the example input.
   - Saves the traced script to save_path.
   - Verifies the saved script can be loaded and produces same output (within 1e-4 tolerance).
   - Prints file size of saved model.

3. load_torchscript(path) -> nn.Module
   - Loads and returns a TorchScript model.

4. compare_outputs(original_model, compiled_model, n_tests=10)
   - Runs n_tests random inputs through both models.
   - Asserts max absolute difference < 1e-4.
   - Prints "✓ Compiled model outputs match original within tolerance."
```

**Success check:** `export_torchscript` runs on the baseline model and saves a valid `.pt` file.

---

## PHASE 7 — UNIT TESTS

### Prompt 7.1 — Model Unit Tests (`tests/test_model.py`)

```
Write tests/test_model.py with pytest tests covering:

TestBaselineCNN:
  - test_forward_shape: input [2,1,128,128] → output [2,1,256,256] for scale_factor=2
  - test_no_nan_output: random input produces no NaN or Inf in output
  - test_parameter_count: parameter count is between 100k and 2M
  - test_output_not_equal_input: model output differs from bilinear-upsampled input

TestRestoreNet:
  - test_forward_shape: same shape test as above
  - test_no_nan_output
  - test_parameter_count: ~1.5M-2M
  - test_residual_learning: with zero input, output should be close to zero (since residual + upsampled)
  - test_scale_factor_4: model with scale_factor=4 maps [1,1,64,64] → [1,1,256,256]

TestBlocks:
  - test_residual_block_identity_init: with zero weights, residual block output should equal input
  - test_channel_attention_shape: output shape matches input shape
  - test_upsample_block: [1,1,64,64] → [1,1,128,128] for scale_factor=2

Run with: python -m pytest tests/test_model.py -v
```

**Success check:** All tests pass.

---

### Prompt 7.2 — Loss & Metrics Unit Tests (`tests/test_losses.py` + `tests/test_metrics.py`)

```
Write tests/test_losses.py:

TestSSIM:
  - test_identical_images: SSIM of tensor against itself == 1.0 (within 1e-4)
  - test_range: SSIM is in [0, 1] for random inputs
  - test_different_images: SSIM of random tensor vs zeros < 0.5

TestRestorationLoss:
  - test_loss_is_positive: loss > 0 for random pred/target
  - test_loss_zero_on_identical: loss on identical pred/target is very small (< 0.01)
  - test_returns_dict: forward returns (loss_tensor, dict) with keys l1, ssim_loss, lpips

Write tests/test_metrics.py:

TestMetrics:
  - test_psnr_identical: compute_psnr(arr, arr) returns > 50.0 dB
  - test_ssim_identical: compute_ssim(arr, arr) returns 1.0
  - test_psnr_noisy: psnr of noisy arr < psnr of clean arr
  - test_metrics_tracker: update 10 times, summary returns correct means

Run with: python -m pytest tests/ -v
```

**Success check:** All tests pass.

---

## PHASE 8 — OOD ROBUSTNESS & ABLATION

### Prompt 8.1 — OOD Evaluation

```
Write a script scripts/evaluate_ood.py that tests model generalization on out-of-distribution content.

The script:
1. Accepts --gt_dir, --model_path, --output_dir.
2. For each GT image in gt_dir, generates 3 OOD synthetic degraded versions using SyntheticDegradationAugmentor with:
   - Higher noise than training: noise_std_range=(0.05, 0.15), speckle_range=(0.3, 2.0)
   - Larger downsample factors: (3, 4, 6)
3. Runs inference on all OOD inputs.
4. Computes PSNR, SSIM, LPIPS vs the original GT.
5. Also generates 5 "pure noise" images (random uniform [0,1]) as a sanity check.
6. Prints a report comparing:
   - In-distribution (official pairs) metrics (from results_summary.json if exists)
   - OOD synthetic metrics
   - Pure noise metrics (should be worst)
7. Saves to results/metrics/ood_results.json.

Print: "OOD delta PSNR: ±X.XX dB vs in-distribution"
```

**Success check:** Script runs on dummy data and produces a JSON report.

---

### Prompt 8.2 — Ablation Study Script

```
Write scripts/ablation.py that performs a structured ablation study.

Ablation configurations to test (each is a variant of the main model):
1. baseline_l1_only: BaselineRestorationCNN + L1 loss only
2. restorenet_l1_only: RestoreNet + L1 loss only (no SSIM, no LPIPS)
3. restorenet_l1_ssim: RestoreNet + L1 + SSIM
4. restorenet_full: RestoreNet + L1 + SSIM + LPIPS (full loss)
5. restorenet_no_attention: RestoreNet with ChannelAttention removed

For each configuration:
- Train for a small number of epochs (--fast_epochs, default 10) on the training split.
- Evaluate on val split: PSNR, SSIM, LPIPS.
- Record results.

Print a markdown table at the end:
| Configuration         | PSNR   | SSIM  | LPIPS |
|-----------------------|--------|-------|-------|
| baseline_l1_only      | XX.XX  | 0.XXX | 0.XXX |
| restorenet_l1_only    | XX.XX  | 0.XXX | 0.XXX |
| ...                   | ...    | ...   | ...   |

Save table to results/metrics/ablation_results.json.

Note: This is for presentation evidence; the full training happens via train.py.
```

**Success check:** Script runs for 2 fast_epochs on all 5 configs without error.

---

## PHASE 9 — DOCUMENTATION & SUBMISSION

### Prompt 9.1 — Complete README

```
Write a complete README.md for the KLA SEMICON India Hackathon submission.

Sections (in order):
1. Header: Project name "RestoreNet", tagline, badges (Python, PyTorch, License).
2. Overview: 3 sentences on what the project does and its approach.
3. Architecture: Brief description of RestoreNet (upsampling + residual blocks + channel attention + residual learning), expected performance table (PSNR/SSIM/LPIPS/Runtime targets).
4. Repository Structure: tree of key files only (not exhaustive).
5. Setup:
   - Python version requirement (3.10+)
   - pip install command
   - Dataset download instructions (KLA Google Drive link placeholder)
   - Verify setup command
6. Dataset: explain GT and NoisyLR format, float32 .npy, out-of-range handling policy.
7. Training:
   - Inspect dataset command
   - Train baseline command
   - Train full model command
   - Resume from checkpoint command
8. Inference (emphasize this is EXACTLY how KLA will run it):
   python inference.py --input_dir ./input --output_dir ./output --model_path ./checkpoints/best_model.pt
9. Evaluation:
   - Evaluate predictions command
   - OOD evaluation command
   - Benchmark runtime command
10. Results: placeholder table with target metrics.
11. External Resources: disclose lpips library, pretrained AlexNet (torchvision), any other.
12. License: MIT.

IMPORTANT: Every command in the README must be copy-pasteable and work without any edits.
```

**Success check:** README renders correctly on GitHub preview (check markdown syntax).

---

### Prompt 9.2 — Final Dry-Run Verification Script

```
Write scripts/dry_run.py — a complete end-to-end smoke test that KLA evaluators (and you) can run before submission.

Steps it performs automatically:
1. Check all required files exist: inference.py, scripts/train.py, src/models/restorenet.py, src/data/dataset.py, configs/train.yaml, requirements.txt, README.md.
2. Check Python imports: torch, numpy, scipy, skimage, lpips, yaml, tqdm all importable.
3. Check CUDA availability (warn if not available, don't fail).
4. Create a temp directory dry_run_test/ with 10 dummy .npy NoisyLR files (128×128, float32, range [-0.05, 1.4]).
5. Run: python inference.py --input_dir dry_run_test/ --output_dir dry_run_outputs/ --device cpu
6. Verify: number of output files == number of input files.
7. Verify: all output files are .npy format and loadable.
8. Verify: all output values are in [0, 1] (clipped correctly).
9. Verify: inference did not raise any exceptions.
10. Spot-check PSNR against dummy GT: should be some finite value.
11. Clean up temp directories.
12. Print final checklist:
   ✓ All required files present
   ✓ All dependencies importable
   ✓ Inference script runs without errors
   ✓ Output count matches input count
   ✓ Output values in [0, 1]
   ✓ READY FOR SUBMISSION

Exit code 0 if all pass, 1 if any fail.
```

**Success check:** `python scripts/dry_run.py` prints all green checkmarks.

---

### Prompt 9.3 — Presentation Slide Content Generator

```
Write scripts/generate_slide_content.py that auto-generates slide content from results files.

It reads:
- results/metrics/results_summary.json (main metrics)
- results/metrics/ablation_results.json (ablation)
- results/metrics/ood_results.json (OOD)
- results/benchmarks/benchmark_results.json (runtime)

And prints a structured markdown outline for a 12-slide presentation:

Slide 1 — Title
  RestoreNet: Degradation-Aware Fidelity-First Image Restoration
  KLA SEMICON India Hackathon 2026

Slide 2 — Problem Statement
  [Summarize the task: NoisyLR → GT, three degradations in unknown order]

Slide 3 — Our Approach / Key Insight
  [Single-stage unified restoration, fidelity-first philosophy, no hallucination]

Slide 4 — Architecture Diagram
  [ASCII art of RestoreNet pipeline from the plan]

Slide 5 — Data Pipeline
  [Dataset stats from results, synthetic augmentation strategy]

Slide 6 — Loss Function Design
  [Multi-term loss equation, weight rationale table]

Slide 7 — Training Strategy
  [Staged training: L1 baseline → add SSIM → add LPIPS → optimize]

Slide 8 — Quantitative Results
  [Auto-filled from results_summary.json: PSNR, SSIM, LPIPS vs baseline]

Slide 9 — Ablation Study
  [Table from ablation_results.json showing contribution of each loss term]

Slide 10 — OOD Robustness
  [Metrics from ood_results.json, delta vs in-distribution]

Slide 11 — Runtime Performance
  [Benchmark table from benchmark_results.json, comparison of modes]

Slide 12 — Conclusion
  [Key achievements, engineering discipline points, future work]

Output the content as a markdown file at results/slide_content.md.
```

**Success check:** Script runs with placeholder JSON files and produces a coherent slide_content.md.

---

## PHASE 10 — INTEGRATION & POLISH

### Prompt 10.1 — Git Hygiene & Final Checklist

```
Do the following final cleanup tasks:

1. Ensure .gitignore contains ALL of these:
   data/
   checkpoints/
   logs/
   *.pyc
   __pycache__/
   .env
   venv/
   *.egg-info/
   dry_run_test/
   dry_run_outputs/
   results/inference_outputs/
   *.npy (in root only, not src/)

2. Pin exact versions in requirements.txt based on what's actually installed. Use `pip freeze | grep -E "torch|numpy|scipy|scikit|lpips|pyyaml|tqdm|tensorboard|matplotlib|Pillow"` to generate the pinned list.

3. In pyproject.toml set:
   - name: kla-image-restoration
   - version: 1.0.0
   - python_requires: >=3.10
   - All dependencies from requirements.txt

4. Verify inference.py has NO hardcoded paths (grep for absolute paths starting with /home or /Users).

5. Run: python -m py_compile inference.py src/**/*.py scripts/*.py
   Fix any syntax errors found.

6. Print the submission checklist and confirm each item:
   [ ] inference.py works without code edits
   [ ] checkpoints/best_model.pt exists
   [ ] README has correct run commands
   [ ] requirements.txt is complete
   [ ] All external resources disclosed in README
   [ ] GitHub repo is public
   [ ] solution_presentation.pptx exists
   [ ] No hardcoded paths
   [ ] No syntax errors
   [ ] OOD evaluation attempted
```

**Success check:** `python -m py_compile inference.py` exits with code 0.

---

### Prompt 10.2 — Final Submission Archive

```
Create a script scripts/create_submission.py that builds the submission archive.

Steps:
1. Run scripts/dry_run.py programmatically; abort if it fails.
2. Verify checkpoints/best_model.pt exists; abort if not.
3. Create a timestamped archive: kla_submission_{YYYYMMDD_HHMM}.tar.gz containing:
   - README.md
   - inference.py
   - requirements.txt
   - configs/
   - src/
   - scripts/train.py
   - scripts/evaluate.py
   - scripts/benchmark.py
   - checkpoints/best_model.pt
   - solution_presentation.pptx (if exists)
   - results/metrics/results_summary.json (if exists)
4. Print file size of archive.
5. Print: "✓ Submission archive created: kla_submission_YYYYMMDD_HHMM.tar.gz"
6. Print: "Upload to KLA hackathon portal and verify receipt email."

Also verify the archive can be extracted cleanly in /tmp and that inference.py is present and syntactically valid inside the extracted archive.
```

**Success check:** Archive is created, is under 500MB, and contains inference.py at root level.

---

## APPENDIX — DEBUGGING PROMPTS

> Use these only if something breaks during development.

### Debug Prompt A — Shape Mismatch

```
I'm getting a tensor shape mismatch error in RestoreNet. The input NoisyLR has shape [B, 1, H, W] but after the residual learning, the upsampled tensor and the residual output tensor have different spatial sizes. Fix the forward() method in src/models/restorenet.py to handle variable input sizes correctly. Ensure that the upsampled tensor and the final output of conv_out both have the same [B, 1, H_gt, W_gt] shape before the residual addition.
```

### Debug Prompt B — LPIPS Import Error

```
The lpips library is failing to import or the pretrained AlexNet weights are not downloading correctly. In src/training/losses.py, modify RestorationLoss to gracefully fall back to a pure L1 + SSIM loss if lpips cannot be imported. Add a warning print: "LPIPS unavailable, using L1+SSIM only." Ensure training still proceeds correctly without LPIPS.
```

### Debug Prompt C — DataLoader Memory Issues

```
The DataLoader is running out of memory during training. In src/data/split.py, reduce num_workers to 2 and disable pin_memory for CPU training. Also add a patch_size argument to RestorationDataset: if patch_size is set (e.g. 128), randomly crop both gt and noisylr to patch_size during __getitem__ to reduce memory footprint per batch.
```

### Debug Prompt D — Checkpoint Loading Failure

```
The inference script fails when loading checkpoints saved by the Trainer class. The error is a state_dict key mismatch. In inference.py, make the load_model function more robust: try loading with strict=True first; if that fails, try strict=False and print which keys were skipped. Also handle the case where the checkpoint was saved with DataParallel (keys start with 'module.') by stripping the prefix.
```

### Debug Prompt E — Out-of-Range Output Values

```
The model is producing output values significantly outside [0, 1] — some outputs are as high as 3.0 or as low as -1.5. Without adding a sigmoid or hard clip inside the model, add a soft clamping mechanism at the end of RestoreNet.forward(): output = torch.sigmoid(output) * 1.2 - 0.1 to allow slight extrapolation beyond [0,1] while keeping values bounded. Only clip to [0,1] at save time in inference.py.
```

---

*End of Vibe-Coding Prompt Sequence*
*Total prompts: 22 main + 5 debug | Expected build time: 4–5 days active coding + training time*
