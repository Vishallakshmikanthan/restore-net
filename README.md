# 🚀 RestoreNet: Deep Learning for Semiconductor Image Restoration

<div align="center">

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![React](https://img.shields.io/badge/React-19.2+-61DAFB.svg)](https://reactjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

**KLA Hackathon 2026 Submission**  
*Team VibeSync*

</div>

---

## 📋 Table of Contents

- [Team Information](#-team-information)
- [Project Overview](#-project-overview)
- [Problem Statement](#-problem-statement)
- [Application Screenshots](#-application-screenshots)
- [Architecture](#-architecture)
- [Data Flow](#-data-flow)
- [CNN Implementation Details](#-cnn-implementation-details)
- [Web Interface](#-web-interface)
- [Performance Metrics](#-performance-metrics)
- [Installation & Setup](#-installation--setup)
- [Usage](#-usage)
- [Deployment](#-deployment)
- [Project Structure](#-project-structure)
- [Technical Stack](#-technical-stack)
- [Results](#-results)
- [Acknowledgments](#-acknowledgments)

---

## 👥 Team Information

**Team Name**: **VibeSync**

### Team Members

| Name | Role | Responsibilities |
|------|------|-----------------|
| **Vishal Lakshmikanthan** | Team Leader & ML Engineer | Architecture design, model training, deployment, full-stack integration |
| **Sneha C** | Developer & UI/UX Designer | Frontend development, user interface, testing, documentation |

**Hackathon**: KLA Hackathon 2026  
**Track**: Semiconductor Image Processing & AI  
**Submission Date**: August 18, 2026

---

## 🎯 Project Overview

**RestoreNet** is an end-to-end deep learning solution designed to restore severely degraded semiconductor electron microscope images. The system addresses the dual challenge of:

1. **Heavy Sensor Noise** (Gaussian + multiplicative speckle)
2. **2× Spatial Downsampling** (resolution loss)

### Key Features

✅ **Real-time Processing**: Sub-100ms inference latency on GPU  
✅ **High Fidelity**: 11.83 dB PSNR improvement over baseline  
✅ **Production Ready**: Full-stack web application with REST API  
✅ **Honest Metrics**: Real evaluation with ground truth, no mock data  
✅ **Deployment Ready**: Docker, cloud deployment guides included  

---

## 🔬 Problem Statement

### The Challenge

Semiconductor manufacturing relies on high-resolution electron microscope imaging for defect detection. However, real-world images suffer from:

```
Input Degradation = Gaussian Noise + Speckle Noise + 2× Downsampling
```

**Original Image** → *Degradation Process* → **Noisy Low-Resolution Input**

Traditional methods struggle with this compound degradation because:
- Simple denoising ignores resolution loss
- Super-resolution methods assume clean inputs
- Multi-stage approaches accumulate errors

### Our Solution

RestoreNet performs **joint restoration** using:
- Deep progressive residual learning
- Channel attention mechanisms
- Multi-term composite loss (L1 + SSIM + LPIPS)
- Global residual connections

---

## 📸 Application Screenshots

### 1. Three-Panel Image Display
![Three-Panel Display](screenshots/Screenshot%202026-08-18%20222636.png)
*Main interface showing Input (NoisyLR), Restored Output, and Residual thermal map*

### 2. Interactive Before/After Comparison
![Comparison Slider](screenshots/Screenshot%202026-08-18%20222701.png)
*Drag the cyan slider to compare degraded input vs. restored output in real-time*

### 3. Real-Time Metrics Dashboard
![Metrics Dashboard](screenshots/Screenshot%202026-08-18%20222710.png)
*Live display of PSNR, SSIM, LPIPS quality metrics and inference latency*

### 4. Pipeline Execution Profile
![Pipeline Visualization](screenshots/Screenshot%202026-08-18%20222720.png)
*Stage-wise timing breakdown showing I/O, preprocessing, inference, and post-processing*

---

## 🏗️ Architecture

### High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                          RestoreNet Pipeline                         │
└─────────────────────────────────────────────────────────────────────┘

Input Image                    Processing Stages                  Output Image
(128×128)                                                          (256×256)
                                                                   
   │                                                                    
   │                    ┌────────────────────┐                         
   └──────────────────> │  Bilinear Upsample │ ───────┐               
                        │      2× Scale      │        │               
                        └────────────────────┘        │               
                                 │                    │ (Skip)        
                                 ▼                    │               
                        ┌────────────────────┐        │               
                        │  Feature Extract   │        │               
                        │    Conv 3×3, 64    │        │               
                        └────────────────────┘        │               
                                 │                    │               
                                 ▼                    │               
                    ┌────────────────────────┐        │               
                    │   Residual Block 1     │        │               
                    │   (Conv-ReLU-Conv)     │        │               
                    └────────────────────────┘        │               
                                 │                    │               
                                 ▼                    │               
                    ┌────────────────────────┐        │               
                    │   Residual Block 2-4   │        │               
                    │   (Conv-ReLU-Conv)     │        │               
                    └────────────────────────┘        │               
                                 │                    │               
                                 ▼                    │               
                    ┌────────────────────────┐        │               
                    │  Channel Attention 1   │        │               
                    │   (SE Module r=16)     │        │               
                    └────────────────────────┘        │               
                                 │                    │               
                                 ▼                    │               
                    ┌────────────────────────┐        │               
                    │   Residual Block 5-9   │        │               
                    │   (Conv-ReLU-Conv)     │        │               
                    └────────────────────────┘        │               
                                 │                    │               
                                 ▼                    │               
                    ┌────────────────────────┐        │               
                    │  Channel Attention 2   │        │               
                    │   (SE Module r=16)     │        │               
                    └────────────────────────┘        │               
                                 │                    │               
                                 ▼                    │               
                    ┌────────────────────────┐        │               
                    │   Residual Block 10    │        │               
                    │   (Conv-ReLU-Conv)     │        │               
                    └────────────────────────┘        │               
                                 │                    │               
                                 ▼                    │               
                        ┌────────────────────┐        │               
                        │   Mid Conv 3×3     │        │               
                        └────────────────────┘        │               
                                 │                    │               
                                 ▼                    │               
                        ┌────────────────────┐        │               
                        │   Out Conv 3×3, 1  │        │               
                        └────────────────────┘        │               
                                 │                    │               
                                 └────────(+)─────────┘               
                                          │                           
                                          ▼                           
                                   Restored Image                     
                                     (256×256)                        
```

### Model Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Scale Factor** | 2× | Spatial upsampling ratio |
| **Feature Channels** | 64 | Base channel dimension |
| **Residual Blocks** | 10 | Depth of feature extraction |
| **Attention Blocks** | 2 | SE modules (every 5 blocks) |
| **Total Parameters** | 777,920 (~0.78M) | Efficient parameter count |
| **Input Size** | 128×128×1 | Single-channel grayscale |
| **Output Size** | 256×256×1 | Restored high-resolution |

---

## 📊 Data Flow

### Complete System Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         System Architecture                              │
└─────────────────────────────────────────────────────────────────────────┘

Frontend (React + Vite)          Backend (FastAPI)           Model (PyTorch)
─────────────────────            ────────────────           ─────────────────

┌─────────────────┐              ┌───────────────┐          ┌──────────────┐
│   File Upload   │              │  POST /api/   │          │  RestoreNet  │
│   (.npy array)  │──────────────>│   restore     │──────────>│   Model      │
└─────────────────┘   FormData   └───────────────┘  Tensor  └──────────────┘
                                          │                         │
                                          │                         │
┌─────────────────┐              ┌───────────────┐          ┌──────────────┐
│  Image Display  │<─────────────│   Response    │<─────────│  Inference   │
│  & Comparison   │   .npy + HDR │   (Binary)    │  ndarray │   Output     │
└─────────────────┘              └───────────────┘          └──────────────┘
        │                                 │
        │                                 │
        ▼                                 ▼
┌─────────────────┐              ┌───────────────┐
│  Metrics Bar    │<─────────────│  HTTP Headers │
│  PSNR/SSIM/     │   X-PSNR,    │  (Metrics)    │
│  LPIPS/Latency  │   X-SSIM,    │               │
└─────────────────┘   X-LPIPS    └───────────────┘
```

### Data Transformation Pipeline

```
Step 1: Data Loading
─────────────────────
File System (.npy)
    │
    ▼
NumPy Array (128, 128) float32
    │
    ▼
Normalize to [0, 1] range


Step 2: Preprocessing
──────────────────────
Input (H, W) float32
    │
    ▼
Add batch dim: (1, H, W)
    │
    ▼
Add channel dim: (1, 1, H, W)
    │
    ▼
Convert to PyTorch Tensor
    │
    ▼
Move to Device (CPU/CUDA)


Step 3: Neural Network Processing
──────────────────────────────────
Tensor (1, 1, 128, 128)
    │
    ├─────────────────────────────┐
    │                             │
    ▼                             │ (Global Skip)
Bilinear Upsample                 │
    │                             │
    ▼                             │
(1, 1, 256, 256)                  │
    │                             │
    ▼                             │
Conv Feature Extract              │
    │                             │
    ▼                             │
(1, 64, 256, 256)                 │
    │                             │
    ▼                             │
Residual Blocks (10×)             │
+ Channel Attention (2×)          │
    │                             │
    ▼                             │
(1, 64, 256, 256)                 │
    │                             │
    ▼                             │
Mid Conv + Output Conv            │
    │                             │
    ▼                             │
Residual (1, 1, 256, 256)         │
    │                             │
    └──────────(+)────────────────┘
                │
                ▼
        Output (1, 1, 256, 256)


Step 4: Post-processing
────────────────────────
Tensor (1, 1, 256, 256)
    │
    ▼
Remove batch/channel dims
    │
    ▼
NumPy Array (256, 256) float32
    │
    ▼
Clip to [0.0, 1.0] range
    │
    ▼
Save as .npy format
    │
    ▼
Return to Frontend
```

---

## 🧠 CNN Implementation Details

### 1. Residual Block Architecture

**Purpose**: Extract hierarchical features while maintaining gradient flow

```python
class ResidualBlock(nn.Module):
    """
    Input:  (B, C, H, W)
    Output: (B, C, H, W)
    
    Architecture:
    x ───┐
         │
         ├──> Conv3×3 ──> ReLU ──> Conv3×3 ──> (+) ──> out
         │                                      ▲
         └──────────────────────────────────────┘
                    (identity skip)
    """
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 
                               kernel_size=3, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 
                               kernel_size=3, padding=1, bias=True)
    
    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return out + residual  # Element-wise addition
```

**Key Features**:
- **Bypass connection** prevents vanishing gradients
- **Two conv layers** extract local patterns
- **ReLU activation** adds non-linearity
- **Preserves spatial dimensions** (same padding)

---

### 2. Channel Attention (Squeeze-and-Excitation)

**Purpose**: Dynamically reweight feature channels based on global context

```python
class ChannelAttention(nn.Module):
    """
    Input:  (B, C, H, W)
    Output: (B, C, H, W) with channel-wise scaling
    
    Architecture:
                    ┌──────────────────────┐
         x ─────────┤                      ├──────> x * w
                    │                      │
                    │  ┌──────────────┐    │
                    └─>│ Global Avg   │    │
                       │ Pool (H,W→1) │    │
                       └───────┬──────┘    │
                               │           │
                       ┌───────▼──────┐    │
                       │  FC (C→C/16) │    │
                       └───────┬──────┘    │
                               │           │
                       ┌───────▼──────┐    │
                       │     ReLU     │    │
                       └───────┬──────┘    │
                               │           │
                       ┌───────▼──────┐    │
                       │  FC (C/16→C) │    │
                       └───────┬──────┘    │
                               │           │
                       ┌───────▼──────┐    │
                       │   Sigmoid    │    │
                       └───────┬──────┘    │
                               │           │
                               └───────────┘
                                   w (channel weights)
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        reduced = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # (B,C,H,W) → (B,C,1,1)
        self.fc1 = nn.Conv2d(channels, reduced, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(reduced, channels, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Squeeze: Global spatial context
        w = self.avg_pool(x)           # (B, C, H, W) → (B, C, 1, 1)
        
        # Excitation: Channel importance
        w = self.relu(self.fc1(w))     # (B, C, 1, 1) → (B, C/16, 1, 1)
        w = self.sigmoid(self.fc2(w))  # (B, C/16, 1, 1) → (B, C, 1, 1)
        
        # Scale: Reweight input channels
        return x * w                   # Broadcast multiply
```

**Key Features**:
- **Global context**: Average pooling captures image-wide information
- **Bottleneck design**: Reduction ratio (16) adds efficiency
- **Sigmoid gating**: Produces attention weights in [0, 1]
- **Channel recalibration**: Emphasizes important features

---

### 3. Global Residual Connection

**Purpose**: Learn only the correction map Δ, not the full output

```python
def forward(self, x):
    """
    x: Input degraded image (B, 1, H, W)
    
    Flow:
    ┌───────────────────────────────────────────────┐
    │  x (128×128)                                  │
    │       │                                       │
    │       ├──────────────────────────────┐        │
    │       │                              │        │
    │       ▼                              │        │
    │  Upsample (Bilinear 2×)              │        │
    │       │                              │        │
    │       ▼                              │        │
    │  upsampled (256×256) ────────────────┤        │
    │       │                              │        │
    │       ▼                              │        │
    │  Feature Extraction                  │        │
    │  Residual Blocks × 10                │        │
    │  Channel Attention × 2               │        │
    │       │                              │        │
    │       ▼                              │        │
    │  Δ (residual correction)             │        │
    │       │                              │        │
    │       └──────────────(+)─────────────┘        │
    │                       │                       │
    │                       ▼                       │
    │              output = upsampled + Δ          │
    └───────────────────────────────────────────────┘
    """
    # Global skip connection path
    upsampled = self.upsample(x)
    
    # Feature extraction path
    feat = self.conv_in(upsampled)
    
    # Residual blocks with attention
    attn_idx = 0
    for i, block in enumerate(self.res_blocks):
        feat = block(feat)
        if self.use_attention and (i + 1) % 5 == 0:
            feat = self.attention_blocks[attn_idx](feat)
            attn_idx += 1
    
    feat = self.conv_mid(feat)
    residual = self.conv_out(feat)  # Δ correction map
    
    # Global residual addition
    return upsampled + residual
```

**Why This Works**:
1. **Easier optimization**: Network learns *what to fix* (Δ), not *what the output should be*
2. **Faster convergence**: Identity initialization naturally occurs
3. **Better gradients**: Direct path for gradient flow during backprop
4. **Preserves structure**: Upsampled input provides spatial prior

---

### 4. Composite Loss Function

**Purpose**: Balance pixel accuracy, structural similarity, and perceptual quality

```python
class RestorationLoss(nn.Module):
    """
    Total Loss = λ₁·L1 + λ₂·(1-SSIM) + λ₃·LPIPS
    
    Where:
    - L1: Pixel-wise absolute error
    - SSIM: Structural similarity (windowed)
    - LPIPS: Perceptual similarity (AlexNet features)
    """
    def __init__(self, 
                 lambda_pixel=1.0,   # L1 weight
                 lambda_ssim=0.3,    # SSIM weight
                 lambda_lpips=0.1):  # LPIPS weight
        super().__init__()
        self.l1 = nn.L1Loss()
        self.ssim_fn = SSIM()
        self.lpips_fn = lpips.LPIPS(net='alex').eval()
    
    def forward(self, pred, target):
        # 1. Pixel Loss: L1 distance
        l1_loss = self.l1(pred, target)
        
        # 2. Structural Loss: 1 - SSIM
        pred_clipped = torch.clamp(pred, 0.0, 1.0)
        ssim_val = self.ssim_fn(pred_clipped, target)
        ssim_loss = 1.0 - ssim_val
        
        # 3. Perceptual Loss: LPIPS
        # Convert to RGB [-1, 1] for LPIPS
        p_rgb = pred_clipped.repeat(1, 3, 1, 1) * 2.0 - 1.0
        t_rgb = target.repeat(1, 3, 1, 1) * 2.0 - 1.0
        lpips_loss = self.lpips_fn(p_rgb, t_rgb).mean()
        
        # Total weighted loss
        total = (lambda_pixel * l1_loss + 
                 lambda_ssim * ssim_loss + 
                 lambda_lpips * lpips_loss)
        
        return total
```

**Loss Components**:

| Loss Type | Formula | Purpose | Weight (λ) |
|-----------|---------|---------|-----------|
| **L1** | `∑|pred - target|` | Pixel accuracy | 1.0 |
| **SSIM** | `1 - SSIM(pred, target)` | Structural similarity | 0.3 |
| **LPIPS** | `AlexNet_dist(pred, target)` | Perceptual quality | 0.1 |

**SSIM Implementation**:
```python
# Gaussian window convolution
μ_x = F.conv2d(x, gaussian_window)  # Mean
σ_x² = F.conv2d(x², gaussian_window) - μ_x²  # Variance
σ_xy = F.conv2d(xy, gaussian_window) - μ_x·μ_y  # Covariance

SSIM = (2μ_x·μ_y + C₁)(2σ_xy + C₂) / [(μ_x² + μ_y² + C₁)(σ_x² + σ_y² + C₂)]
```

---

## 🎨 Web Interface

### Frontend Architecture

Built with **React 19** + **Vite** + **Tailwind CSS**

#### Application Overview

![Main Interface Overview](screenshots/Screenshot%202026-08-18%20222636.png)
*RestoreNet web application with integrated control panel and visualization*

### Key Components

#### 1. **Upload Zone** (Left Panel)
```javascript
// Features:
- Drag & drop .npy file upload
- Array metadata display (shape, dtype, min/max/mean)
- Ground truth upload for real metrics
- Synthetic wafer generator for demos
- Advanced config (denoising strength)
- Model checkpoint indicator (best_model.pt)
```

#### 2. **Image Display** (Center)
```javascript
// Three-panel comparison:
1. INPUT (NoisyLR)    - Original degraded image
2. RESTORED (Output)  - AI-enhanced result with interactive slider
3. RESIDUAL (Δ Map)   - Thermal heatmap of corrections
```

![Image Comparison Interface](screenshots/Screenshot%202026-08-18%20222701.png)
*Interactive comparison slider showing before/after restoration*

**Interactive Slider**:
- Drag left/right to compare before/after
- Real-time visual comparison
- Smooth transition animation

#### 3. **Metrics Dashboard** (Bottom)
```javascript
// Real-time metrics display:
- PSNR (dB): Peak Signal-to-Noise Ratio
- SSIM: Structural Similarity Index
- LPIPS: Learned Perceptual Image Patch Similarity
- E2E Runtime (ms): End-to-end inference latency

// Shows "N/A (GT Required)" when no ground truth provided
// Displays real computed metrics when GT is uploaded
```

![Metrics Dashboard](screenshots/Screenshot%202026-08-18%20222710.png)
*Real-time metrics display showing PSNR, SSIM, LPIPS, and latency measurements*

#### 4. **System Trace** (Top)
```javascript
// Live processing visualization:
- IDLE: Waiting for input
- STREAM_ACTIVE: Processing in progress
- Animated waveform during inference
```

#### 5. **Pipeline Visualization** (Bottom)
```javascript
// Stage-wise timing breakdown:
┌─────────┬──────────┬───────────┬───────────┬─────────┐
│ I/O     │ PRE      │ AUAL_GEN  │ UNFOLD    │ POST    │
│ (10%)   │ (18%)    │ (18%)     │ (42%)     │ (12%)   │
└─────────┴──────────┴───────────┴───────────┴─────────┘
```

![Pipeline Trace Visualization](screenshots/Screenshot%202026-08-18%20222720.png)
*Pipeline execution profile showing stage-wise timing breakdown and total latency*

### UI/UX Design Principles

✨ **Dark Theme**: Optimized for technical users  
✨ **Monospace Typography**: Professional, data-focused aesthetic  
✨ **Cyan Accents**: High-contrast, sci-fi inspired color scheme  
✨ **Real-time Feedback**: Status indicators, progress animations  
✨ **Responsive Layout**: Adapts to different screen sizes  

---

## � Performance Metrics

### Quantitative Results

Comprehensive evaluation on KLA test dataset (paired images):

| Metric | Baseline CNN | RestoreNet | Improvement |
|--------|-------------|------------|-------------|
| **PSNR** ↑ | 12.81 dB | **24.64 dB** | **+11.83 dB** ✅ |
| **SSIM** ↑ | 0.4210 | **0.6646** | **+0.2436** ✅ |
| **LPIPS** ↓ | 0.5120 | **0.3636** | **-0.1484 (29% better)** ✅ |
| **Parameters** | 222.8K | 777.9K | +3.5× |
| **Latency (CPU)** | 24.2 ms | 105.7 ms | +4.4× |
| **Latency (GPU)** | ~8 ms | **<10 ms** | **Real-time** ✅ |

### Benchmark Analysis

```
Metric Improvements:
┌─────────────────────────────────────────────────┐
│ PSNR:  12.81 ▓▓▓▓▓░░░░░░░░░░░░░░░░░             │
│        24.64 ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  (+92%)     │
├─────────────────────────────────────────────────┤
│ SSIM:  0.421 ▓▓▓▓▓▓▓▓░░░░░░░░░░░░░              │
│        0.665 ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  (+58%)        │
├─────────────────────────────────────────────────┤
│ LPIPS: 0.512 ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓              │
│        0.364 ▓▓▓▓▓▓▓▓▓▓▓▓▓  (-29%)              │
└─────────────────────────────────────────────────┘
```

### Target vs. Achieved

| Target | Achieved | Status |
|--------|----------|--------|
| PSNR > 28.0 dB | 24.64 dB | ⚠️ Close (87.9%) |
| SSIM > 0.85 | 0.6646 | ⚠️ Moderate (78.2%) |
| LPIPS < 0.15 | 0.3636 | ⚠️ Above target |
| Latency < 100ms (GPU) | <10ms | ✅ **Exceeded** |

**Analysis**:
- **Strong PSNR improvement** but below target (requires deeper network or longer training)
- **Excellent latency** well within real-time constraints
- **Significant perceptual quality gains** over baseline
- **Production-ready** for practical deployment

---

## 🛠️ Installation & Setup

### Prerequisites

- **Python**: 3.10 or higher
- **Node.js**: 16 or higher
- **CUDA**: 11.8+ (optional, for GPU acceleration)
- **Git**: For version control

### Quick Start (Windows)

#### Option 1: Automated Setup

```bash
# Clone repository
cd c:\Users\Lenovo\Downloads\restore-net\kla-image-restoration

# Run automated setup
setup_deployment.bat

# Start production servers
start_production.bat
```

#### Option 2: Manual Setup

**Step 1: Backend Setup**

```bash
# Navigate to project directory
cd c:\Users\Lenovo\Downloads\restore-net\kla-image-restoration

# Create virtual environment
python -m venv .venv

# Activate virtual environment
.venv\Scripts\activate

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Verify model checkpoint
dir checkpoints\best_model.pt
```

**Step 2: Frontend Setup**

```bash
# Navigate to frontend directory
cd frontend

# Install Node.js dependencies
npm install

# Build production bundle
npm run build
```

**Step 3: Start Services**

Terminal 1 (Backend):
```bash
cd c:\Users\Lenovo\Downloads\restore-net\kla-image-restoration
.venv\Scripts\activate
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 2
```

Terminal 2 (Frontend):
```bash
cd c:\Users\Lenovo\Downloads\restore-net\kla-image-restoration\frontend
npm run preview
```

**Access the application**:
- Frontend: `http://localhost:4173`
- Backend API: `http://localhost:8000`
- API Health: `http://localhost:8000/api/health`

### Linux/Mac Setup

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start backend
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 2 &

# Build and serve frontend
cd frontend
npm install
npm run build
npm run preview
```

---

## 🚀 Usage

### 1. Basic Inference (No Ground Truth)

```bash
# Using Web Interface:
1. Open http://localhost:4173
2. Click "Load Synthetic Wafer" or drag .npy file
3. Click "RUN INFERENCE"
4. View restored output and residual map
5. Metrics show "N/A" (GT Required)

# Using CLI:
python inference.py \
  --input_dir data/NoisyLR \
  --output_dir results/outputs \
  --model_path checkpoints/best_model.pt \
  --device cuda
```

### 2. Evaluation with Ground Truth

```bash
# Using Web Interface:
1. Upload input .npy file
2. Click "+ Add GT" and upload ground truth .npy
3. Click "Evaluate (Real Metrics)"
4. View real PSNR/SSIM/LPIPS values

# Using CLI:
python scripts/evaluate.py \
  --gt_dir data/GT \
  --pred_dir results/outputs \
  --output_json results/metrics/evaluation.json
```

### 3. Batch Processing

```bash
# Process entire directory
python inference.py \
  --input_dir data/NoisyLR \
  --output_dir results/batch \
  --model_path checkpoints/best_model.pt \
  --batch_size 4 \
  --device cuda
```

### 4. Training Custom Model

```bash
# Train from scratch
python scripts/train.py \
  --config configs/train.yaml \
  --epochs 50 \
  --batch_size 8 \
  --lr 1e-4

# Resume from checkpoint
python scripts/train.py \
  --resume checkpoints/best_model.pt \
  --epochs 20
```

### API Usage Examples

**Python**:
```python
import requests
import numpy as np

# Load input
input_data = np.load('sample_input.npy')

# Prepare request
files = {'file': ('input.npy', input_data.tobytes())}
response = requests.post('http://localhost:8000/api/restore', files=files)

# Parse results
output_data = np.frombuffer(response.content, dtype=np.float32)
psnr = float(response.headers['X-PSNR'])
ssim = float(response.headers['X-SSIM'])
latency = float(response.headers['X-Latency-Ms'])

print(f"PSNR: {psnr:.2f} dB, SSIM: {ssim:.3f}, Latency: {latency:.1f} ms")
```

**cURL**:
```bash
curl -X POST http://localhost:8000/api/restore \
  -F "file=@sample_input.npy" \
  --output restored_output.npy \
  -D headers.txt
```

---

## 🐳 Deployment

### Quick Deploy to Vercel + Render

**Fastest way to get RestoreNet live**:

```bash
# 1. Validate setup
deploy-checklist.bat

# 2. Push to GitHub
quick-deploy.bat

# 3. Deploy backend to Render (5-10 min)
# 4. Deploy frontend to Vercel (2-3 min)
# 5. Test deployment
test-deployment.bat
```

📚 **Deployment Resources**:
- **Quick Start**: [DEPLOYMENT_QUICK_REFERENCE.md](DEPLOYMENT_QUICK_REFERENCE.md) - 1-page guide
- **Complete Guide**: [VERCEL_RENDER_DEPLOYMENT.md](VERCEL_RENDER_DEPLOYMENT.md) - Step-by-step
- **Overview**: [DEPLOYMENT_SUMMARY.md](DEPLOYMENT_SUMMARY.md) - What's included
- **Navigation**: [DEPLOYMENT_INDEX.md](DEPLOYMENT_INDEX.md) - Find the right resource

### Other Deployment Options

#### Docker Deployment

```bash
# Build and start all services
docker-compose up -d --build

# Check logs
docker-compose logs -f

# Stop services
docker-compose down

# Access application
# Frontend: http://localhost:3000
# Backend: http://localhost:8000
```

#### Local Production Setup

See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for detailed instructions on:
- Local production servers
- AWS / GCP / Azure cloud deployment
- Kubernetes orchestration
- Custom domain configuration

---

## 📁 Project Structure

```
kla-image-restoration/
├── 📄 README.md                    # This file
├── 📄 requirements.txt             # Python dependencies
├── 📄 package.json                 # Node.js dependencies
├── 📄 DEPLOYMENT_GUIDE.md          # Deployment instructions
├── 📄 VIDEO_SCRIPT.md              # Demo video script
├── 📄 inference.py                 # Main inference entry point
├── 📄 docker-compose.yml           # Multi-container orchestration
├── 📄 Dockerfile                   # Backend container image
├── 📄 setup_deployment.bat         # Windows setup script
├── 📄 start_production.bat         # Windows startup script
│
├── 📁 src/                         # Source code
│   ├── 📁 models/                  # Neural network architectures
│   │   ├── baseline.py             # Baseline CNN implementation
│   │   ├── restorenet.py           # Main RestoreNet model
│   │   └── blocks.py               # Reusable building blocks
│   │       ├── ResidualBlock       # Residual connection module
│   │       ├── ChannelAttention    # Squeeze-Excitation module
│   │       ├── UpsampleBlock       # Interpolation upsampler
│   │       └── PixelShuffleBlock   # Learned upsampler
│   │
│   ├── 📁 training/                # Training utilities
│   │   ├── losses.py               # Loss functions (L1, SSIM, LPIPS)
│   │   ├── metrics.py              # Evaluation metrics
│   │   ├── trainer.py              # Training loop implementation
│   │   └── validation.py           # Validation logic
│   │
│   ├── 📁 data/                    # Data handling
│   │   ├── dataset.py              # Dataset loader
│   │   ├── augmentation.py         # Data augmentation
│   │   └── preprocessing.py        # Preprocessing utilities
│   │
│   ├── 📁 api/                     # REST API
│   │   └── main.py                 # FastAPI application
│   │       ├── POST /api/restore   # Inference endpoint
│   │       ├── POST /api/evaluate  # Evaluation endpoint
│   │       └── GET /api/health     # Health check
│   │
│   └── 📁 utils/                   # Helper utilities
│       ├── visualization.py        # Plotting and visualization
│       └── logger.py               # Logging configuration
│
├── 📁 frontend/                    # React web application
│   ├── 📄 package.json             # Frontend dependencies
│   ├── 📄 vite.config.js           # Vite build configuration
│   ├── 📄 tailwind.config.js       # Tailwind CSS configuration
│   ├── 📄 Dockerfile               # Frontend container image
│   ├── 📄 nginx.conf               # Nginx web server config
│   │
│   ├── 📁 src/
│   │   ├── App.jsx                 # Main application component
│   │   │
│   │   ├── 📁 components/          # React components
│   │   │   ├── UploadZone.jsx      # File upload interface
│   │   │   ├── ImageDisplay.jsx    # Image comparison viewer
│   │   │   ├── MetricsBar.jsx      # Metrics dashboard
│   │   │   ├── WaveformTrace.jsx   # System trace visualization
│   │   │   └── PipelineTrace.jsx   # Pipeline timing chart
│   │   │
│   │   ├── 📁 api/                 # API client
│   │   │   └── client.js           # HTTP request handlers
│   │   │
│   │   ├── 📁 utils/               # Frontend utilities
│   │   │   └── sampleData.js       # Synthetic data generator
│   │   │
│   │   └── 📁 styles/              # CSS styles
│   │       └── index.css           # Global styles + Tailwind
│   │
│   └── 📁 dist/                    # Production build output
│
├── 📁 scripts/                     # Utility scripts
│   ├── train.py                    # Model training
│   ├── train_baseline.py           # Baseline training
│   ├── evaluate.py                 # Quantitative evaluation
│   ├── benchmark.py                # Performance benchmarking
│   ├── ablation.py                 # Ablation studies
│   ├── inspect_dataset.py          # Dataset statistics
│   ├── dry_run.py                  # End-to-end smoke test
│   └── generate_pptx.py            # Generate presentation
│
├── 📁 configs/                     # Configuration files
│   ├── train.yaml                  # Training hyperparameters
│   └── baseline.yaml               # Baseline configuration
│
├── 📁 checkpoints/                 # Model checkpoints
│   ├── best_model.pt               # Best trained model (PRODUCTION)
│   ├── baseline_best.pt            # Best baseline model
│   └── checkpoint_epoch_*.pt       # Training snapshots
│
├── 📁 data/                        # Dataset directory
│   ├── 📁 GT/                      # Ground truth images
│   └── 📁 NoisyLR/                 # Degraded input images
│
├── 📁 results/                     # Output directory
│   ├── 📁 inference_outputs/       # Restored images
│   ├── 📁 metrics/                 # Evaluation results (JSON)
│   └── 📁 visualizations/          # Plots and figures
│
├── 📁 tests/                       # Unit tests
│   ├── test_models.py              # Model tests
│   ├── test_losses.py              # Loss function tests
│   └── test_api.py                 # API endpoint tests
│
└── 📁 logs/                        # Training logs
    └── tensorboard/                # TensorBoard event files
```

---

## 🔧 Technical Stack

### Backend

| Technology | Version | Purpose |
|------------|---------|---------|
| **Python** | 3.10+ | Core programming language |
| **PyTorch** | 2.1.0 | Deep learning framework |
| **FastAPI** | 0.100+ | REST API framework |
| **Uvicorn** | 0.23+ | ASGI server |
| **NumPy** | 1.24+ | Numerical computing |
| **scikit-image** | 0.21+ | Image processing utilities |
| **LPIPS** | 0.1.4+ | Perceptual loss function |

### Frontend

| Technology | Version | Purpose |
|------------|---------|---------|
| **React** | 19.2+ | UI framework |
| **Vite** | 8.2+ | Build tool & dev server |
| **Tailwind CSS** | 4.3+ | Utility-first styling |
| **Lucide React** | 1.31+ | Icon library |
| **Recharts** | 3.10+ | Data visualization |

### DevOps

| Technology | Purpose |
|------------|---------|
| **Docker** | Containerization |
| **Docker Compose** | Multi-container orchestration |
| **Nginx** | Static file serving |
| **Git** | Version control |

---

## 📊 Results

### Visual Comparison

**Example 1: Wafer Defect Restoration**
```
Input (NoisyLR)          Restored (Output)        Ground Truth
  128×128                    256×256                  256×256
┌───────────┐            ┌───────────┐            ┌───────────┐
│ ▒▒▒▒▒░░░  │   ───────> │ ████████  │            │ ████████  │
│ ▒░▒▒░▒▒░  │   RestoreN │ ████████  │            │ ████████  │
│ ░▒░▒▒▒░▒  │     Net    │ ████████  │            │ ████████  │
│ ▒▒░░▒▒▒░  │            │ ████████  │            │ ████████  │
└───────────┘            └───────────┘            └───────────┘
  Noisy +                  Denoised +               Reference
  Low-Res                  Upscaled

PSNR: 24.8 dB            SSIM: 0.67               LPIPS: 0.35
```

### Ablation Study Results

| Configuration | PSNR (dB) | SSIM | LPIPS |
|--------------|-----------|------|-------|
| Baseline (3-block CNN) | 12.81 | 0.421 | 0.512 |
| + Residual Blocks (10) | 18.42 | 0.524 | 0.445 |
| + Channel Attention (2) | 22.15 | 0.618 | 0.385 |
| + Composite Loss (L1+SSIM+LPIPS) | **24.64** | **0.665** | **0.364** |

**Key Takeaways**:
1. **Residual connections** provide +5.61 dB PSNR improvement
2. **Channel attention** adds +3.73 dB, crucial for feature selection
3. **Composite loss** fine-tunes perceptual quality (+2.49 dB)

---

## 🙏 Acknowledgments

### Team VibeSync

This project was developed as part of the **KLA Hackathon 2026** by:

- **Vishal Lakshmikanthan** (Team Leader) - Architecture, training, deployment
- **Sneha C** - Frontend development, UI/UX design, testing

### External Resources

We acknowledge the use of the following open-source libraries and resources:

1. **PyTorch** - Deep learning framework ([pytorch.org](https://pytorch.org))
2. **LPIPS** - Learned Perceptual Image Patch Similarity ([github.com/richzhang/PerceptualSimilarity](https://github.com/richzhang/PerceptualSimilarity))
3. **FastAPI** - Modern REST API framework ([fastapi.tiangolo.com](https://fastapi.tiangolo.com))
4. **React** - UI library ([react.dev](https://react.dev))
5. **Tailwind CSS** - Utility-first CSS framework ([tailwindcss.com](https://tailwindcss.com))

### References

**Key Papers & Techniques**:
- He et al., "Deep Residual Learning for Image Recognition" (CVPR 2016)
- Hu et al., "Squeeze-and-Excitation Networks" (CVPR 2018)
- Wang et al., "Image Quality Assessment: From Error Visibility to Structural Similarity" (IEEE TIP 2004)
- Zhang et al., "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric" (CVPR 2018)

---

## 📜 License

This project is licensed under the **MIT License**.

```
MIT License

Copyright (c) 2026 Team VibeSync (Vishal Lakshmikanthan, Sneha C)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 📞 Contact & Support

### Team Contacts

- **Vishal Lakshmikanthan** (Team Leader)
  - GitHub: [@Vishallakshmikanthan](https://github.com/Vishallakshmikanthan)
  - Role: ML Architecture & Backend Development

- **Sneha C**
  - Role: Frontend Development & UI/UX Design

### Repository

- **GitHub**: [github.com/Vishallakshmikanthan/restore-net](https://github.com/Vishallakshmikanthan/restore-net)
- **Issues**: [Report bugs or request features](https://github.com/Vishallakshmikanthan/restore-net/issues)
- **Documentation**: See `docs/` folder for detailed guides

### Hackathon Submission

- **Event**: KLA Hackathon 2026
- **Team**: VibeSync
- **Track**: Semiconductor Image Processing & AI
- **Submission**: August 18, 2026

---

## 🎯 Future Work

### Planned Improvements

1. **Model Enhancements**
   - [ ] Transformer-based attention mechanisms
   - [ ] Multi-scale feature extraction
   - [ ] Dynamic depth adjustment
   - [ ] Uncertainty quantification

2. **Performance Optimization**
   - [ ] TensorRT optimization for NVIDIA GPUs
   - [ ] ONNX export for cross-platform inference
   - [ ] Quantization (INT8) for edge deployment
   - [ ] Model pruning and distillation

3. **Feature Additions**
   - [ ] Real-time video processing
   - [ ] Batch processing queue system
   - [ ] User authentication & project management
   - [ ] Cloud storage integration
   - [ ] Collaborative annotation tools

4. **Dataset & Training**
   - [ ] Expand training dataset
   - [ ] Domain adaptation techniques
   - [ ] Self-supervised pre-training
   - [ ] Active learning pipeline

---

## 📚 Documentation

### Additional Resources

- [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) - Complete deployment instructions
- [VIDEO_SCRIPT.md](VIDEO_SCRIPT.md) - Demo video production guide
- [API Documentation](http://localhost:8000/docs) - Interactive API docs (when server is running)
- [Architecture Notebook](docs/architecture.ipynb) - Detailed architecture walkthrough

### Troubleshooting

**Common Issues**:

1. **Backend won't start**
   ```bash
   # Check if port 8000 is in use
   netstat -ano | findstr :8000
   # Kill process if needed
   taskkill /PID <PID> /F
   ```

2. **Frontend build fails**
   ```bash
   # Clear node_modules and reinstall
   cd frontend
   rm -rf node_modules package-lock.json
   npm install
   ```

3. **Model file not found**
   ```bash
   # Verify checkpoint exists
   ls -lh checkpoints/best_model.pt
   # If missing, train the model
   python scripts/train.py --config configs/train.yaml
   ```

4. **CUDA out of memory**
   ```python
   # Use CPU instead
   --device cpu
   # Or reduce batch size
   --batch_size 1
   ```

---

<div align="center">

## 🌟 Star this repository if you found it helpful!

**Made with ❤️ by Team VibeSync for KLA Hackathon 2026**

[⬆ Back to Top](#-restorenet-deep-learning-for-semiconductor-image-restoration)

</div>
