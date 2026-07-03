# Hybrid CNN–Transformer Image Denoising

A **hybrid residual CNN–Transformer architecture** for image denoising that combines DnCNN-style convolutional blocks with Transformer-based attention mechanisms (inspired by Restormer). This model is trained and evaluated on the **BSD500** dataset, supporting both **Gaussian** and **non-Gaussian** noise removal with comprehensive evaluation metrics.

---

## ✨ Features

- **Hybrid CNN + Transformer Architecture**: Combines the efficiency of CNNs with the global receptive field of Transformers
- **Residual Learning**: Model predicts noise residuals rather than clean images directly
- **Multiple Noise Types**: Support for Gaussian and non-Gaussian (e.g., speckle, impulse) noise
- **Gated Fusion Layer**: Intelligent fusion mechanism to combine CNN and Transformer features
- **Comprehensive Metrics**: Evaluation using PSNR and SSIM metrics
- **Batch Processing & Inference**: Efficient denoising with support for image batches
- **Interactive GUI**: User-friendly panel for batch denoising operations
- **Configurable Training**: YAML-based configuration for easy experimentation
- **TensorBoard Integration**: Real-time training monitoring and visualization

---

## 📁 Project Structure

```
Hybrid-CNN-Transformer-Denoising/
│
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── config.yaml                    # Training and model configuration
├── test_torch.py                  # PyTorch installation verification
│
├── data/                          # Dataset storage
│   ├── raw/                       # Original BSD500 images
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── noisy/                     # Noisy images organized by noise type
│       ├── gaussian/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       ├── mixed/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       └── nongaussian/
│           ├── train/
│           ├── val/
│           └── test/
│
├── datasets/                      # Data loading and preprocessing
│   ├── __init__.py
│   ├── dataset_loader.py          # DataLoader for BSD500
│   ├── noise_generator.py         # Noise generation utilities
│   └── data_augmentation.py       # Image augmentation techniques
│
├── models/                        # Model architecture components
│   ├── __init__.py
│   ├── cnn_blocks.py              # DnCNN-style CNN blocks
│   ├── attention_blocks.py        # Transformer attention blocks
│   ├── fusion_layer.py            # CNN-Transformer fusion mechanism
│   ├── hybrid_model.py            # Main hybrid architecture
│   └── restormer_model.py         # Restormer-based components
│
├── train/                         # Training pipeline
│   ├── __init__.py
│   ├── train.py                   # Main training script
│   └── loss.py                    # Loss functions (Charbonnier + SSIM)
│
├── evaluate/                      # Evaluation module
│   ├── __init__.py
│   ├── evaluate.py                # Evaluation script
│   └── metrics.py                 # PSNR, SSIM metrics
│
├── inference/                     # Inference and denoising
│   └── denoise.py                 # Denoising utilities
│
├── GUI/                           # Interactive GUI application
│   └── guipanel.py                # GUI panel for batch denoising
│
├── experiments/                   # Training artifacts
│   ├── checkpoints/               # Saved model weights
│   │   └── hybrid_best.pth        # Best model checkpoint
│   └── logs/                      # TensorBoard event files
│
├── results/                       # Output and results
│   ├── paper_text.txt             # Paper/documentation
│   └── images/                    # Denoised output images
│
├── scripts/                       # Utility scripts
│   ├── add_noise.py               # Add noise to clean images
│   ├── moderate_denoise.py        # Moderate-level denoising
│   └── postprocess.py             # Post-processing utilities
│
├── Test/                          # Test data and scripts
│
└── venv/                          # Python virtual environment
```

---

## 🚀 Quick Start

### 1. Installation

**Prerequisites**: Python 3.8+, CUDA 11.0+ (for GPU support)

```bash
# Clone the repository
cd "Hybrid CNN Transformer Denoising"

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify Installation

```bash
python test_torch.py
```

---

## ⚙️ Configuration

Edit `config.yaml` to customize training parameters:

```yaml
dataset:
  name: BSD500
  root: data
  patch_size: 128          # Patch size for training
  train_split: train       # Training split
  val_split: val           # Validation split
  augment: true            # Enable data augmentation

noise:
  gaussian:
    enabled: true
    sigma_values: [15, 25, 50]    # Noise levels
  non_gaussian:
    enabled: true
  mode: "gaussian"         # "gaussian", "non_gaussian", or "both"

model:
  type: "hybrid"           # Model type
  base_channels: 48        # Base channel count
  num_cnn_blocks: 17       # DnCNN depth
  num_transformer_blocks: [2, 2, 4, 4]  # Transformer stages
  fusion_type: "gate"      # Fusion mechanism
  use_residual_learning: true

training:
  batch_size: 2
  epochs: 100
  lr: 0.0003
  optimizer: "adamw"
  scheduler: "cosine"
  loss_type: "charbonnier_ssim"
  save_checkpoint_dir: "experiments/checkpoints"
  log_dir: "experiments/logs"
```

---

## 📊 Dataset Preparation

### Add Noise to Dataset

```bash
python scripts/add_noise.py
```

This script:
- Loads clean BSD500 images
- Adds Gaussian noise with specified σ values
- Adds non-Gaussian noise (speckle, impulse)
- Saves noisy images to `data/noisy/` with appropriate splits

---

## 🏋️ Training

### Basic Training

```bash
cd train
python train.py
```

The script will:
- Load configuration from `config.yaml`
- Initialize the hybrid model
- Train on BSD500 dataset
- Save checkpoints to `experiments/checkpoints/`
- Log metrics to TensorBoard (`experiments/logs/`)

### Monitor Training with TensorBoard

```bash
tensorboard --logdir experiments/logs
```

Open `http://localhost:6006` in your browser.

---

## 📈 Evaluation

### Evaluate on Test Set

```bash
cd evaluate
python evaluate.py
```

Generates:
- PSNR and SSIM scores
- Results saved to `results/`

### Available Metrics

- **PSNR** (Peak Signal-to-Noise Ratio): Measures reconstruction quality
- **SSIM** (Structural Similarity Index): Measures perceptual similarity

---

## 🖼️ Inference

### Denoise Single Images

```bash
from inference.denoise import denoise_image
import torch

# Load model
model = torch.load("experiments/checkpoints/hybrid_best.pth")

# Denoise image
denoised = denoise_image(image_path="path/to/noisy/image.png", model=model)
```

### Batch Denoising via GUI

```bash
cd GUI
python guipanel.py
```

The GUI provides:
- Batch image processing
- Real-time visualization
- Output format options
- Pre-trained model loading

---

## 🔧 Model Architecture

### Hybrid Denoiser Components

```
Input Image
    ↓
┌─────────────────────────────────────┐
│   CNN Pathway (DnCNN-style)         │
│   - 17 Conv blocks with ReLU        │
│   - Residual connections            │
└─────────────┬───────────────────────┘
              │
         ┌────┴────┐
         ↓         ↓
    ┌─────────────────┐
    │  Fusion Layer   │
    │  (Gated)        │
    └─────────────────┘
         ↓         ↑
┌─────────────────────────────────────┐
│ Transformer Pathway (Restormer)     │
│ - Multi-head self-attention         │
│ - Feed-forward networks             │
│ - 4 stages with increasing depth    │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│   Residual Connection               │
│   Output = Input - Predicted_Noise  │
└─────────────────────────────────────┘
    ↓
Denoised Image
```

### Key Design Decisions

- **Gated Fusion**: Learns optimal combination of CNN and Transformer features
- **Residual Learning**: Improves training stability and convergence
- **Multi-scale Processing**: Transformer stages handle different receptive fields
- **Efficient Attention**: Reduces computational overhead in attention mechanisms

---

## 📝 Training Details

### Loss Function
- **Charbonnier Loss**: Robust to outliers, smooth gradients
- **SSIM Loss**: Perceptual quality preservation
- **Combined Loss**: Weighted sum of both components

### Optimization
- **Optimizer**: AdamW with weight decay regularization
- **Scheduler**: Cosine annealing for learning rate
- **Batch Size**: 2 (adjustable based on GPU memory)
- **Epochs**: 100 with early stopping capability

### Data Augmentation
- Random horizontal/vertical flips
- Random rotations (90°, 180°, 270°)
- Random crop/zoom operations

---

## 📊 Experimental Results

### Performance Metrics

Training conducted on BSD500 dataset:
- **Test PSNR**: ~28-32 dB (varies by noise level)
- **Test SSIM**: ~0.80-0.95
- **Inference Time**: ~0.2-0.5s per image (GPU)

### Checkpoint Location
```
experiments/checkpoints/hybrid_best.pth
```

---

## 🛠️ Utilities & Scripts

### Add Noise to Images
```bash
python scripts/add_noise.py
```

### Moderate Denoising
```bash
python scripts/moderate_denoise.py
```

### Post-processing
```bash
python scripts/postprocess.py
```

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| torch | Deep learning framework |
| torchvision | Computer vision utilities |
| numpy | Numerical processing |
| opencv-python | Image I/O and processing |
| Pillow | Image handling |
| scikit-image | Advanced image processing |
| matplotlib | Visualization |
| pyyaml | Configuration parsing |
| tensorboard | Training visualization |
| einops | Tensor manipulation |

See `requirements.txt` for versions.

---

## 🤝 Project Organization

- **Modular Design**: Separate modules for data, models, training, and evaluation
- **Configuration-Driven**: YAML config for reproducibility
- **Logging & Monitoring**: TensorBoard integration for experiment tracking
- **Extensible**: Easy to add new architectures, loss functions, or noise types

---

## 💡 Tips for Best Results

1. **Adjust Patch Size**: Larger patches for complex images, smaller for speed
2. **Tune Learning Rate**: Start with 1e-3, decrease if unstable
3. **Monitor SSIM Loss**: Often more important than PSNR for visual quality
4. **Use Mixed Noise**: Training on mixed noise types improves generalization
5. **Data Augmentation**: Essential for preventing overfitting

---

## 📝 Notes

- **GPU Memory**: Minimum 4GB VRAM recommended; adjust batch size if needed
- **Training Time**: ~10-20 hours on GPU (NVIDIA RTX 3090)
- **Model Size**: ~15-20MB checkpoint file
- **Inference Speed**: GPU required for real-time processing

---

## 📄 References

- DnCNN: [Image Denoising via CNNs: An Adversarial Approach](https://arxiv.org/abs/1608.03981)
- Restormer: [Restormer: Efficient Transformer for High-Resolution Image Restoration](https://arxiv.org/abs/2111.09881)
- BSD500: [Contour Detection and Hierarchical Image Segmentation](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/)

---

## 📧 Contact & Support

For issues or questions, please open an issue in the repository or contact the project maintainer.

---

**Last Updated**: 2026-07-03