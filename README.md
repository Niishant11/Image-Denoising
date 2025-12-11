# Hybrid CNN–Transformer Image Denoising

This repository implements a **hybrid residual CNN–Transformer architecture** for image denoising,
combining DnCNN-style convolutional blocks with Transformer-based attention (inspired by Restormer).
The model is trained and evaluated primarily on the **BSD500** dataset under Gaussian and non-Gaussian noise.

---

## 🔧 Features

- Hybrid **CNN + Transformer** denoising architecture  
- **Residual learning**: model predicts noise, not clean image directly  
- Supports **Gaussian** and **non-Gaussian** noise  
- **Noise-aware** (with noise-level map) and **noise-blind** modes  
- Evaluation using **PSNR** and **SSIM**  
- Cross-dataset generalization (BSD500 → BSD68, DIV2K, etc.)  
- Scripts for dataset preparation, noise generation, training, evaluation, and inference  

---

## 📁 Project Structure (Simplified)

```text
Hybrid-CNN-Transformer-Denoising/
│
├── README.md
├── requirements.txt
├── config.yaml
│
├── data/
│   ├── raw/           # Original BSD500 (train/val/test)
│   ├── noisy/         # Noisy images (Gaussian / Non-Gaussian)
│   └── processed/     # Patches / normalized data
│
├── datasets/
│   ├── dataset_loader.py
│   ├── noise_generator.py
│   └── data_augmentation.py
│
├── models/
│   ├── cnn_blocks.py
│   ├── attention_blocks.py
│   ├── fusion_layer.py
│   └── hybrid_model.py
│
├── train/
│   ├── train.py
│   ├── trainer.py
│   ├── loss.py
│   └── scheduler.py
│
├── evaluate/
│   ├── evaluate.py
│   ├── generalization_test.py
│   └── metrics.py
│
├── inference/
│   ├── denoise.py
│   └── realtime.py
│
├── experiments/
│   ├── logs/
│   └── checkpoints/
│
├── results/
│   ├── images/
│   └── tables/
│
└── scripts/
    ├── prepare_dataset.py
    └── add_noise.py
