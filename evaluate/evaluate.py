import os
import sys
import argparse

import torch
import yaml
from torchvision.utils import save_image

# Add project root to sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.append(PROJECT_ROOT)

from models.hybrid_model import HybridDenoiser
from datasets.dataset_loader import get_bsd500_dataloader
from metrics import psnr, ssim_batch


def load_config(config_path: str = "config.yaml"):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def set_seed(seed: int = 42):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate_model(
    model,
    dataloader,
    device,
    save_images: bool = False,
    save_dir: str = "results/images",
    max_batches: int = None,
):
    model.eval()
    total_psnr = 0.0
    total_ssim = 0.0
    count = 0

    os.makedirs(save_dir, exist_ok=True)

    for batch_idx, (noisy, clean) in enumerate(dataloader):
        noisy = noisy.to(device)
        clean = clean.to(device)

        output = model(noisy)
        output_clamped = output.clamp(0.0, 1.0)
        clean_clamped = clean.clamp(0.0, 1.0)

        batch_psnr = psnr(output_clamped, clean_clamped, max_val=1.0)
        batch_ssim = ssim_batch(output_clamped, clean_clamped, max_val=1.0)

        total_psnr += batch_psnr
        total_ssim += batch_ssim
        count += 1

        if save_images and batch_idx < 5:  # save first few batches
            # Save first image from batch
            out_img = output_clamped[0].detach().cpu()
            noisy_img = noisy[0].detach().cpu().clamp(0.0, 1.0)
            clean_img = clean[0].detach().cpu().clamp(0.0, 1.0)

            save_image(noisy_img, os.path.join(save_dir, f"noisy_{batch_idx}.png"))
            save_image(out_img, os.path.join(save_dir, f"denoised_{batch_idx}.png"))
            save_image(clean_img, os.path.join(save_dir, f"clean_{batch_idx}.png"))

        if max_batches is not None and (batch_idx + 1) >= max_batches:
            break

    avg_psnr = total_psnr / max(count, 1)
    avg_ssim = total_ssim / max(count, 1)

    return avg_psnr, avg_ssim


def main():
    parser = argparse.ArgumentParser(description="Evaluate Hybrid Denoiser on BSD500 test set")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="experiments/checkpoints/hybrid_best.pth",
        help="Path to trained model checkpoint.",
    )
    parser.add_argument(
        "--save_images",
        action="store_true",
        help="Save some denoised images for visual inspection.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Device
    use_gpu = cfg.get("device", {}).get("use_gpu", True)
    gpu_id = cfg.get("device", {}).get("gpu_id", 0)
    seed = cfg.get("device", {}).get("seed", 42)
    set_seed(seed)

    if use_gpu and torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # Dataset / dataloader config
    dataset_cfg = cfg.get("dataset", {})
    patch_size = dataset_cfg.get("patch_size", 128)

    batch_size = cfg["training"].get("batch_size", 16)
    num_workers = cfg["training"].get("num_workers", 4)

    test_loader = get_bsd500_dataloader(
        root_dir=".",
        split="test",
        noise_type="gaussian",  # or "nongaussian"/"mixed" depending on setup
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        patch_size=patch_size,
    )

    # Model config
    model_cfg = cfg.get("model", {})
    in_channels = 3
    base_channels = model_cfg.get("base_channels", 64)
    num_cnn_blocks = model_cfg.get("num_cnn_blocks", 5)
    num_transformer_blocks = model_cfg.get("num_transformer_blocks", 4)
    fusion_type = model_cfg.get("fusion_type", "concat")
    residual_learning = model_cfg.get("use_residual_learning", True)

    model = HybridDenoiser(
        in_channels=in_channels,
        base_channels=base_channels,
        num_cnn_blocks=num_cnn_blocks,
        num_transformer_blocks=num_transformer_blocks,
        fusion_type=fusion_type,
        residual_learning=residual_learning,
    ).to(device)

    # Load checkpoint
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded model from epoch {ckpt.get('epoch', 'N/A')} with val PSNR={ckpt.get('val_psnr', 'N/A')}")

    # Evaluate
    results_dir = cfg["evaluation"].get("results_dir", "results/images")
    avg_psnr, avg_ssim = evaluate_model(
        model,
        test_loader,
        device,
        save_images=args.save_images,
        save_dir=results_dir,
    )

    print(f"Test set results - PSNR: {avg_psnr:.2f} dB, SSIM: {avg_ssim:.4f}")


if __name__ == "__main__":
    main()
