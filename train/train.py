import os
import sys
import argparse
import time
import math


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import yaml

# Add project root to sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.append(PROJECT_ROOT)

from models.hybrid_model import HybridDenoiser
from datasets.dataset_loader import get_bsd500_dataloader
from loss import DenoisingLoss  # ⬅ Option A (package style)



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


def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """
    Compute PSNR for a batch of images. Assumes inputs in [0,1].
    """
    mse = torch.mean((pred - target) ** 2).item()
    if mse <= 1e-10:
        return 100.0
    return 10.0 * math.log10((max_val ** 2) / mse)



def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    epoch: int,
    log_interval: int = 50,
    writer=None,
):
    model.train()
    running_loss = 0.0

    for batch_idx, (noisy, clean) in enumerate(dataloader):
        noisy = noisy.to(device)
        clean = clean.to(device)

        optimizer.zero_grad()
        output = model(noisy)
        loss = criterion(output, clean)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (batch_idx + 1) % log_interval == 0:
            avg_loss = running_loss / log_interval
            print(
                f"[Epoch {epoch}] Batch {batch_idx+1}/{len(dataloader)} - "
                f"Loss: {avg_loss:.6f}"
            )
            if writer is not None:
                global_step = (epoch - 1) * len(dataloader) + batch_idx + 1
                writer.add_scalar("Train/Loss", avg_loss, global_step)
            running_loss = 0.0


@torch.no_grad()
def validate(model, dataloader, criterion, device, epoch: int, writer=None):
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    count = 0

    for noisy, clean in dataloader:
        noisy = noisy.to(device)
        clean = clean.to(device)

        output = model(noisy)
        loss = criterion(output, clean)
        total_loss += loss.item()

        # assuming images are in [0,1]
        batch_psnr = psnr(output.clamp(0.0, 1.0), clean.clamp(0.0, 1.0), max_val=1.0)
        total_psnr += batch_psnr
        count += 1

    avg_loss = total_loss / max(count, 1)
    avg_psnr = total_psnr / max(count, 1)

    print(f"Validation - Epoch {epoch}: Loss={avg_loss:.6f}, PSNR={avg_psnr:.2f}dB")

    if writer is not None:
        writer.add_scalar("Val/Loss", avg_loss, epoch)
        writer.add_scalar("Val/PSNR", avg_psnr, epoch)

    return avg_loss, avg_psnr


def main():
    parser = argparse.ArgumentParser(description="Hybrid CNN–Transformer Denoising Training")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file (YAML).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Device setup
    use_gpu = cfg.get("device", {}).get("use_gpu", True)
    gpu_id = cfg.get("device", {}).get("gpu_id", 0)
    seed = cfg.get("device", {}).get("seed", 42)

    set_seed(seed)

    if use_gpu and torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # Paths
    save_dir = cfg["training"].get("save_checkpoint_dir", "experiments/checkpoints")
    log_dir = cfg["training"].get("log_dir", "experiments/logs")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=log_dir)

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

    print(model)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # Loss
    criterion = DenoisingLoss(loss_type="mse").to(device)

    # Optimizer & scheduler
    train_cfg = cfg["training"]
    lr = train_cfg.get("lr", 1e-3)
    weight_decay = train_cfg.get("weight_decay", 0.0)
    optimizer_name = train_cfg.get("optimizer", "adam").lower()

    if optimizer_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    lr_step = train_cfg.get("lr_step", 20)
    lr_gamma = train_cfg.get("lr_gamma", 0.1)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=lr_gamma)

    # Dataloaders
    batch_size = train_cfg.get("batch_size", 16)
    num_workers = train_cfg.get("num_workers", 4)
    dataset_cfg = cfg.get("dataset", {})
    patch_size = dataset_cfg.get("patch_size", 128)  # 🔹 from config.yaml

    train_loader = get_bsd500_dataloader(
        root_dir=".",
        split="train",
        noise_type="gaussian",  # change if needed
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        patch_size=patch_size,
    )

    val_loader = get_bsd500_dataloader(
        root_dir=".",
        split="val",
        noise_type="gaussian",  # should match training noise or test variant
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        patch_size=patch_size,
    )


    # Training loop
    best_val_psnr = 0.0
    num_epochs = train_cfg.get("epochs", 50)

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch=epoch,
            log_interval=50,
            writer=writer,
        )

        val_loss, val_psnr = validate(
            model,
            val_loader,
            criterion,
            device,
            epoch=epoch,
            writer=writer,
        )

        scheduler.step()

        # Save best model
        if val_psnr > best_val_psnr:
            best_val_psnr = val_psnr
            best_path = os.path.join(save_dir, "hybrid_best.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_psnr": val_psnr,
                },
                best_path,
            )
            print(f"New best model saved at epoch {epoch} with PSNR={val_psnr:.2f}dB")

        elapsed = time.time() - start_time
        print(f"Epoch {epoch} completed in {elapsed:.2f} seconds\n")

    writer.close()
    print("Training completed.")


if __name__ == "__main__":
    main()
