import os
import sys
import argparse
import torch
import yaml
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.utils import save_image

# project root
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.append(PROJECT_ROOT)

from models.hybrid_model import HybridDenoiser


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_image(img_path, resize=None):
    img = Image.open(img_path).convert("RGB")

    if resize is not None:
        img = img.resize((resize, resize))

    transform = T.ToTensor()
    return transform(img).unsqueeze(0)   # [1, C, H, W]


def save_output(tensor, out_path):
    tensor = tensor.squeeze(0).clamp(0.0, 1.0)
    save_image(tensor, out_path)


@torch.no_grad()
def infer_image(model, img_tensor, device, window_size: int = 8):
    """
    Run inference with padding so H and W are divisible by window_size.
    Args:
        model: denoising model
        img_tensor: [1, C, H, W] in [0,1]
        device: torch.device
        window_size: window size used in Transformer (default=8)
    """
    model.eval()
    img_tensor = img_tensor.to(device)

    _, _, h, w = img_tensor.shape

    # Compute padding so H and W become multiples of window_size
    pad_h = (window_size - h % window_size) % window_size
    pad_w = (window_size - w % window_size) % window_size

    if pad_h != 0 or pad_w != 0:
        # Pad on bottom and right: (left, right, top, bottom)
        img_padded = F.pad(img_tensor, (0, pad_w, 0, pad_h), mode="reflect")
    else:
        img_padded = img_tensor

    # Denoise
    output_padded = model(img_padded)

    # Remove padding to get back original size
    output = output_padded[:, :, :h, :w]

    return output



def main():
    parser = argparse.ArgumentParser(description="Hybrid CNN Transformer Image Denoising Inference")
    parser.add_argument("--image", type=str, required=True, help="Path to input noisy image")
    parser.add_argument("--checkpoint", type=str, default="experiments/checkpoints/hybrid_best.pth")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--out", type=str, default="results/inference.png")
    parser.add_argument("--resize", type=int, default=None, help="Resize image to square size (optional)")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Device
    use_gpu = cfg.get("device", {}).get("use_gpu", True)
    gpu_id = cfg.get("device", {}).get("gpu_id", 0)

    if use_gpu and torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # Load model config
    model_cfg = cfg.get("model", {})
    model = HybridDenoiser(
        in_channels=3,
        base_channels=model_cfg.get("base_channels", 48),
        num_cnn_blocks=model_cfg.get("num_cnn_blocks", 17),
        num_transformer_blocks=model_cfg.get("num_transformer_blocks", [2, 2, 4, 4]),
        num_heads=model_cfg.get("num_heads", [1, 2, 4, 8]),
        expansion_factor=model_cfg.get("expansion_factor", 2.0),
        fusion_type=model_cfg.get("fusion_type", "concat"),
        residual_learning=model_cfg.get("use_residual_learning", True),
    ).to(device)

    # Load weights
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print("Model loaded successfully!")

    # Load image
    img = load_image(args.image, resize=args.resize)

    # Denoise
    output = infer_image(model, img, device, window_size=8)


    # Save result
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    save_output(output, args.out)

    print(f"Denoised image saved at: {args.out}")


if __name__ == "__main__":
    main()



# python inference\denoise.py --image "path_to_noisy_image.png" --out results\my_denoised.png
# Ntest\Noised-image.jpg
# python inference\denoise.py --image "Ntest/text images.jpg" --out results\my_denoised2.jpg
# python inference\denoise.py --image "Ntest/images.jpg" --out results\my_denoised1111.jpg
