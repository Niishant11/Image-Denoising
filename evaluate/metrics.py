import math
import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim_fn


def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """
    Compute PSNR between two batches of images.
    Assumes pred and target are in [0, 1].
    """
    mse = torch.mean((pred - target) ** 2).item()
    if mse <= 1e-10:
        return 100.0
    return 10.0 * math.log10((max_val ** 2) / mse)


def ssim_batch(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """
    Compute average SSIM over a batch.
    Inputs: pred, target in [0,1], shape [B, C, H, W], C=1 or 3.
    """
    pred = pred.detach().cpu().clamp(0.0, 1.0)
    target = target.detach().cpu().clamp(0.0, 1.0)

    b, c, h, w = pred.shape
    total_ssim = 0.0

    # Loop over batch
    for i in range(b):
        p = pred[i]
        t = target[i]

        # [C, H, W] -> [H, W, C]
        p_np = p.permute(1, 2, 0).numpy()
        t_np = t.permute(1, 2, 0).numpy()

        # Convert to grayscale for SSIM if 3-channel
        if c == 3:
            p_gray = 0.299 * p_np[:, :, 0] + 0.587 * p_np[:, :, 1] + 0.114 * p_np[:, :, 2]
            t_gray = 0.299 * t_np[:, :, 0] + 0.587 * t_np[:, :, 1] + 0.114 * t_np[:, :, 2]
        else:
            p_gray = p_np.squeeze()
            t_gray = t_np.squeeze()

        ssim_val = ssim_fn(
            t_gray,
            p_gray,
            data_range=max_val,
            gaussian_weights=True,
            multichannel=False,
        )
        total_ssim += ssim_val

    return total_ssim / max(b, 1)


if __name__ == "__main__":
    # Quick sanity test
    x = torch.rand(2, 3, 64, 64)
    y = x.clone() + 0.01 * torch.randn_like(x)

    print("PSNR:", psnr(y, x))
    print("SSIM:", ssim_batch(y, x))
