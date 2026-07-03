import torch
import torch.nn as nn
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    """Charbonnier loss: sqrt((pred-target)^2 + eps^2). More robust than L1 to outliers."""

    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps2 = eps ** 2

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.eps2))


class SSIMLoss(nn.Module):
    """
    Differentiable SSIM loss computed via convolution.
    Returns 1 - SSIM so it can be minimized.
    """

    def __init__(self, window_size: int = 11, C1: float = 0.01**2, C2: float = 0.03**2):
        super().__init__()
        self.C1 = C1
        self.C2 = C2
        self.window_size = window_size
        # 1D Gaussian kernel
        sigma = 1.5
        gauss = torch.exp(-torch.arange(window_size, dtype=torch.float32).sub(window_size // 2).pow(2) / (2 * sigma ** 2))
        gauss = gauss / gauss.sum()
        # 2D kernel via outer product
        kernel_2d = gauss.unsqueeze(1) * gauss.unsqueeze(0)
        # [1, 1, ws, ws] — will be expanded per channel at runtime
        self.register_buffer("kernel", kernel_2d.unsqueeze(0).unsqueeze(0))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        C = pred.shape[1]
        # Expand kernel to match channels: [C, 1, ws, ws]
        kernel = self.kernel.expand(C, -1, -1, -1)
        pad = self.window_size // 2

        mu_pred = F.conv2d(pred, kernel, padding=pad, groups=C)
        mu_target = F.conv2d(target, kernel, padding=pad, groups=C)

        mu_pred_sq = mu_pred * mu_pred
        mu_target_sq = mu_target * mu_target
        mu_cross = mu_pred * mu_target

        sigma_pred_sq = F.conv2d(pred * pred, kernel, padding=pad, groups=C) - mu_pred_sq
        sigma_target_sq = F.conv2d(target * target, kernel, padding=pad, groups=C) - mu_target_sq
        sigma_cross = F.conv2d(pred * target, kernel, padding=pad, groups=C) - mu_cross

        ssim_map = ((2 * mu_cross + self.C1) * (2 * sigma_cross + self.C2)) / \
                   ((mu_pred_sq + mu_target_sq + self.C1) * (sigma_pred_sq + sigma_target_sq + self.C2))

        return 1.0 - ssim_map.mean()


class DenoisingLoss(nn.Module):
    """
    Flexible loss for image denoising.

    Supported loss_type values:
        - "mse"               : Mean Squared Error
        - "l1"                : L1 / MAE
        - "charbonnier"       : Charbonnier loss (smooth L1)
        - "charbonnier_ssim"  : Charbonnier + 0.1 * SSIM loss (recommended)
    """

    def __init__(self, loss_type: str = "charbonnier_ssim"):
        super().__init__()

        loss_type = loss_type.lower()
        self.loss_type = loss_type

        if loss_type == "mse":
            self.loss_fn = nn.MSELoss()
        elif loss_type == "l1":
            self.loss_fn = nn.L1Loss()
        elif loss_type == "charbonnier":
            self.loss_fn = CharbonnierLoss()
        elif loss_type == "charbonnier_ssim":
            self.charb = CharbonnierLoss()
            self.ssim = SSIMLoss()
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type}")

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "charbonnier_ssim":
            return self.charb(pred, target) + 0.1 * self.ssim(pred, target)
        return self.loss_fn(pred, target)


if __name__ == "__main__":
    # Quick self-test
    x = torch.randn(4, 3, 64, 64)
    y = torch.randn(4, 3, 64, 64)

    criterion = DenoisingLoss(loss_type="mse")
    loss = criterion(x, y)
    print("Test loss:", loss.item())
