import torch
import torch.nn as nn


class DenoisingLoss(nn.Module):
    """
    Basic loss for image denoising.

    By default, uses Mean Squared Error (MSE) between
    predicted clean image and ground-truth clean image.
    """

    def __init__(self, loss_type: str = "mse"):
        super().__init__()

        loss_type = loss_type.lower()
        if loss_type == "mse":
            self.loss_fn = nn.MSELoss()
        elif loss_type == "l1":
            self.loss_fn = nn.L1Loss()
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type}")

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred:   [B, C, H, W] predicted clean image
            target: [B, C, H, W] ground-truth clean image

        Returns:
            scalar loss
        """
        return self.loss_fn(pred, target)


if __name__ == "__main__":
    # Quick self-test
    x = torch.randn(4, 3, 64, 64)
    y = torch.randn(4, 3, 64, 64)

    criterion = DenoisingLoss(loss_type="mse")
    loss = criterion(x, y)
    print("Test loss:", loss.item())
