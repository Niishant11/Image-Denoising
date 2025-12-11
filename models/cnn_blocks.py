import torch
import torch.nn as nn
from typing import Tuple


class ResidualBlock(nn.Module):
    """
    Basic residual block:
    Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN + skip connection -> ReLU
    Used as a building block for deeper CNN feature extractors.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.relu(out)
        return out


class DnCNN(nn.Module):
    """
    DnCNN-style network for image denoising.

    Reference idea:
      - First layer: Conv + ReLU
      - Middle layers: Conv + BN + ReLU
      - Last layer: Conv
      - Residual learning: model predicts noise, output = input - noise

    This can be used:
      - As a standalone denoiser
      - As the CNN branch inside the hybrid CNN–Transformer model
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        num_features: int = 64,
        num_layers: int = 17,
        residual_learning: bool = True,
    ):
        """
        Args:
            in_channels: Number of input channels (3 for RGB, 1 for grayscale).
            out_channels: Number of output channels (usually same as input).
            num_features: Number of feature maps in hidden layers.
            num_layers: Total number of conv layers (>= 3).
            residual_learning: If True, network predicts noise and returns x - noise.
        """
        super().__init__()

        assert num_layers >= 3, "DnCNN requires at least 3 layers."

        self.residual_learning = residual_learning

        layers = []

        # First layer: Conv + ReLU (no BN)
        layers.append(
            nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1, bias=True)
        )
        layers.append(nn.ReLU(inplace=True))

        # Middle layers: (num_layers - 2) blocks of Conv + BN + ReLU
        for _ in range(num_layers - 2):
            layers.append(
                nn.Conv2d(
                    num_features,
                    num_features,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                )
            )
            layers.append(nn.BatchNorm2d(num_features))
            layers.append(nn.ReLU(inplace=True))

        # Last layer: Conv (maps to out_channels)
        layers.append(
            nn.Conv2d(num_features, out_channels, kernel_size=3, padding=1, bias=True)
        )

        self.dncnn = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        If residual_learning=True:
           output = x - noise_pred
        else:
           output = direct prediction from network.
        """
        noise_pred = self.dncnn(x)
        if self.residual_learning:
            return x - noise_pred
        return noise_pred


class CNNFeatureExtractor(nn.Module):
    """
    Lightweight CNN feature extractor to be used as the 'CNN branch'
    in the hybrid CNN–Transformer model.

    It shares the spirit of DnCNN (stack of conv layers), but
    instead of predicting noise, it returns feature maps.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_features: int = 64,
        num_blocks: int = 5,
    ):
        """
        Args:
            in_channels: Input channels (e.g., 3 for RGB).
            num_features: Base channel width.
            num_blocks: Number of residual blocks.
        """
        super().__init__()

        # Initial conv to lift to feature space
        self.entry = nn.Sequential(
            nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )

        # Residual blocks
        blocks = []
        for _ in range(num_blocks):
            blocks.append(ResidualBlock(num_features))
        self.blocks = nn.Sequential(*blocks)

        # Optional exit conv if needed later
        self.exit = nn.Identity()  # can be replaced with Conv if needed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            Feature maps of shape [B, C, H, W] that encode local textures.
        """
        x = self.entry(x)
        x = self.blocks(x)
        x = self.exit(x)
        return x


if __name__ == "__main__":
    # Quick self-test
    inp = torch.randn(1, 3, 128, 128)

    print("Testing DnCNN...")
    dncnn = DnCNN(in_channels=3, out_channels=3, num_features=64, num_layers=17)
    out = dncnn(inp)
    print("DnCNN output shape:", out.shape)

    print("Testing CNNFeatureExtractor...")
    feat_extractor = CNNFeatureExtractor(in_channels=3, num_features=64, num_blocks=5)
    feats = feat_extractor(inp)
    print("FeatureExtractor output shape:", feats.shape)
