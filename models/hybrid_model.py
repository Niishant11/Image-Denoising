import os
import sys
import torch
import torch.nn as nn

# Ensure project root is on sys.path for absolute imports
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.append(PROJECT_ROOT)

from models.cnn_blocks import CNNFeatureExtractor, DnCNN
from models.attention_blocks import TransformerFeatureExtractor
from models.fusion_layer import FeatureFusion



class HybridDenoiser(nn.Module):
    """
    Hybrid CNN–Transformer image denoising model.

    - CNN branch: local feature extraction (DnCNN-style residual CNN blocks)
    - Transformer branch: global context modeling (self-attention blocks)
    - Fusion: combines CNN and Transformer features
    - Output head: predicts residual noise or clean image

    By default:
      - Input:  noisy image  [B, 3, H, W]
      - Output: denoised image [B, 3, H, W]
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        num_cnn_blocks: int = 5,
        num_transformer_blocks: int = 4,
        fusion_type: str = "concat",  # "concat", "add", or "gate"
        residual_learning: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.base_channels = base_channels
        self.residual_learning = residual_learning

        # Shallow feature extraction from input
        self.shallow_feat = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )

        # CNN branch (local textures)
        self.cnn_branch = CNNFeatureExtractor(
            in_channels=base_channels,
            num_features=base_channels,
            num_blocks=num_cnn_blocks,
        )

        # Transformer branch (global context)
        self.transformer_branch = TransformerFeatureExtractor(
            in_channels=base_channels,
            num_blocks=num_transformer_blocks,
            num_heads=4,
            expansion_factor=2.0,
        )

        # Fusion of CNN and Transformer features
        self.fusion = FeatureFusion(
            channels=base_channels,
            fusion_type=fusion_type,
        )

        # Output head: map fused features back to image space
        self.output_head = nn.Conv2d(
            base_channels, in_channels, kernel_size=3, padding=1, bias=True
        )

        # If desired, you can also keep an optional refinement DnCNN:
        # self.refine_dncnn = DnCNN(
        #     in_channels=in_channels,
        #     out_channels=in_channels,
        #     num_features=base_channels,
        #     num_layers=10,
        #     residual_learning=True,
        # )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] noisy input image

        Returns:
            denoised image [B, C, H, W]
        """
        # Save noisy input if using residual learning
        noisy_input = x

        # Shallow feature extraction
        feat0 = self.shallow_feat(x)  # [B, C, H, W]

        # CNN branch
        cnn_feats = self.cnn_branch(feat0)  # [B, C, H, W]

        # Transformer branch
        tr_feats = self.transformer_branch(feat0)  # [B, C, H, W]

        # Fuse local + global features
        fused_feats = self.fusion(cnn_feats, tr_feats)  # [B, C, H, W]

        # Predict residual noise or clean image
        pred = self.output_head(fused_feats)  # [B, in_channels, H, W]

        if self.residual_learning:
            # Model predicts noise -> denoised = input - noise
            out = noisy_input - pred
        else:
            # Model predicts clean image directly
            out = pred

        # Optional DN-CNN refinement (if enabled)
        # out = self.refine_dncnn(out)

        return out


if __name__ == "__main__":
    # Quick sanity test
    model = HybridDenoiser(
        in_channels=3,
        base_channels=64,
        num_cnn_blocks=5,
        num_transformer_blocks=4,
        fusion_type="concat",
        residual_learning=True,
    )

    x = torch.randn(1, 3, 128, 128)
    y = model(x)

    print("Input shape: ", x.shape)
    print("Output shape:", y.shape)
