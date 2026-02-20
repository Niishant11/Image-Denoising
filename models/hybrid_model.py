import os
import sys
import torch
import torch.nn as nn

# Ensure project root is on sys.path for absolute imports
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.append(PROJECT_ROOT)

from models.cnn_blocks import DnCNN
from models.restormer_model import RestormerNet



class RestormerDenoiser(nn.Module):
    """
    Multi-scale Restormer denoiser head.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 48,
        num_blocks=(2, 2, 4, 4),
        num_heads=(1, 2, 4, 8),
        expansion_factor: float = 2.0,
        residual_learning: bool = True,
    ):
        super().__init__()
        self.net = RestormerNet(
            in_channels=in_channels,
            base_channels=base_channels,
            num_blocks=num_blocks,
            num_heads=num_heads,
            expansion_factor=expansion_factor,
            residual_learning=residual_learning,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HybridDenoiser(nn.Module):
    """
    Hybrid DnCNN + Restormer denoising model.

    - DnCNN branch: strong local denoising
    - Restormer branch: global context restoration
    - Fusion: combine two denoised outputs to refine final result
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        num_cnn_blocks: int = 17,
        num_transformer_blocks=(2, 2, 4, 4),
        num_heads=(1, 2, 4, 8),
        expansion_factor: float = 2.0,
        fusion_type: str = "concat",  # "concat", "add", "avg", or "gate"
        residual_learning: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.base_channels = base_channels
        self.residual_learning = residual_learning

        # DnCNN branch
        self.dncnn = DnCNN(
            in_channels=in_channels,
            out_channels=in_channels,
            num_features=base_channels,
            num_layers=num_cnn_blocks,
            residual_learning=True,
        )

        # Restormer branch
        self.restormer = RestormerDenoiser(
            in_channels=in_channels,
            base_channels=base_channels,
            num_blocks=num_transformer_blocks,
            num_heads=num_heads,
            expansion_factor=expansion_factor,
            residual_learning=True,
        )

        self.fusion_type = fusion_type
        if fusion_type == "concat":
            self.fuse_conv = nn.Sequential(
                nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=True),
            )
        elif fusion_type == "add":
            self.fuse_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=True)
        elif fusion_type == "gate":
            self.gate_conv = nn.Sequential(
                nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, 2, kernel_size=1, bias=True),
                nn.Sigmoid(),
            )
            self.fuse_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=True)
        elif fusion_type == "avg":
            self.fuse_conv = nn.Identity()
        else:
            raise ValueError(f"Unsupported fusion_type: {fusion_type}")

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
        dncnn_out = self.dncnn(x)
        rest_out = self.restormer(x)

        if self.fusion_type == "concat":
            fused = self.fuse_conv(torch.cat([dncnn_out, rest_out], dim=1))
        elif self.fusion_type == "add":
            fused = self.fuse_conv(dncnn_out + rest_out)
        elif self.fusion_type == "gate":
            gates = self.gate_conv(torch.cat([dncnn_out, rest_out], dim=1))
            gate_dn = gates[:, 0:1, :, :].expand_as(dncnn_out)
            gate_rt = gates[:, 1:2, :, :].expand_as(rest_out)
            fused = self.fuse_conv(gate_dn * dncnn_out + gate_rt * rest_out)
        elif self.fusion_type == "avg":
            fused = (dncnn_out + rest_out) / 2.0
        else:
            raise ValueError(f"Unsupported fusion_type: {self.fusion_type}")

        if self.residual_learning:
            return x - fused
        return fused


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
