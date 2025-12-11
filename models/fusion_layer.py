import torch
import torch.nn as nn
from typing import Literal


class FeatureFusion(nn.Module):
    """
    Feature fusion module to combine CNN and Transformer feature maps.

    Input:
        cnn_feats:  [B, C, H, W]
        tr_feats:   [B, C, H, W]

    Output:
        fused_feats: [B, C, H, W]

    Supported fusion types:
        - "concat": concatenate along channel dim and reduce with Conv
        - "add": element-wise sum after 1x1 alignment (if needed)
        - "gate": learn adaptive weights for each branch
    """

    def __init__(
        self,
        channels: int,
        fusion_type: Literal["concat", "add", "gate"] = "concat",
    ):
        super().__init__()

        self.fusion_type = fusion_type
        self.channels = channels

        if fusion_type == "concat":
            # Concatenate along channels -> 2C, then reduce back to C
            self.fuse_conv = nn.Sequential(
                nn.Conv2d(2 * channels, channels, kernel_size=1, bias=True),
                nn.ReLU(inplace=True),
            )
        elif fusion_type == "add":
            # Optionally a small conv to refine after addition
            self.refine_conv = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True),
            )
        elif fusion_type == "gate":
            # Learnable gating between CNN and Transformer features
            #  - Take concatenation of stats or channels
            #  - Produce 2 weights (per-channel or scalar) via Sigmoid
            self.gate_conv = nn.Sequential(
                nn.Conv2d(2 * channels, channels, kernel_size=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, 2, kernel_size=1, bias=True),
                nn.Sigmoid(),
            )
            # Optional refinement after fusion
            self.refine_conv = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True),
            )
        else:
            raise ValueError(f"Unsupported fusion_type: {fusion_type}")

    def forward(self, cnn_feats: torch.Tensor, tr_feats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cnn_feats: [B, C, H, W] from CNN branch
            tr_feats:  [B, C, H, W] from Transformer branch

        Returns:
            fused_feats: [B, C, H, W]
        """
        if self.fusion_type == "concat":
            # [B, 2C, H, W] -> [B, C, H, W]
            x = torch.cat([cnn_feats, tr_feats], dim=1)
            out = self.fuse_conv(x)
            return out

        elif self.fusion_type == "add":
            # simple element-wise addition + refinement
            x = cnn_feats + tr_feats
            out = self.refine_conv(x)
            return out

        elif self.fusion_type == "gate":
            # Gated fusion
            # Compute weights from concatenated features
            x_cat = torch.cat([cnn_feats, tr_feats], dim=1)  # [B, 2C, H, W]

            gates = self.gate_conv(x_cat)  # [B, 2, H, W], values in [0,1]
            gate_cnn = gates[:, 0:1, :, :]  # [B, 1, H, W]
            gate_tr = gates[:, 1:2, :, :]   # [B, 1, H, W]

            # Broadcast to channel dimension
            gate_cnn = gate_cnn.expand_as(cnn_feats)
            gate_tr = gate_tr.expand_as(tr_feats)

            fused = gate_cnn * cnn_feats + gate_tr * tr_feats
            out = self.refine_conv(fused)
            return out

        else:
            raise ValueError(f"Unsupported fusion_type: {self.fusion_type}")


if __name__ == "__main__":
    # Quick self-test
    B, C, H, W = 2, 64, 64, 64
    cnn_feats = torch.randn(B, C, H, W)
    tr_feats = torch.randn(B, C, H, W)

    print("Testing FeatureFusion with 'concat'...")
    fusion_concat = FeatureFusion(channels=C, fusion_type="concat")
    out1 = fusion_concat(cnn_feats, tr_feats)
    print("Output shape (concat):", out1.shape)

    print("Testing FeatureFusion with 'add'...")
    fusion_add = FeatureFusion(channels=C, fusion_type="add")
    out2 = fusion_add(cnn_feats, tr_feats)
    print("Output shape (add):", out2.shape)

    print("Testing FeatureFusion with 'gate'...")
    fusion_gate = FeatureFusion(channels=C, fusion_type="gate")
    out3 = fusion_gate(cnn_feats, tr_feats)
    print("Output shape (gate):", out3.shape)
