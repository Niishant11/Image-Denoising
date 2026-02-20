import torch
import torch.nn as nn

from models.attention_blocks import LayerNorm2d, MDTA, GDFN


class RestormerBlock(nn.Module):
    """
    Restormer block with MDTA + GDFN.
    """

    def __init__(self, dim: int, num_heads: int, expansion_factor: float = 2.0, bias: bool = True):
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        self.attn = MDTA(dim=dim, num_heads=num_heads, bias=bias)
        self.norm2 = LayerNorm2d(dim)
        self.ffn = GDFN(dim=dim, expansion_factor=expansion_factor, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class Downsample(nn.Module):
    """
    Downsample by factor 2 using strided convolution.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.body = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class Upsample(nn.Module):
    """
    Upsample by factor 2 using convolution + pixel shuffle.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * 4, kernel_size=3, padding=1, bias=True),
            nn.PixelShuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class RestormerNet(nn.Module):
    """
    Multi-scale Restormer-style encoder-decoder.
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
        self.residual_learning = residual_learning

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8

        self.shallow = nn.Conv2d(in_channels, c1, kernel_size=3, padding=1, bias=True)

        self.enc1 = nn.Sequential(*[RestormerBlock(c1, num_heads[0], expansion_factor) for _ in range(num_blocks[0])])
        self.down1 = Downsample(c1, c2)

        self.enc2 = nn.Sequential(*[RestormerBlock(c2, num_heads[1], expansion_factor) for _ in range(num_blocks[1])])
        self.down2 = Downsample(c2, c3)

        self.enc3 = nn.Sequential(*[RestormerBlock(c3, num_heads[2], expansion_factor) for _ in range(num_blocks[2])])
        self.down3 = Downsample(c3, c4)

        self.bottleneck = nn.Sequential(*[RestormerBlock(c4, num_heads[3], expansion_factor) for _ in range(num_blocks[3])])

        self.up3 = Upsample(c4, c3)
        self.dec3 = nn.Sequential(*[RestormerBlock(c3, num_heads[2], expansion_factor) for _ in range(num_blocks[2])])

        self.up2 = Upsample(c3, c2)
        self.dec2 = nn.Sequential(*[RestormerBlock(c2, num_heads[1], expansion_factor) for _ in range(num_blocks[1])])

        self.up1 = Upsample(c2, c1)
        self.dec1 = nn.Sequential(*[RestormerBlock(c1, num_heads[0], expansion_factor) for _ in range(num_blocks[0])])

        self.out_head = nn.Conv2d(c1, in_channels, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.shallow(x)

        e1 = self.enc1(x0)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        b = self.bottleneck(self.down3(e3))

        d3 = self.up3(b) + e3
        d3 = self.dec3(d3)
        d2 = self.up2(d3) + e2
        d2 = self.dec2(d2)
        d1 = self.up1(d2) + e1
        d1 = self.dec1(d1)

        pred = self.out_head(d1)
        if self.residual_learning:
            return x - pred
        return pred
