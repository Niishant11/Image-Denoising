import torch
import torch.nn as nn
from einops import rearrange
from typing import Optional


class LayerNorm2d(nn.Module):
    """
    LayerNorm applied over the channel dimension for 2D feature maps.

    Input shape: [B, C, H, W]
    We apply LayerNorm over C, treating (H, W) as spatial dims.
    """

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.ln = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] -> [B, H, W, C]
        x_perm = x.permute(0, 2, 3, 1)
        x_norm = self.ln(x_perm)
        # back to [B, C, H, W]
        x_norm = x_norm.permute(0, 3, 1, 2)
        return x_norm


class WindowSelfAttention(nn.Module):
    """
    Window-based multi-head self-attention for image features.

    Instead of computing attention over all HW tokens,
    we split the feature map into non-overlapping windows
    of size (window_size x window_size) and compute attention
    inside each window only.

    Input:  [B, C, H, W]
    Output: [B, C, H, W]

    H and W must be divisible by window_size.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        window_size: int = 8,
        bias: bool = True,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.window_size = window_size

        # 1x1 Conv for QKV
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        # 1x1 Conv for output projection
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W]
        """
        B, C, H, W = x.shape
        ws = self.window_size

        assert H % ws == 0 and W % ws == 0, (
            f"H and W must be divisible by window_size={ws}, "
            f"got H={H}, W={W}"
        )

        qkv = self.qkv(x)  # [B, 3C, H, W]
        q, k, v = torch.chunk(qkv, chunks=3, dim=1)  # each: [B, C, H, W]

        # Split into windows:
        # [B, C, H, W] -> [B, C, Gh, ws, Gw, ws]
        Gh = H // ws
        Gw = W // ws

        # Rearrange to [B, head, num_windows, tokens, dim_per_head]
        # tokens = ws * ws
        q = rearrange(
            q,
            "b (h d) (gh ws1) (gw ws2) -> b h (gh gw) (ws1 ws2) d",
            h=self.num_heads,
            gh=Gh,
            gw=Gw,
            ws1=ws,
            ws2=ws,
        )
        k = rearrange(
            k,
            "b (h d) (gh ws1) (gw ws2) -> b h (gh gw) (ws1 ws2) d",
            h=self.num_heads,
            gh=Gh,
            gw=Gw,
            ws1=ws,
            ws2=ws,
        )
        v = rearrange(
            v,
            "b (h d) (gh ws1) (gw ws2) -> b h (gh gw) (ws1 ws2) d",
            h=self.num_heads,
            gh=Gh,
            gw=Gw,
            ws1=ws,
            ws2=ws,
        )
        # shapes: [B, H, Nw, T, D] where Nw = number of windows, T = ws*ws

        # Compute attention within each window:
        # attn: [B, H, Nw, T, T]
        attn = torch.einsum("b h n i d, b h n j d -> b h n i j", q * self.scale, k)
        attn = attn.softmax(dim=-1)

        # Weighted sum:
        # out_win: [B, H, Nw, T, D]
        out_win = torch.einsum("b h n i j, b h n j d -> b h n i d", attn, v)

        # Merge windows back to [B, C, H, W]
        out = rearrange(
            out_win,
            "b h (gh gw) (ws1 ws2) d -> b (h d) (gh ws1) (gw ws2)",
            gh=Gh,
            gw=Gw,
            ws1=ws,
            ws2=ws,
        )

        out = self.proj(out)
        return out


class FeedForward(nn.Module):
    """
    Simple feed-forward network with expansion and depthwise conv for
    better spatial modeling (inspired by Restormer GDFN).

    Input / output: [B, C, H, W]
    """

    def __init__(self, dim: int, expansion_factor: float = 2.0, bias: bool = True):
        super().__init__()
        hidden_dim = int(dim * expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden_dim,
            hidden_dim,
            kernel_size=3,
            padding=1,
            groups=hidden_dim,  # depthwise
            bias=bias,
        )
        self.act = nn.GELU()
        self.project_out = nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project_in(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.project_out(x)
        return x


class TransformerBlock(nn.Module):
    """
    Transformer-style block for image restoration:
      - LayerNorm2d
      - Window-based SelfAttention
      - Residual connection
      - LayerNorm2d
      - FeedForward
      - Residual connection
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        expansion_factor: float = 2.0,
        window_size: int = 8,
        bias: bool = True,
    ):
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        self.attn = WindowSelfAttention(
            dim=dim, num_heads=num_heads, window_size=window_size, bias=bias
        )
        self.norm2 = LayerNorm2d(dim)
        self.ffn = FeedForward(dim=dim, expansion_factor=expansion_factor, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention branch
        x = x + self.attn(self.norm1(x))
        # Feed-forward branch
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerFeatureExtractor(nn.Module):
    """
    Transformer-based feature extractor to act as the 'Transformer branch'
    in the hybrid CNN–Transformer model.

    It processes feature maps and returns context-enriched representations.

    Input:  [B, C, H, W]
    Output: [B, C, H, W]
    """

    def __init__(
        self,
        in_channels: int = 64,
        num_blocks: int = 4,
        num_heads: int = 4,
        expansion_factor: float = 2.0,
        window_size: int = 8,
    ):
        super().__init__()

        self.entry = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=True)

        blocks = []
        for _ in range(num_blocks):
            blocks.append(
                TransformerBlock(
                    dim=in_channels,
                    num_heads=num_heads,
                    expansion_factor=expansion_factor,
                    window_size=window_size,
                )
            )
        self.blocks = nn.Sequential(*blocks)

        self.exit = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] feature maps (e.g., from CNN branch or stem).

        Returns:
            [B, C, H, W] context-enriched features.
        """
        x = self.entry(x)
        x = self.blocks(x)
        x = self.exit(x)
        return x


class MDTA(nn.Module):
    """
    Multi-Dconv Transpose Attention (MDTA) inspired block.
    Simplified to channel-wise attention over spatial tokens.
    """

    def __init__(self, dim: int, num_heads: int = 4, bias: bool = True):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, padding=1, groups=dim * 3, bias=bias)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.dwconv(self.qkv(x))
        q, k, v = torch.chunk(qkv, chunks=3, dim=1)

        q = q.view(b, self.num_heads, self.head_dim, h * w)
        k = k.view(b, self.num_heads, self.head_dim, h * w)
        v = v.view(b, self.num_heads, self.head_dim, h * w)

        q = q / (h * w) ** 0.5
        attn = torch.matmul(q, k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)

        out = torch.matmul(attn, v)
        out = out.view(b, c, h, w)
        out = self.proj(out)
        return out


class GDFN(nn.Module):
    """
    Gated-Dconv Feed-Forward Network (GDFN) inspired block.
    """

    def __init__(self, dim: int, expansion_factor: float = 2.0, bias: bool = True):
        super().__init__()
        hidden_dim = int(dim * expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_dim * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_dim * 2, hidden_dim * 2, kernel_size=3, padding=1, groups=hidden_dim * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project_in(x)
        x = self.dwconv(x)
        x1, x2 = torch.chunk(x, chunks=2, dim=1)
        x = torch.nn.functional.gelu(x1) * x2
        x = self.project_out(x)
        return x


class RestormerBlock(nn.Module):
    """
    Restormer-style block with MDTA + GDFN.
    """

    def __init__(self, dim: int, num_heads: int = 4, expansion_factor: float = 2.0, bias: bool = True):
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        self.attn = MDTA(dim=dim, num_heads=num_heads, bias=bias)
        self.norm2 = LayerNorm2d(dim)
        self.ffn = GDFN(dim=dim, expansion_factor=expansion_factor, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class RestormerFeatureExtractor(nn.Module):
    """
    Lightweight Restormer-style feature extractor.
    """

    def __init__(self, in_channels: int = 64, num_blocks: int = 4, num_heads: int = 4, expansion_factor: float = 2.0):
        super().__init__()
        self.entry = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=True)
        self.blocks = nn.Sequential(
            *[
                RestormerBlock(
                    dim=in_channels,
                    num_heads=num_heads,
                    expansion_factor=expansion_factor,
                )
                for _ in range(num_blocks)
            ]
        )
        self.exit = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.entry(x)
        x = self.blocks(x)
        x = self.exit(x)
        return x


if __name__ == "__main__":
    # Quick test
    x = torch.randn(1, 64, 128, 128)  # H and W must be divisible by window_size=8

    print("Testing WindowSelfAttention...")
    attn = WindowSelfAttention(dim=64, num_heads=4, window_size=8)
    y = attn(x)
    print("WindowSelfAttention output shape:", y.shape)

    print("Testing TransformerBlock...")
    block = TransformerBlock(dim=64, num_heads=4, window_size=8)
    y2 = block(x)
    print("TransformerBlock output shape:", y2.shape)

    print("Testing TransformerFeatureExtractor...")
    tfeat = TransformerFeatureExtractor(
        in_channels=64, num_blocks=3, num_heads=4, window_size=8
    )
    y3 = tfeat(x)
    print("TransformerFeatureExtractor output shape:", y3.shape)
