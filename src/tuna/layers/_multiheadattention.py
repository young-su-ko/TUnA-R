import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from tuna.models.model_utils import make_linear_layer

class MultiHeadAttention(nn.Module):
    def __init__(self, hid_dim: int, n_heads: int, dropout: float, spectral_norm: bool):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.dropout = dropout
        self.spectral_norm = spectral_norm
        self.head_dim = hid_dim // n_heads
        assert hid_dim % n_heads == 0, "hid_dim must be divisible by n_heads"

        self.q = make_linear_layer(hid_dim, hid_dim, spectral_norm)
        self.k = make_linear_layer(hid_dim, hid_dim, spectral_norm)
        self.v = make_linear_layer(hid_dim, hid_dim, spectral_norm)
        self.output = make_linear_layer(hid_dim, hid_dim, spectral_norm)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        q = rearrange(q, 'b s (h d) -> b h s d', h=self.n_heads)
        k = rearrange(k, 'b s (h d) -> b h s d', h=self.n_heads)
        v = rearrange(v, 'b s (h d) -> b h s d', h=self.n_heads)

        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=self.dropout,
            is_causal=False
        )

        attn_out = rearrange(attn_out, 'b h s d -> b s (h d)')
        attn_out = self.output(attn_out)

        return attn_out