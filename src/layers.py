import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from einops import rearrange


class MultiHeadAttention(nn.Module):
    def __init__(self, hid_dim: int, n_heads: int, dropout: float, llgp: bool):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.dropout = dropout
        self.llgp = llgp
        self.head_dim = hid_dim // n_heads
        assert hid_dim % n_heads == 0, "hid_dim must be divisible by n_heads"

        self.q = nn.Linear(hid_dim, hid_dim)
        self.k = nn.Linear(hid_dim, hid_dim)
        self.v = nn.Linear(hid_dim, hid_dim)
        self.output = nn.Linear(hid_dim, hid_dim)

        if self.llgp:
            self.q = spectral_norm(self.q)
            self.k = spectral_norm(self.k)
            self.v = spectral_norm(self.v)
            self.output = spectral_norm(self.output)


    def forward(self, x, mask=None):
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

class TransformerBlock(nn.Module):
    def __init__(self, hid_dim: int, ffn_dim: int, n_heads: int, dropout: float, llgp: bool):
        super().__init__()
        self.hid_dim = hid_dim
        self.ffn_dim = ffn_dim
        self.n_heads = n_heads
        self.dropout = dropout
        self.llgp = llgp

        self.attention_layer_norm = nn.LayerNorm(hid_dim)
        self.attention = MultiHeadAttention(hid_dim, n_heads, dropout, llgp)
        self.do = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(hid_dim, ffn_dim),
            nn.SiLU(),
            nn.Linear(ffn_dim, hid_dim),
        )
        self.ffn_layer_norm = nn.LayerNorm(hid_dim)
        self.output_layer_norm = nn.LayerNorm(hid_dim)

    def forward(self, x, mask=None):
        residual = x
        x = self.attention(x, mask=mask)
        x = self.do(x)
        x = x + residual
        x = self.attention_layer_norm(x)

        residual = x
        x = self.ffn(x)
        x = self.do(x)
        x = x + residual
        x = self.output_layer_norm(x)

        return x

class IntraEncoder(nn.Module):
    def __init__(self, hid_dim: int, n_layers: int, n_heads: int, ffn_dim: int, dropout: float, llgp: bool):