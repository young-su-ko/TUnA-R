import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional
from tuna.layers._multiheadattention import MultiHeadAttention
from tuna.models.model_utils import make_linear_layer

class TransformerBlock(nn.Module):
    def __init__(self, hid_dim: int, ffn_dim: int, n_heads: int, dropout: float, spectral_norm: bool):
        super().__init__()
        self.hid_dim = hid_dim
        self.ffn_dim = ffn_dim
        self.n_heads = n_heads
        self.dropout = dropout
        self.spectral_norm = spectral_norm

        self.attention_layer_norm = nn.LayerNorm(hid_dim)
        self.attention = MultiHeadAttention(hid_dim, n_heads, dropout, spectral_norm)
        self.do = nn.Dropout(dropout)
        
        self.ffn = nn.Sequential(
            make_linear_layer(hid_dim, ffn_dim, spectral_norm),
            nn.SiLU(),
            make_linear_layer(ffn_dim, hid_dim, spectral_norm),
        )

        self.ffn_layer_norm = nn.LayerNorm(hid_dim)
        self.output_layer_norm = nn.LayerNorm(hid_dim)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
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

class Encoder(nn.Module):
    def __init__(self, hid_dim: int, n_layers: int, n_heads: int, ffn_dim: int, dropout: float, spectral_norm: bool):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        self.spectral_norm = spectral_norm

        self.layers = nn.ModuleList([TransformerBlock(hid_dim, ffn_dim, n_heads, dropout, spectral_norm) for _ in range(n_layers)])

        def forward(self, x: Tensor, mask: Optional[Tensor] = None):
            for layer in self.layers:
                x = layer(x, mask=mask)
            return x