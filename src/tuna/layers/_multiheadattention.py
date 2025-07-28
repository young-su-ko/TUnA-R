import torch
import torch.nn as nn
import torch.nn.functional as F

from tuna.models.model_utils import make_linear_layer


class MultiHeadAttention(nn.Module):
    def __init__(
        self, hid_dim: int, n_heads: int, dropout: float, use_spectral_norm: bool
    ):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.dropout = dropout
        self.use_spectral_norm = use_spectral_norm
        self.head_dim = hid_dim // n_heads
        assert hid_dim % n_heads == 0, "hid_dim must be divisible by n_heads"

        self.q = make_linear_layer(self.hid_dim, self.hid_dim, self.use_spectral_norm)
        self.k = make_linear_layer(self.hid_dim, self.hid_dim, self.use_spectral_norm)
        self.v = make_linear_layer(self.hid_dim, self.hid_dim, self.use_spectral_norm)
        self.output = make_linear_layer(
            self.hid_dim, self.hid_dim, self.use_spectral_norm
        )

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        batch_size = x.size(0)

        q = self.q(x).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )

        attn_out = (
            attn_out.transpose(1, 2).contiguous().view(batch_size, -1, self.hid_dim)
        )
        return self.output(attn_out)
