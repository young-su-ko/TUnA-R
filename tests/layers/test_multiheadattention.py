import pytest
import torch
import torch.nn.utils.parametrize as P

from tuna.layers._multiheadattention import MultiHeadAttention


class TestMultiHeadAttention:
    @pytest.mark.parametrize("use_spectral_norm", [False, True])
    def test_output_shape(self, use_spectral_norm):
        model = MultiHeadAttention(
            hid_dim=128, n_heads=4, dropout=0.1, use_spectral_norm=use_spectral_norm
        )
        x = torch.randn(1, 10, 128)
        out = model(x)
        assert out.shape == (1, 10, 128)

    @pytest.mark.parametrize("use_spectral_norm", [False, True])
    def test_mask(self, use_spectral_norm):
        model = MultiHeadAttention(
            hid_dim=128, n_heads=4, dropout=0.1, use_spectral_norm=use_spectral_norm
        )
        x = torch.randn(1, 10, 128)
        mask = torch.randn(1, 10, 10)
        out = model(x, mask)
        assert out.shape == (1, 10, 128)

    def test_spectral_norm(self):
        model = MultiHeadAttention(
            hid_dim=128, n_heads=4, dropout=0.1, use_spectral_norm=True
        )
        for name in ["q", "k", "v", "output"]:
            layer = getattr(model, name)
            assert P.is_parametrized(layer)

    def test_no_spectral_norm(self):
        model = MultiHeadAttention(
            hid_dim=128, n_heads=4, dropout=0.1, use_spectral_norm=False
        )
        for name in ["q", "k", "v", "output"]:
            layer = getattr(model, name)
            assert not P.is_parametrized(layer)
