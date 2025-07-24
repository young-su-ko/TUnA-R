import pytest
import torch

from tuna.layers._transformer_block import Encoder, TransformerBlock


class TestTransformerBlock:
    @pytest.mark.parametrize("use_spectral_norm", [False, True])
    def test_output_shape(self, use_spectral_norm):
        model = TransformerBlock(
            hid_dim=128,
            n_heads=4,
            ffn_dim=128 * 4,
            dropout=0.1,
            use_spectral_norm=use_spectral_norm,
        )

        x = torch.randn(1, 10, 128)
        out = model(x)
        assert out.shape == (1, 10, 128)


class TestEncoder:
    @pytest.mark.parametrize("use_spectral_norm", [False, True])
    def test_output_shape(self, use_spectral_norm):
        model = Encoder(
            hid_dim=128,
            n_layers=1,
            n_heads=4,
            ffn_dim=128 * 4,
            dropout=0.1,
            use_spectral_norm=use_spectral_norm,
        )

        x = torch.randn(1, 10, 128)
        out = model(x)
        assert out.shape == (1, 10, 128)
