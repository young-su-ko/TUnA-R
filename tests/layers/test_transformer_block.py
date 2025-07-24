import pytest
import torch

from tuna.layers._transformer_block import Encoder, TransformerBlock


@pytest.fixture
def transformer_block_with_spectral_norm():
    return TransformerBlock(
        hid_dim=128,
        n_heads=4,
        ffn_dim=128 * 4,
        dropout=0.1,
        use_spectral_norm=True,
    )


@pytest.fixture
def transformer_block_without_spectral_norm():
    return TransformerBlock(
        hid_dim=128,
        n_heads=4,
        ffn_dim=128 * 4,
        dropout=0.1,
        use_spectral_norm=False,
    )


class TestTransformerBlock:
    def test_forward_pass_with_spectral_norm(
        self, transformer_block_with_spectral_norm
    ):
        x = torch.randn(1, 10, 128)
        out = transformer_block_with_spectral_norm(x)
        assert out.shape == (1, 10, 128)

    def test_forward_pass_without_spectral_norm(
        self, transformer_block_without_spectral_norm
    ):
        x = torch.randn(1, 10, 128)
        out = transformer_block_without_spectral_norm(x)
        assert out.shape == (1, 10, 128)


@pytest.fixture
def encoder_with_spectral_norm():
    return Encoder(
        hid_dim=128,
        n_layers=1,
        n_heads=4,
        ffn_dim=128 * 4,
        dropout=0.1,
        use_spectral_norm=True,
    )


@pytest.fixture
def encoder_without_spectral_norm():
    return Encoder(
        hid_dim=128,
        n_layers=1,
        n_heads=4,
        ffn_dim=128 * 4,
        dropout=0.1,
        use_spectral_norm=False,
    )


class TestEncoder:
    def test_forward_pass_with_spectral_norm(self, encoder_with_spectral_norm):
        x = torch.randn(1, 10, 128)
        out = encoder_with_spectral_norm(x)
        assert out.shape == (1, 10, 128)

    def test_forward_pass_without_spectral_norm(self, encoder_without_spectral_norm):
        x = torch.randn(1, 10, 128)
        out = encoder_without_spectral_norm(x)
        assert out.shape == (1, 10, 128)

    def test_forward_pass_with_mask_and_spectral_norm(self, encoder_with_spectral_norm):
        x = torch.randn(1, 10, 128)
        mask = torch.randn(1, 10, 10)
        out = encoder_with_spectral_norm(x, mask)
        assert out.shape == (1, 10, 128)

    def test_forward_pass_with_mask_without_spectral_norm(
        self, encoder_without_spectral_norm
    ):
        x = torch.randn(1, 10, 128)
        mask = torch.randn(1, 10, 10)
        out = encoder_without_spectral_norm(x, mask)
        assert out.shape == (1, 10, 128)
