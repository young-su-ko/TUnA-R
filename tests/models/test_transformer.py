import pytest
import torch

from tuna.models._transformer import Transformer


@pytest.fixture
def transformer_model_without_llgp():
    return Transformer(
        protein_dim=1024,
        hid_dim=128,
        dropout=0.1,
        n_layers=2,
        n_heads=4,
        ff_dim=128,
        llgp=False,
        use_spectral_norm=True,
        out_targets=1,
        gp_config=None,
    )


@pytest.fixture
def transformer_model_with_llgp(gp_config):
    return Transformer(
        protein_dim=1024,
        hid_dim=128,
        dropout=0.1,
        n_layers=2,
        n_heads=4,
        ff_dim=128,
        llgp=True,
        use_spectral_norm=True,
        out_targets=1,
        gp_config=gp_config,
    )


class TestTransformer:
    def test_forward_pass(self, transformer_model_without_llgp):
        proteinA = torch.randn(1, 10, 1024)
        proteinB = torch.randn(1, 10, 1024)
        masks = (
            torch.ones(1, 1, 10, 10),
            torch.ones(1, 1, 10, 10),
            torch.ones(1, 1, 20, 20),
            torch.ones(1, 1, 20, 20),
        )
        output = transformer_model_without_llgp(proteinA, proteinB, masks)
        assert output.shape == (1, 1)

    def test_forward_pass_with_llgp(self, transformer_model_with_llgp):
        proteinA = torch.randn(1, 10, 1024)
        proteinB = torch.randn(1, 10, 1024)
        masks = (
            torch.ones(1, 1, 10, 10),
            torch.ones(1, 1, 10, 10),
            torch.ones(1, 1, 20, 20),
            torch.ones(1, 1, 20, 20),
        )
        output = transformer_model_with_llgp(proteinA, proteinB, masks)
        assert output.shape == (1, 1)
