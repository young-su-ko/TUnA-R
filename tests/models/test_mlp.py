import pytest
import torch

from tuna.models._mlp import MLP


@pytest.fixture
def mlp_model_without_llgp():
    return MLP(
        protein_dim=1024,
        hid_dim=128,
        dropout=0.1,
        llgp=False,
        use_spectral_norm=True,
        out_targets=1,
        gp_config=None,
    )


@pytest.fixture
def mlp_model_with_llgp(gp_config):
    return MLP(
        protein_dim=1024,
        hid_dim=128,
        dropout=0.1,
        llgp=True,
        use_spectral_norm=True,
        out_targets=1,
        gp_config=gp_config,
    )


class TestMLP:
    def test_forward_pass(self, mlp_model_without_llgp):
        model = mlp_model_without_llgp
        proteinA = torch.randn(10, 1024)
        proteinB = torch.randn(10, 1024)
        out = model(proteinA, proteinB)
        assert out.shape == (10, 1)

    def test_forward_pass_with_llgp(self, mlp_model_with_llgp):
        model = mlp_model_with_llgp
        proteinA = torch.randn(10, 1024)
        proteinB = torch.randn(10, 1024)
        out = model(proteinA, proteinB)
        assert out.shape == (10, 1)
