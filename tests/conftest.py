import omegaconf
import pytest


@pytest.fixture
def gp_config():
    return omegaconf.DictConfig(
        {
            "rff_features": 128,
            "gp_cov_momentum": -1,
            "gp_ridge_penalty": 1,
            "likelihood": "gaussian",
        }
    )
