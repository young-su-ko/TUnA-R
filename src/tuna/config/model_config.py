from dataclasses import dataclass

@dataclass
class BaseModelConfig:
    protein_dim: int
    hid_dim: int
    dropout: float
    llgp: bool
    spectral_norm: bool
    out_targets: int = 1
    rff_features: int | None
    gp_cov_momentum: float | None
    gp_ridge_penalty: float | None
    likelihood_function: str | None

@dataclass
class TUnAConfig(BaseModelConfig):
    n_layers: int
    n_heads: int
    ff_dim: int

@dataclass
class MLPConfig(BaseModelConfig):
    pass