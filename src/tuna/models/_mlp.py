import torch
import torch.nn as nn
from uncertaintyAwareDeepLearn import VanillaRFFLayer
from tuna.models.model_utils import make_linear_layer
from tuna.config.model_config import MLPConfig
import warnings

class MLP(nn.Module):
    @classmethod
    def from_config(cls, config: MLPConfig) -> "MLP":
        return cls(
            protein_dim=config.architecture.protein_dim,
            hid_dim=config.architecture.hid_dim,
            dropout=config.architecture.dropout,
            llgp=config.architecture.llgp,
            spectral_norm=config.architecture.spectral_norm,
            out_targets=config.architecture.out_targets,
            rff_features=config.architecture.rff_features,
            gp_cov_momentum=config.architecture.gp_cov_momentum,
            gp_ridge_penalty=config.architecture.gp_ridge_penalty,
            likelihood_function=config.architecture.likelihood_function,
        )

    def __init__(
        self,
        protein_dim: int,
        hid_dim: int,
        dropout: float,
        llgp: bool,
        spectral_norm: bool,
        out_targets: int = 1,
        rff_features: int | None,
        gp_cov_momentum: float | None,
        gp_ridge_penalty: float | None,
        likelihood_function: str | None,
    ):
        super().__init__()
        self.protein_dim = protein_dim
        self.hid_dim = hid_dim
        self.dropout = dropout
        self.llgp = llgp
        self.spectral_norm = spectral_norm

        if self.llgp and not self.spectral_norm:
            warnings.warn(
                "It is recommended to use spectral normalization when llgp is True.",
                UserWarning
            )

        self.fc1 = make_linear_layer(self.protein_dim*2, self.hid_dim, spectral_norm)
        self.fc2 = make_linear_layer(self.hid_dim, self.hid_dim, spectral_norm)
        self.relu = nn.ReLU()
        self.do = nn.Dropout(self.dropout)
        
        if self.llgp:
            self.output_layer = VanillaRFFLayer(
                in_features=self.hid_dim,
                RFFs=rff_features,
                out_targets=out_targets,
                gp_cov_momentum=gp_cov_momentum,
                gp_ridge_penalty=gp_ridge_penalty,
                likelihood_function=likelihood_function,
            )
        else:
            self.output_layer = make_linear_layer(self.hid_dim, out_targets, self.spectral_norm)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)

    def forward(self, proteinA: torch.Tensor, proteinB: torch.Tensor, update_precision: bool, get_variance: bool) -> torch.Tensor:
        concatenated = torch.cat((proteinA, proteinB), dim=1)
        x = self.fc1(concatenated)
        x = self.relu(x)
        x = self.do(x)

        x = self.fc2(x)
        x = self.relu(x)
        x = self.do(x)

        if self.llgp:
            logits = self.output_layer(x, update_precision=update_precision, get_var=get_variance)
        else:
            logits = self.output_layer(x)

        return logits