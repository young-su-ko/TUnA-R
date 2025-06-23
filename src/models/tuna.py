import torch
import torch.nn as nn
from uncertaintyAwareDeepLearn import VanillaRFFLayer
from torch import Tensor
from typing import Optional
from src.layers.transformer import Encoder
from model_utils import make_linear_layer
import warnings

class TUnA(nn.Module):
    def __init__(
        self,
        protein_dim: int,
        hid_dim: int,
        dropout: float,
        n_layers: int,
        n_heads: int,
        ff_dim: int,
        llgp: bool,
        spectral_norm: bool,
        out_targets: int = 1,
        rff_features: Optional[int] = 4096,
        gp_cov_momentum: Optional[float] = -1,
        gp_ridge_penalty: Optional[float] = 1,
        likelihood_function: Optional[str] = "binary_logistic",
    ):
        super().__init__()
        self.protein_dim = protein_dim
        self.hid_dim = hid_dim
        self.dropout = dropout
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.ff_dim = ff_dim
        self.llgp = llgp
        self.spectral_norm = spectral_norm

        if self.llgp and not self.spectral_norm:
            warnings.warn(
                "It is recommended to use spectral normalization when llgp is True.",
                UserWarning
            )

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
            self.output_layer = make_linear_layer(self.hid_dim, 1, self.spectral_norm)

        self.down_project = make_linear_layer(self.protein_dim, self.hid_dim, self.spectral_norm)
        self.intra_encoder = Encoder(self.protein_dim, self.hid_dim, self.n_layers, self.n_heads, self.ff_dim, self.dropout, self.spectral_norm)
        self.inter_encoder = Encoder(self.protein_dim, self.hid_dim, self.n_layers, self.n_heads, self.ff_dim, self.dropout, self.spectral_norm)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)

    def forward(self, proteinA: Tensor, proteinB: Tensor, mask: Optional[Tensor] = None, update_precision: Optional[bool] = False, get_variance: Optional[bool] = False) -> Tensor:
        proteinA = self.down_project(proteinA)
        proteinB = self.down_project(proteinB)

        proteinA = self.intra_encoder(proteinA)
        proteinB = self.intra_encoder(proteinB)

        proteinAB = torch.cat((proteinA, proteinB), dim=1)
        proteinBA = torch.cat((proteinB, proteinA), dim=1)
        
        proteinAB = self.inter_encoder(proteinAB)
        proteinBA = self.inter_encoder(proteinBA)

        ppi_feature, _ = torch.max(torch.cat((proteinAB, proteinBA), dim=-1), dim=1)
        logits = self.output_layer(ppi_feature, update_precision=update_precision, get_var=get_variance)

        return logits