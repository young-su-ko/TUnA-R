import torch
import torch.nn as nn
from uncertaintyAwareDeepLearn import VanillaRFFLayer
from tuna.layers._transformer import Encoder
from tuna.models.model_utils import make_linear_layer
from tuna.config.model_config import TUnAConfig
import warnings

class TUnA(nn.Module):
    @classmethod
    def from_config(cls, config: TUnAConfig) -> "TUnA":
        return cls(
            protein_dim=config.architecture.protein_dim,
            hid_dim=config.architecture.hid_dim,
            dropout=config.architecture.dropout,
            n_layers=config.architecture.n_layers,
            n_heads=config.architecture.n_heads,
            ff_dim=config.architecture.ff_dim,
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
        n_layers: int,
        n_heads: int,
        ff_dim: int,
        llgp: bool,
        spectral_norm: bool,
        out_targets: int = 1,
        rff_features: int | None = None,
        gp_cov_momentum: float | None = None,
        gp_ridge_penalty: float | None = None,
        likelihood_function: str | None = None,
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
            self.output_layer = make_linear_layer(self.hid_dim, out_targets, self.spectral_norm)

        self.down_project = make_linear_layer(self.protein_dim, self.hid_dim, self.spectral_norm)
        self.intra_encoder = Encoder(self.protein_dim, self.hid_dim, self.n_layers, self.n_heads, self.ff_dim, self.dropout, self.spectral_norm)
        self.inter_encoder = Encoder(self.protein_dim, self.hid_dim, self.n_layers, self.n_heads, self.ff_dim, self.dropout, self.spectral_norm)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)

    def forward(self, proteinA: torch.Tensor, proteinB: torch.Tensor, masks: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], update_precision: bool, get_variance: bool) -> torch.Tensor:
        maskA, maskB, maskAB, maskBA = masks
        
        proteinA = self.down_project(proteinA)
        proteinB = self.down_project(proteinB)

        proteinA = self.intra_encoder(proteinA, maskA)
        proteinB = self.intra_encoder(proteinB, maskB)

        proteinAB = torch.cat((proteinA, proteinB), dim=1)
        proteinBA = torch.cat((proteinB, proteinA), dim=1)
        
        proteinAB = self.inter_encoder(proteinAB, maskAB)
        proteinBA = self.inter_encoder(proteinBA, maskBA)

        ppi_feature, _ = torch.max(torch.cat((proteinAB, proteinBA), dim=-1), dim=1)

        if self.llgp:
            logits = self.output_layer(ppi_feature, update_precision=update_precision, get_var=get_variance)
        else:
            logits = self.output_layer(ppi_feature)

        return logits