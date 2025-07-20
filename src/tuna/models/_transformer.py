import warnings

import torch
import torch.nn as nn
from omegaconf import DictConfig
from uncertaintyAwareDeepLearn import VanillaRFFLayer

from tuna.layers._transformer_block import Encoder
from tuna.models.model_utils import make_linear_layer


class Transformer(nn.Module):
    def __init__(
        self,
        protein_dim: int,
        hid_dim: int,
        dropout: float,
        n_layers: int,
        n_heads: int,
        ff_dim: int,
        llgp: bool,
        use_spectral_norm: bool,
        out_targets: int,
        gp_config: DictConfig | None,
    ):
        super().__init__()
        self.protein_dim = protein_dim
        self.hid_dim = hid_dim
        self.dropout = dropout
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.ff_dim = ff_dim
        self.llgp = llgp
        self.use_spectral_norm = use_spectral_norm
        self.out_targets = out_targets
        self.gp_config = gp_config

        if self.llgp and not self.use_spectral_norm:
            warnings.warn(
                "It is recommended to use spectral normalization when llgp is True.",
                UserWarning,
            )

        if self.llgp:
            if self.gp_config is None:
                raise ValueError("gp_config must be provided when llgp=True")

            self.output_layer = VanillaRFFLayer(
                in_features=self.hid_dim,
                RFFs=self.gp_config.rff_features,
                out_targets=self.out_targets,
                gp_cov_momentum=self.gp_config.gp_cov_momentum,
                gp_ridge_penalty=self.gp_config.gp_ridge_penalty,
                likelihood=self.gp_config.likelihood,
            )
        else:
            self.output_layer = make_linear_layer(
                self.hid_dim, self.out_targets, use_spectral_norm=self.use_spectral_norm
            )

        self.down_project = make_linear_layer(
            self.protein_dim, self.hid_dim, use_spectral_norm=self.use_spectral_norm
        )
        self.intra_encoder = Encoder(
            self.hid_dim,
            self.n_layers,
            self.n_heads,
            self.ff_dim,
            self.dropout,
            self.use_spectral_norm,
        )
        self.inter_encoder = Encoder(
            self.hid_dim,
            self.n_layers,
            self.n_heads,
            self.ff_dim,
            self.dropout,
            self.use_spectral_norm,
        )

        self._update_precision = False
        self._get_variance = False

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)

    def _set_llgp_mode(
        self, update_precision: bool = False, get_variance: bool = False
    ):
        if not self.llgp:
            return
        self._update_precision = update_precision
        self._get_variance = get_variance

    def forward(
        self,
        proteinA: torch.Tensor,
        proteinB: torch.Tensor,
        masks: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
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
            return self.output_layer(
                ppi_feature,
                update_precision=self._update_precision,
                get_var=self._get_variance,
            )
        return self.output_layer(ppi_feature)
