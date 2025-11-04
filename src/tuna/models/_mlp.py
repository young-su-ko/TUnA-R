import warnings

import torch
import torch.nn as nn
from omegaconf import DictConfig
from uncertaintyAwareDeepLearn import VanillaRFFLayer

from tuna.models.model_utils import make_linear_layer


class MLP(nn.Module):
    embedding_type: str = "protein"

    def __init__(
        self,
        protein_dim: int,
        hid_dim: int,
        dropout: float,
        llgp: bool,
        use_spectral_norm: bool,
        out_targets: int,
        gp_config: DictConfig | None = None,
    ):
        super().__init__()
        self.protein_dim = protein_dim
        self.hid_dim = hid_dim
        self.dropout = dropout
        self.out_targets = out_targets
        self.llgp = llgp
        self.use_spectral_norm = use_spectral_norm
        self.gp_config = gp_config

        if self.llgp and not self.use_spectral_norm:
            warnings.warn(
                "It is recommended to use spectral normalization when llgp is True.",
                UserWarning,
            )

        self.fc1 = make_linear_layer(
            self.protein_dim * 2, self.hid_dim, self.use_spectral_norm
        )
        self.fc2 = make_linear_layer(self.hid_dim, self.hid_dim, self.use_spectral_norm)
        self.relu = nn.ReLU()
        self.do = nn.Dropout(self.dropout)

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
                self.hid_dim, self.out_targets, self.use_spectral_norm
            )

        self._update_precision = False
        self._get_variance = False

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
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
        self, proteinA: torch.Tensor, proteinB: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        concatenated = torch.cat((proteinA, proteinB), dim=-1)
        x = self.fc1(concatenated)
        x = self.relu(x)
        x = self.do(x)

        x = self.fc2(x)
        x = self.relu(x)
        x = self.do(x)

        if self.llgp:
            return self.output_layer(
                x, update_precision=self._update_precision, get_var=self._get_variance
            )
        return self.output_layer(x)
