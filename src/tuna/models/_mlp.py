import torch
import torch.nn as nn
from uncertaintyAwareDeepLearn import VanillaRFFLayer
from torch import Tensor
from typing import Optional
from tuna.models.model_utils import make_linear_layer
import warnings

class MLP(nn.Module):
    def __init__(
        self,
        protein_dim: int,
        hid_dim: int,
        dropout: float,
        llgp: bool,
        spectral_norm: bool,
        out_targets: int = 1,
        rff_features: Optional[int] = 4096,
        gp_cov_momentum: Optional[float] = -1,
        gp_ridge_penalty: Optional[float] = 1,
        likelihood_function: str = "binary_logistic",
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
            self.output_layer = make_linear_layer(self.hid_dim, 1, self.spectral_norm)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)

    def forward(self, proteinA: Tensor, proteinB: Tensor, update_precision: Optional[bool] = False, get_variance: Optional[bool] = False) -> Tensor:
        concatenated = torch.cat((proteinA, proteinB), dim=1)
        x = self.fc1(concatenated)
        x = self.relu(x)
        x = self.do(x)

        x = self.fc2(x)
        x = self.relu(x)
        x = self.do(x)

        logits = self.output_layer(x, update_precision=update_precision, get_var=get_variance)

        return logits