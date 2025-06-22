import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from uncertaintyAwareDeepLearn import VanillaRFFLayer


class MLP(nn.Module):
    def __init__(
        self,
        protein_dim: int,
        hid_dim: int,
        dropout: float,
        llgp: bool,
        rff_features: int = 4096,
        out_targets: int = 1,
        gp_cov_momentum: float = -1,
        gp_ridge_penalty: float = 1,
        likelihood_function: str = "binary_logistic",
    ):
        super().__init__()
        self.protein_dim = protein_dim
        self.hid_dim = hid_dim
        self.dropout = dropout
        self.llgp = llgp
        self.fc1 = nn.Linear(self.protein_dim*2, self.hid_dim)
        self.fc2 = nn.Linear(self.hid_dim, self.hid_dim)
        self.output_layer = nn.Linear(self.hid_dim, 1)
        
        if self.llgp:
            self.fc1 = spectral_norm(self.fc1)
            self.fc2 = spectral_norm(self.fc2)
            self.output_layer = VanillaRFFLayer(
                in_features=self.hid_dim,
                RFFs=rff_features,
                out_targets=out_targets,
                gp_cov_momentum=gp_cov_momentum,
                gp_ridge_penalty=gp_ridge_penalty,
                likelihood_function=likelihood_function,
            )

        self.relu = nn.ReLU()
        self.do = nn.Dropout(self.dropout)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)


    def forward(self, proteinA, proteinB):
        concatenated = torch.cat((proteinA, proteinB), dim=1)
        x = self.fc1(concatenated)
        x = self.relu(x)
        x = self.do(x)

        x = self.fc2(x)
        x = self.relu(x)
        x = self.do(x)

        logit = self.output_layer(x)

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
        rff_features: int = 4096,
        out_targets: int = 1,
        gp_cov_momentum: float = -1,
        gp_ridge_penalty: float = 1,
        likelihood_function: str = "binary_logistic",
    ):
        super().__init__()
        self.protein_dim = protein_dim
        self.hid_dim = hid_dim
        self.dropout = dropout
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.ff_dim = ff_dim
        self.llgp = llgp
        self.down_project = nn.Linear(self.protein_dim, self.hid_dim)

        if self.llgp:
            self.down_project = spectral_norm(self.down_project)

        self.intra_encoder = IntraEncoder(self.protein_dim, self.hid_dim, self.n_layers, self.n_heads, self.ff_dim, self.dropout, self.llgp)
        self.inter_encoder = InterEncoder(self.protein_dim, self.hid_dim, self.n_layers, self.n_heads, self.ff_dim, self.dropout, self.llgp)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
