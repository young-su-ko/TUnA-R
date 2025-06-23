import torch
import torch.nn as nn
from torch import Tensor
import math

def mean_field_average(logits: Tensor, variance: Tensor) -> Tensor:
    """
    When we have the variance of the logits during inference, we 
    adjust the logits to get a better estimate of the probability.
    See https://arxiv.org/abs/2006.10108 for more details.
    """
    adjusted_score = logits / torch.sqrt(1. + (math.pi /8.)*variance)
    adjusted_score = torch.sigmoid(adjusted_score).squeeze()
    
    return adjusted_score

def make_linear_layer(in_features: int, out_features: int, spectral_norm: bool = True) -> nn.Linear:
    """
    This is a helper function to build linear layers with the option to use spectral norm.
    """
    layer = nn.Linear(in_features, out_features)
    if spectral_norm:
        layer = spectral_norm(layer)
    return layer

