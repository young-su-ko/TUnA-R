import math

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm


def mean_field_average(logits: torch.Tensor, variance: torch.Tensor) -> torch.Tensor:
    """
    When we have the variance of the logits during inference, we
    adjust the logits to get a better estimate of the probability.
    See https://arxiv.org/abs/2006.10108 for more details.
    """
    adjusted_logits = logits / torch.sqrt(1.0 + (math.pi / 8.0) * variance)
    # right now we return logits
    # adjusted_score = torch.sigmoid(adjusted_score).squeeze()

    return adjusted_logits


def make_linear_layer(
    in_features: int, out_features: int, use_spectral_norm: bool = True
) -> nn.Linear:
    """
    This is a helper function to build linear layers with the option to use spectral norm.
    """
    layer = nn.Linear(in_features, out_features)
    if use_spectral_norm:
        layer = spectral_norm(layer)
    return layer


def is_llgp(model: nn.Module) -> bool:
    """
    This is a helper function to check if the model is a LLGP model.
    """
    return model.llgp
