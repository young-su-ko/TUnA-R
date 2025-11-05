from enum import Enum
from typing import Protocol

import torch.nn as nn


class LLGPModel(Protocol):
    llgp: bool
    _update_precision: bool
    _get_variance: bool


class LLGPMode(Enum):
    TRAINING = "training"
    VALIDATION = "validation"
    INFERENCE = "inference"


def is_llgp(model: nn.Module) -> bool:
    """
    Checks if the model is using a last-layer Gaussian process (LLGP).
    """
    return getattr(model, "llgp", False)


def set_llgp_mode(model: LLGPModel, mode: LLGPMode, is_last_epoch: bool) -> None:
    """
    Sets the LLGP mode flags on the model backbone.
    This directly updates _update_precision and _get_variance attributes.
    """
    if mode == LLGPMode.TRAINING:
        # TRAINING mode: update_precision is true only on last epoch, get_var always false
        model._update_precision = is_last_epoch
        model._get_variance = False
    elif mode == LLGPMode.VALIDATION:
        # VALIDATION mode: update_precision always false, get_var always false
        model._update_precision = False
        model._get_variance = False
    else:
        # INFERENCE mode: update_precision always false, get_var always true
        model._update_precision = False
        model._get_variance = True
