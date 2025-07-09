from enum import Enum
from typing import Protocol


class LLGPModel(Protocol):
    def _set_llgp_mode(self, update_precision: bool, get_variance: bool) -> None: ...


class LLGPMode(Enum):
    TRAINING = "training"  # update_precision depends on epoch, no variance
    VALIDATION = "validation"  # no precision update, no variance
    INFERENCE = "inference"  # no precision update, get variance


def set_llgp_mode(
    model: LLGPModel, mode: LLGPMode, *, is_last_epoch: bool = False
) -> None:
    if mode == LLGPMode.TRAINING:
        model._set_llgp_mode(update_precision=is_last_epoch, get_variance=False)
    elif mode == LLGPMode.VALIDATION:
        model._set_llgp_mode(update_precision=False, get_variance=False)
    else:  # INFERENCE mode
        model._set_llgp_mode(update_precision=False, get_variance=True)
