import torch
import torch.nn as nn

from tuna.models._mlp import MLP
from tuna.models._transformer import Transformer
from tuna.models.mask_maker import MaskMaker
from tuna.models.model_utils import is_llgp, mean_field_average
from tuna.pl_modules.llgp_utils import LLGPMode, set_llgp_mode


class PPIPredictor(nn.Module):
    """
    The general class for predicting protein-protein interactions (PPIs) using a specified model backbone.
    This allows for flexibility in choosing different architectures for PPI prediction (e.g., Transformer, MLP).

    The TUnA architecture from the original manuscript is achieved by using a Transformer with LLGP as the backbone.
    """

    def __init__(self, model_backbone: Transformer | MLP):
        super().__init__()
        self.model_backbone = model_backbone
        if isinstance(self.model_backbone, Transformer):
            self.mask_maker = MaskMaker(device="cpu")
            # Device will be updated on fit start to GPU if available.

    def forward(
        self,
        proteinA: torch.Tensor,
        proteinB: torch.Tensor,
        lengthsA: list[int] | None = None,
        lengthsB: list[int] | None = None,
        mode: LLGPMode | None = None,
        is_last_epoch: bool = False,
    ):
        if is_llgp(self.model_backbone):
            if mode is None:
                mode = LLGPMode.INFERENCE
            set_llgp_mode(self.model_backbone, mode, is_last_epoch=is_last_epoch)

        if isinstance(self.model_backbone, Transformer):
            if lengthsA is None or lengthsB is None:
                raise ValueError(
                    "lengthsA and lengthsB are required when using Transformer backbone"
                )
            masks = self.mask_maker.make_masks(
                lengthsA, lengthsB, proteinA.size(1), proteinB.size(1)
            )
            output = self.model_backbone(proteinA, proteinB, masks)
        elif isinstance(self.model_backbone, MLP):
            output = self.model_backbone(proteinA, proteinB)

        if is_llgp(self.model_backbone) and mode == LLGPMode.INFERENCE:
            logits, var = output
            logits = mean_field_average(logits, var)
        else:
            logits = output

        return logits.reshape(-1)
