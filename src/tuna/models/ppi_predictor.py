import torch
import torch.nn as nn

from tuna.models._mlp import MLP
from tuna.models._transformer import Transformer
from tuna.models.llgp_utils import LLGPMode, is_llgp, set_llgp_mode
from tuna.models.mask_utils import make_masks
from tuna.models.model_utils import mean_field_average


class PPIPredictor(nn.Module):
    """
    A general class for predicting protein-protein interactions (PPIs) using a specified model backbone.
    Allows for flexibility in choosing different architectures for PPI prediction (e.g., Transformer, MLP).

    The TUnA architecture from the original manuscript is achieved by using a Transformer with LLGP as the backbone.
    """

    def __init__(self, model_backbone: Transformer | MLP):
        super().__init__()
        self.model_backbone = model_backbone
        self.llgp = is_llgp(self.model_backbone)

    def forward(
        self,
        proteinA: torch.Tensor,
        proteinB: torch.Tensor,
        mode: LLGPMode,
        lengthsA: list[int] | None = None,
        lengthsB: list[int] | None = None,
        is_last_epoch: bool = False,
    ):
        if self.llgp:
            set_llgp_mode(self.model_backbone, mode, is_last_epoch=is_last_epoch)

        # Transformer logic
        if isinstance(self.model_backbone, Transformer):
            if lengthsA is None or lengthsB is None:
                raise ValueError(
                    "lengthsA and lengthsB are required when using Transformer backbone"
                )
            masks = make_masks(
                lengthsA,
                lengthsB,
                proteinA.size(1),
                proteinB.size(1),
                device=proteinA.device,
            )
            output = self.model_backbone(proteinA, proteinB, masks)

        # MLP logic
        elif isinstance(self.model_backbone, MLP):
            output = self.model_backbone(proteinA, proteinB)

        if self.llgp and mode == LLGPMode.INFERENCE:
            logits, var = output
            logits = mean_field_average(logits, var)
        else:
            logits = output

        return logits.reshape(-1)
