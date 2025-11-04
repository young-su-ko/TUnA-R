import torch.nn as nn
from omegaconf import OmegaConf

from tuna.models._mlp import MLP
from tuna.models._transformer import Transformer
from tuna.models.ppi_predictor import PPIPredictor
from tuna.pl_modules.base_module import BaseModule
from tuna.pl_modules.llgp_utils import LLGPMode


class LitPPI(BaseModule):
    def __init__(self, model_backbone: Transformer | MLP, optimizer_config: dict):
        super().__init__(config=OmegaConf.create(optimizer_config))
        self.model = PPIPredictor(model_backbone=model_backbone)
        self._initialize_weights()
        self.criterion = nn.BCEWithLogitsLoss()
        self.save_hyperparameters()

    def on_fit_start(self):
        if isinstance(self.model.model_backbone, Transformer) and hasattr(
            self.model.model_backbone, "mask_maker"
        ):
            self.model.mask_maker.device = self.device

    def _initialize_weights(self):
        for p in self.model.model_backbone.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _shared_step(self, batch, mode: LLGPMode, prefix: str):
        # - protein embeddings: (proteinA, proteinB, y) - 3 items
        # - residue embeddings: (proteinA, proteinB, y, proteinA_lens, proteinB_lens) - 5 items
        if len(batch) == 3:
            # Protein-level embeddings (MLP case)
            proteinA, proteinB, y = batch
            proteinA_lens = None
            proteinB_lens = None
        elif len(batch) == 5:
            # Residue-level embeddings (Transformer case)
            proteinA, proteinB, y, proteinA_lens, proteinB_lens = batch
        else:
            raise ValueError(f"Unexpected batch format with {len(batch)} items")

        logits = self.model(
            proteinA,
            proteinB,
            proteinA_lens,
            proteinB_lens,
            mode,
            is_last_epoch=self._is_last_epoch(),
        )

        probs, preds = self._process_logits(logits)
        loss = self.criterion(logits, y.float())

        self.log(
            f"{prefix}/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(proteinA),
        )

        self._update_metrics(y, preds, probs, stage=prefix)

        return loss
