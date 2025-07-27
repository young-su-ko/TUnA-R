import torch
import torch.nn as nn
from omegaconf import OmegaConf

from tuna.models._transformer import Transformer
from tuna.models.model_utils import is_llgp, mean_field_average
from tuna.pl_modules.base_module import BaseModule
from tuna.pl_modules.llgp_utils import LLGPMode, set_llgp_mode


class LitTransformer(BaseModule):
    embedding_type: str = "residue"

    def __init__(self, model_config: dict, train_config: dict):
        super().__init__(config=OmegaConf.create(train_config))
        self.model = Transformer(**model_config)
        self.criterion = nn.BCEWithLogitsLoss()
        self.save_hyperparameters()

    def _get_raw_output(
        self,
        proteinA: torch.Tensor,
        proteinB: torch.Tensor,
        masks: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        mode: LLGPMode,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if is_llgp(self.model):
            set_llgp_mode(self.model, mode, is_last_epoch=self._is_last_epoch())
        return self.model(proteinA, proteinB, masks)

    def _get_logits(
        self,
        proteinA: torch.Tensor,
        proteinB: torch.Tensor,
        masks: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        mode: LLGPMode,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        output = self._get_raw_output(proteinA, proteinB, masks, mode)
        if is_llgp(self.model) and mode == LLGPMode.INFERENCE:
            logits, var = output
            return mean_field_average(logits, var)
        return output

    def _process_logits(
        self, logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        probs = torch.sigmoid(logits).squeeze(-1)
        preds = (probs > 0.5).long()
        return probs, preds

    def forward(
        self,
        proteinA: torch.Tensor,
        proteinB: torch.Tensor,
        masks: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        logits = self._get_logits(proteinA, proteinB, masks, mode=LLGPMode.INFERENCE)
        return torch.sigmoid(logits)

    def _shared_step(self, batch, mode: LLGPMode, prefix: str):
        proteinA, proteinB, y, masks = batch
        logits = self._get_logits(proteinA, proteinB, masks, mode)
        probs, preds = self._process_logits(logits)
        loss = self.criterion(logits.squeeze(-1), y.float())
        self.log(f"{prefix}/loss", loss)
        self._log_binary_classification_metrics(y, preds, probs, prefix=f"{prefix}/")
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, LLGPMode.TRAINING, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, LLGPMode.VALIDATION, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, LLGPMode.INFERENCE, "test")
