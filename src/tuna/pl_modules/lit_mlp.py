import torch
import torch.nn as nn
from hydra.utils import instantiate

from tuna.models.model_utils import is_llgp, mean_field_average
from tuna.pl_modules.base_module import BaseModule
from tuna.pl_modules.llgp_utils import LLGPMode, set_llgp_mode


class LitMLP(BaseModule):
    def __init__(self, config):
        super().__init__(config)
        self.model = instantiate(config.model)
        self.criterion = nn.BCEWithLogitsLoss()
        self.save_hyperparameters()

    def _get_raw_output(
        self, proteinA: torch.Tensor, proteinB: torch.Tensor, mode: LLGPMode
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if is_llgp(self.model):
            set_llgp_mode(self.model, mode, is_last_epoch=self._is_last_epoch())
        return self.model(proteinA, proteinB)

    def _get_logits(
        self, proteinA: torch.Tensor, proteinB: torch.Tensor, mode: LLGPMode
    ) -> torch.Tensor:
        output = self._get_raw_output(proteinA, proteinB, mode)

        if is_llgp(self.model) and mode == LLGPMode.INFERENCE:
            logits, var = output
            return mean_field_average(logits, var)
        return output

    def forward(self, proteinA: torch.Tensor, proteinB: torch.Tensor) -> torch.Tensor:
        logits = self._get_logits(proteinA, proteinB, mode=LLGPMode.INFERENCE)
        return torch.sigmoid(logits)

    def training_step(self, batch, batch_idx):
        proteinA, proteinB, y = batch
        logit = self._get_logits(proteinA, proteinB, mode=LLGPMode.TRAINING)
        loss = self.criterion(logit.squeeze(), y.float())
        self.log("train_loss", loss)
        self._log_binary_classification_metrics(
            y, logit.squeeze(), logit, prefix="train/"
        )
        return loss

    def validation_step(self, batch, batch_idx):
        proteinA, proteinB, y = batch
        logit = self._get_logits(proteinA, proteinB, mode=LLGPMode.VALIDATION)
        loss = self.criterion(logit.squeeze(), y.float())
        self.log("val_loss", loss)
        self._log_binary_classification_metrics(
            y, logit.squeeze(), logit, prefix="val/"
        )
        return loss

    def test_step(self, batch, batch_idx):
        proteinA, proteinB, y = batch
        logit = self._get_logits(proteinA, proteinB, mode=LLGPMode.INFERENCE)
        loss = self.criterion(logit.squeeze(), y.float())
        self.log("test_loss", loss)
        self._log_binary_classification_metrics(
            y, logit.squeeze(), logit, prefix="test/"
        )
        return loss
