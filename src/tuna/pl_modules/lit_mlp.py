import torch
import torch.nn as nn
from omegaconf import OmegaConf

from tuna.models._mlp import MLP
from tuna.models.model_utils import is_llgp, mean_field_average
from tuna.pl_modules.base_module import BaseModule
from tuna.pl_modules.llgp_utils import LLGPMode, set_llgp_mode


class LitMLP(BaseModule):
    embedding_type: str = "protein"

    def __init__(self, model_config: dict, train_config: dict):
        super().__init__(config=OmegaConf.create(train_config))
        self.model = MLP(**model_config)
        self._initialize_weights()
        self.criterion = nn.BCEWithLogitsLoss()
        self.save_hyperparameters()

    def _initialize_weights(self):
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

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
        return torch.sigmoid(logits).squeeze()

    def _shared_step(self, batch, mode: LLGPMode, prefix: str):
        proteinA, proteinB, y = batch
        logit = self._get_logits(proteinA, proteinB, mode)
        probs, preds = self._process_logits(logit)
        loss = self.criterion(logit.squeeze(), y.float())

        self.log(
            f"{prefix}/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        self._update_metrics(y, preds, probs, stage=prefix)

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, LLGPMode.TRAINING, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, LLGPMode.VALIDATION, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, LLGPMode.INFERENCE, "test")

    def on_train_epoch_end(self):
        self._log_epoch_metrics("train")

    def on_validation_epoch_end(self):
        self._log_epoch_metrics("val")

    def on_test_epoch_end(self):
        self._log_epoch_metrics("test")
