import torch
from hydra.utils import instantiate

from tuna.models.model_utils import is_llgp, mean_field_average
from tuna.pl_modules.base_module import BaseModule


class LitTransformer(BaseModule):
    def __init__(self, config):
        super().__init__(config)
        self.model = instantiate(config.model_cfg)
        self.save_hyperparameters()

    def forward(
        self,
        proteinA: torch.Tensor,
        proteinB: torch.Tensor,
        masks: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None,
    ) -> torch.Tensor:
        return self.model(proteinA, proteinB, masks)

    def training_step(self, batch, batch_idx):
        proteinA, proteinB, y, masks = batch
        if is_llgp(self.model):
            self.model._set_llgp_mode(
                update_precision=self._is_last_epoch(), get_variance=False
            )
        logit = self(proteinA, proteinB, masks)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logit.squeeze(), y.float()
        )
        self.log("train_loss", loss)
        self._log_binary_classification_metrics(
            y, logit.squeeze(), logit, prefix="train/"
        )
        return loss

    def validation_step(self, batch, batch_idx):
        proteinA, proteinB, y, masks = batch
        if is_llgp(self.model):
            self.model._set_llgp_mode(update_precision=False, get_variance=False)
        logit = self(proteinA, proteinB, masks)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logit.squeeze(), y.float()
        )
        self.log("val_loss", loss)
        self._log_binary_classification_metrics(
            y, logit.squeeze(), logit, prefix="val/"
        )
        return loss

    def test_step(self, batch, batch_idx):
        proteinA, proteinB, y, masks = batch
        if is_llgp(self.model):
            self.model._set_llgp_mode(
                update_precision=False,
                get_variance=True,
            )
            logit, var = self(proteinA, proteinB, masks)
            logit = mean_field_average(logit, var)
        else:
            logit = self(proteinA, proteinB, masks)

        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logit.squeeze(), y.float()
        )
        self.log("test_loss", loss)
        self._log_binary_classification_metrics(
            y, logit.squeeze(), logit, prefix="test/"
        )
        return loss

    def predict_step(self, batch, batch_idx):
        proteinA, proteinB, _, masks = batch  # We don't use labels in prediction
        if is_llgp(self.model):
            self.model._set_llgp_mode(update_precision=False, get_variance=True)
            logit, var = self(proteinA, proteinB, masks)
            logit = mean_field_average(logit, var)
            return logit
        else:
            return self(proteinA, proteinB, masks)
