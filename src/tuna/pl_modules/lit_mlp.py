import torch
import pytorch_lightning as pl
from tuna.models._mlp import MLP
from tuna.pl_modules.base_module import BaseModule
from tuna.models.model_utils import is_llgp, mean_field_average
from hydra.utils import instantiate

class LitMLP(BaseModule):
    def __init__(self, config):
        super().__init__(config)
        self.model = instantiate(config.model_cfg)
        self.save_hyperparameters()

    def forward(self, proteinA: torch.Tensor, proteinB: torch.Tensor) -> torch.Tensor:
        return self.model(proteinA, proteinB)

    def training_step(self, batch, batch_idx):
        proteinA, proteinB, y = batch
        if is_llgp(self.model):
            self.model._set_llgp_mode(
                update_precision=self._is_last_epoch(),
                get_variance=False
            )
        logit = self(proteinA, proteinB)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit.squeeze(), y.float())
        self.log('train_loss', loss)
        self._log_binary_classification_metrics(y, logit.squeeze(), logit, prefix="train/")
        return loss
    
    def validation_step(self, batch, batch_idx):
        proteinA, proteinB, y = batch
        if is_llgp(self.model):
            self.model._set_llgp_mode(update_precision=False, get_variance=False)
        logit = self(proteinA, proteinB)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit.squeeze(), y.float())
        self.log('val_loss', loss)
        self._log_binary_classification_metrics(y, logit.squeeze(), logit, prefix="val/")
        return loss

    def test_step(self, batch, batch_idx):
        proteinA, proteinB, y = batch
        if is_llgp(self.model):
            self.model._set_llgp_mode(update_precision=False, get_variance=True)
            logits, var = self(proteinA, proteinB)
            logit = mean_field_average(logits, var)
        else:
            logit = self(proteinA, proteinB)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit.squeeze(), y.float())
        self.log('test_loss', loss)
        self._log_binary_classification_metrics(y, logit.squeeze(), logit, prefix="test/")
        return loss 

    def predict_step(self, batch, batch_idx):
        proteinA, proteinB, _ = batch  # We don't use labels in prediction
        if is_llgp(self.model):
            self.model._set_llgp_mode(update_precision=False, get_variance=True)
            logits, var = self(proteinA, proteinB)
            return mean_field_average(logits, var)
        return self(proteinA, proteinB)