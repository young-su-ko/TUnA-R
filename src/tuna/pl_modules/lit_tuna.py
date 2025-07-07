import torch
import pytorch_lightning as pl
from tuna.models._tuna import TUnA
from tuna.pl_modules.base_module import BaseModule
from tuna.models.model_utils import is_llgp, mean_field_average
from tuna.config.model_config import TUnAConfig

class LitTUnA(BaseModule):
    def __init__(self, config: TUnAConfig):
        super().__init__(config)
        self.model = TUnA.from_config(config)
        self.save_hyperparameters()
    
    def forward(self, proteinA: torch.Tensor, proteinB: torch.Tensor, masks: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None, update_precision: bool = False, get_var: bool = False) -> torch.Tensor:
        if is_llgp(self.model):
            return self.model(proteinA, proteinB, update_precision=update_precision, get_variance=get_var)
        return self.model(proteinA, proteinB)

    def training_step(self, batch, batch_idx):
        proteinA, proteinB, y, masks = batch
        last_epoch = self._is_last_epoch()
        if is_llgp(self.model):
            logit = self(proteinA, proteinB, masks, update_precision=last_epoch)
        else:
            logit = self(proteinA, proteinB, masks)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit.squeeze(), y.float())
        self.log('train_loss', loss)
        self._log_binary_classification_metrics(y, logit.squeeze(), logit, prefix="train/")
        return loss

    def validation_step(self, batch, batch_idx):
        proteinA, proteinB, y, masks = batch
        if is_llgp(self.model):
            logit = self(proteinA, proteinB, masks, update_precision=False, get_var=False)
        else:
            logit = self(proteinA, proteinB, masks)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit.squeeze(), y.float())
        self.log('val_loss', loss)
        self._log_binary_classification_metrics(y, logit.squeeze(), logit, prefix="val/")
        return loss

    def test_step(self, batch, batch_idx):
        proteinA, proteinB, y, masks = batch
        if is_llgp(self.model):
            logit, var = self(proteinA, proteinB, masks, update_precision=False, get_var=True)
            logit = mean_field_average(logit, var)
        else:
            logit = self(proteinA, proteinB, masks)

        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit.squeeze(), y.float())
        self.log('test_loss', loss)
        self._log_binary_classification_metrics(y, logit.squeeze(), logit, prefix="test/")
        return loss 

    def predict_step(self, batch, batch_idx):
        proteinA, proteinB, _, masks = batch  # We don't use labels in prediction
        if is_llgp(self.model):
            logit, var = self(proteinA, proteinB, masks, update_precision=False, get_var=True)
            logit = mean_field_average(logit, var)
        else:
            logit = self(proteinA, proteinB, masks)

        return logit