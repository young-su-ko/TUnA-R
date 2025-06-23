import torch
import pytorch_lightning as pl
from ..models.tuna import TUnA
from .base_module import BaseModule

class LitTUnA(BaseModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.model = TUnA(*args, **kwargs)
        self.save_hyperparameters()

    def forward(self, proteinA, proteinB, update_precision=False, get_var=False):
        return self.model(proteinA, proteinB, update_precision=update_precision, get_variance=get_var)

    def training_step(self, batch, batch_idx):
        proteinA, proteinB, y = batch
        last_epoch = self.is_last_epoch()
        logit = self(proteinA, proteinB, update_precision=last_epoch)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit.squeeze(), y.float())
        self.log('train_loss', loss)
        self.log_binary_classification_metrics(y, logit.squeeze(), logit, prefix="train/")
        return loss

    def validation_step(self, batch, batch_idx):
        proteinA, proteinB, y = batch
        logit = self(proteinA, proteinB, update_precision=False, get_var=False)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit.squeeze(), y.float())
        self.log('val_loss', loss)
        self.log_binary_classification_metrics(y, logit.squeeze(), logit, prefix="val/")
        return loss

    def predict_step(self, batch, batch_idx):
        proteinA, proteinB, y = batch
        last_epoch = self.is_last_epoch()
        if last_epoch:
            logit, var = self(proteinA, proteinB, update_precision=False, get_var=True)
            self.log('test_var', var.mean())
            self.log_binary_classification_metrics(y, logit.squeeze(), logit, prefix="test/")
        else:
            logit = self(proteinA, proteinB, update_precision=False, get_var=False)
            self.log_binary_classification_metrics(y, logit.squeeze(), logit, prefix="test/")
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit.squeeze(), y.float())
        self.log('test_loss', loss)
        return loss 