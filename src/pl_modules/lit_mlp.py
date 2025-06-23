import torch
import pytorch_lightning as pl
from ..models.mlp import MLP
from .base_module import BaseModule
from ..utils import mean_field_average

class LitMLP(BaseModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.model = MLP(*args, **kwargs)
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
        self.log_binary_classification_metrics(y, logit, logit, prefix="val/")
        return loss

    def predict_step(self, batch, batch_idx):
        proteinA, proteinB, y = batch
        last_epoch = self.is_last_epoch()
        if last_epoch:
            logit, var = self(proteinA, proteinB, update_precision=False, get_var=True)
            logit = mean_field_average(logit, var)
        else:
            logit = self(proteinA, proteinB, update_precision=False, get_var=False)

        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit.squeeze(), y.float())
        self.log('test_loss', loss)
        self.log_binary_classification_metrics(y, logit, logit, prefix="test/")