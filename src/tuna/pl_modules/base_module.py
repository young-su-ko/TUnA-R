import pytorch_lightning as pl
import torch
import torchmetrics
from torch.optim import Adam, lr_scheduler
from torch_optimizer import Lookahead

class BaseModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def _is_last_epoch(self) -> bool:
        return (self.current_epoch == self.trainer.max_epochs - 1)

    def _log_binary_classification_metrics(self, y_true: torch.Tensor, y_pred: torch.Tensor, y_prob: torch.Tensor, prefix: str = ""):
        # y_true: ground truth labels (tensor, 0/1)
        # y_pred: predicted labels (tensor, 0/1)
        # y_prob: predicted probabilities (tensor, float)
        metrics = {
            f"{prefix}accuracy": torchmetrics.functional.accuracy(y_pred, y_true, task="binary"),
            f"{prefix}auroc": torchmetrics.functional.auroc(y_prob, y_true, task="binary"),
            f"{prefix}auprc": torchmetrics.functional.average_precision(y_prob, y_true, task="binary"),
            f"{prefix}precision": torchmetrics.functional.precision(y_pred, y_true, task="binary"),
            f"{prefix}recall": torchmetrics.functional.recall(y_pred, y_true, task="binary"),
            f"{prefix}f1": torchmetrics.functional.f1_score(y_pred, y_true, task="binary"),
            f"{prefix}mcc": torchmetrics.functional.matthews_corrcoef(y_pred, y_true, task="binary"),
        }
        # Specificity = TN / (TN + FP)
        tn = ((y_pred == 0) & (y_true == 0)).sum().float()
        fp = ((y_pred == 1) & (y_true == 0)).sum().float()
        specificity = tn / (tn + fp + 1e-8)
        metrics[f"{prefix}specificity"] = specificity
        for k, v in metrics.items():
            self.log(k, v, prog_bar=True, on_epoch=True, on_step=False)

    def _separate_weights_and_biases(self):
        weight_p, bias_p = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "bias" in name:
                bias_p.append(param)
            else:
                weight_p.append(param)
        return weight_p, bias_p

    def configure_optimizers(self):
        lr = self.config.learning_rate
        weight_decay = self.config.weight_decay

        weight_p, bias_p = self._separate_weights_and_biases()

        optimizer_inner = Adam(
            [
                {'params': weight_p, 'weight_decay': weight_decay},
                {'params': bias_p, 'weight_decay': 0.0}
            ],
            lr=lr
        )

        optimizer = Lookahead(optimizer_inner, alpha=0.8, k=5)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=self.config.step_size, gamma=self.config.gamma)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1
            }
        }