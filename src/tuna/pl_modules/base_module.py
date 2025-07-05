import pytorch_lightning as pl
import torch
import torchmetrics

class BaseModule(pl.LightningModule):
    # These are helper functions to check the current state of the training,
    # Used for the last layer gaussian process, not used for T-FC and ESM-MLP.
    def _is_last_epoch(self) -> bool:
        # Returns True if this is the last epoch
        return (self.current_epoch == self.trainer.max_epochs - 1)

    def _is_training(self):
        # Returns True if in training step
        return self.training

    def _is_testing(self):
        # Returns True if in test step
        return self.testing

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
