import pytorch_lightning as pl
import torch
import torchmetrics
from pytorch_optimizer import Lookahead
from torch.optim import Adam, lr_scheduler

from tuna.pl_modules.llgp_utils import LLGPMode


class BaseModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.train_metrics = self._init_metrics(prefix="train/")
        self.val_metrics = self._init_metrics(prefix="val/")
        self.test_metrics = self._init_metrics(prefix="test/")

    def _init_metrics(self, prefix: str):
        return torchmetrics.MetricCollection(
            {
                "accuracy": torchmetrics.Accuracy(task="binary"),
                "auroc": torchmetrics.AUROC(task="binary"),
                "auprc": torchmetrics.AveragePrecision(task="binary"),
                "precision": torchmetrics.Precision(task="binary"),
                "recall": torchmetrics.Recall(task="binary"),
                "f1": torchmetrics.F1Score(task="binary"),
                "mcc": torchmetrics.MatthewsCorrCoef(task="binary"),
            },
            prefix=prefix,
        )

    def _is_last_epoch(self) -> bool:
        return self.current_epoch == self.trainer.max_epochs - 1

    def _process_logits(
        self, logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        return probs, preds

    def _update_metrics(self, y_true, y_pred, y_prob, stage: str):
        if stage == "train":
            self.train_metrics.update(y_prob, y_true)
        elif stage == "val":
            self.val_metrics.update(y_prob, y_true)
        elif stage == "test":
            self.test_metrics.update(y_prob, y_true)

    def _log_epoch_metrics(self, stage: str):
        metrics = {
            "train": self.train_metrics,
            "val": self.val_metrics,
            "test": self.test_metrics,
        }[stage]

        computed = metrics.compute()
        self.log_dict(computed, prog_bar=True, on_epoch=True, on_step=False)
        metrics.reset()

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
                {"params": weight_p, "weight_decay": weight_decay},
                {"params": bias_p, "weight_decay": 0.0},
            ],
            lr=lr,
        )

        optimizer = Lookahead(optimizer_inner, alpha=0.8, k=5)
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=self.config.step_size, gamma=self.config.gamma
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
