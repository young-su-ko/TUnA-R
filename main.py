import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import hydra
from omegaconf import DictConfig
import torch
import wandb
from torch.optim.lr_scheduler import StepLR
from torch.optim import Adam

from tuna.pl_modules.lit_mlp import LitMLP
from tuna.pl_modules.lit_tuna import LitTUnA
from tuna.datamodule.ppi_module import PPIDataModule

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    pl.seed_everything(cfg.seed)
    
    wandb_logger = WandbLogger(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        mode=cfg.wandb.mode,
    )
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath="checkpoints",
            filename=f"{cfg.model.type}-{cfg.dataset.name}-{{epoch:02d}}-{{val_loss:.2f}}",
            monitor="val_loss",
            mode="min",
            save_top_k=1
        ),
        LearningRateMonitor(logging_interval="step"),
        EarlyStopping(
            monitor="val_loss",
            patience=cfg.model.training.early_stopping_patience,
            mode="min"
        )
    ]

    if cfg.model.type == "tuna" or cfg.model.type == "t-fc":
        model = LitTUnA(cfg.model)
    elif cfg.model.type == "esm-mlp" or cfg.model.type == "esm-gp":
        model = LitMLP(cfg.model)
    else:
        raise ValueError(f"Model type {cfg.model.type} not supported")

    trainer = pl.Trainer(
        max_epochs=cfg.model.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        logger=wandb_logger,
        callbacks=callbacks,
    )
    
    data_module = PPIDataModule(
        train_file=cfg.dataset.paths.train,
        val_file=cfg.dataset.paths.val,
        test_file=cfg.dataset.paths.test,
        embedding_file=cfg.dataset.paths.embeddings,
        batch_size=cfg.datamodule.batch_size,
        num_workers=cfg.datamodule.num_workers,
        pin_memory=cfg.datamodule.pin_memory,
        persistent_workers=cfg.datamodule.persistent_workers,
        max_sequence_length=cfg.dataset.max_sequence_length,
        class_weights=cfg.dataset.class_weights,
    )
    
    trainer.fit(model, data_module)
    trainer.test(model, data_module)
    
    wandb.finish()

if __name__ == "__main__":
    main()
