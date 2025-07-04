import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import hydra
from omegaconf import DictConfig
import torch
import wandb

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
    
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="tuna-{epoch:02d}-{val_loss:.2f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, lr_monitor],
    )
    
    model = get_model(cfg.model)
    data_module = PPIDataModule(cfg)
    
    trainer.fit(model, data_module)
    
    wandb.finish()


if __name__ == "__main__":
    main()
