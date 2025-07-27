import hydra
import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

import wandb
from tuna.datamodule.ppi_module import PPIDataModule


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    pl.seed_everything(cfg.seed)

    wandb_logger = WandbLogger(project=cfg.wandb.project)

    callbacks = [
        ModelCheckpoint(
            dirpath="checkpoints",
            filename=f"{cfg.model}-{cfg.dataset}-{{epoch:02d}}-{{val_loss:.2f}}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    lit_module = instantiate(cfg.model)

    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        logger=wandb_logger,
        callbacks=callbacks,
    )

    data_module = PPIDataModule(config=cfg, embedding_type=lit_module.embedding_type)

    trainer.fit(lit_module, data_module)
    trainer.test(lit_module, data_module)

    wandb.finish()


if __name__ == "__main__":
    main()
