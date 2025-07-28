import hydra
import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
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
            filename=f"{cfg.model}-{cfg.dataset}-{{epoch:02d}}-{{val_auroc:.2f}}",
            monitor="val/auroc",
            mode="min",
            save_top_k=1,
        ),
    ]

    lit_module = instantiate(cfg.model)

    trainer = pl.Trainer(
        max_epochs=1,
        limit_train_batches=50,
        limit_val_batches=0,
        profiler="simple",
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
