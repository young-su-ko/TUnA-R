import os
import uuid

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
    run_id = str(uuid.uuid4())[:8]
    wandb_logger = WandbLogger(project=cfg.wandb.project, name=run_id)

    checkpoint_dir = f"checkpoints/{run_id}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="{epoch:02d}-{auroc:.4f}",
            monitor="val/auroc",
            mode="max",
            save_top_k=1,
        ),
    ]

    lit_module = instantiate(cfg.model)

    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=callbacks,
        **cfg.trainer,
    )

    data_module = PPIDataModule(
        config=cfg, embedding_type=lit_module.model_backbone.embedding_type
    )

    trainer.fit(lit_module, data_module)
    trainer.test(lit_module, data_module)

    wandb.finish()


if __name__ == "__main__":
    main()
