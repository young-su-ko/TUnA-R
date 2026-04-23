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
from tuna.inference.export import save_backbone_from_checkpoint


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    pl.seed_everything(cfg.seed)
    run_id = str(uuid.uuid4())[:8]
    wandb_logger = WandbLogger(project=cfg.wandb.project, name=run_id)

    checkpoint_dir = f"checkpoints/{run_id}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    lit_module = instantiate(cfg.model, _recursive_=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="last.ckpt",
        save_last=True,
    )

    trainer_kwargs = dict(cfg.trainer)
    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        **trainer_kwargs,
    )

    data_module = PPIDataModule(
        config=cfg, embedding_type=lit_module.model.model_backbone.embedding_type
    )

    trainer.fit(lit_module, data_module)
    trainer.test(model=lit_module, datamodule=data_module)

    if cfg.save_model.save:
        save_dir = os.path.join(cfg.save_model.save_dir, run_id)
        save_backbone_from_checkpoint(
            checkpoint_callback.last_model_path, cfg.model.model_backbone, save_dir
        )

    wandb.finish()


if __name__ == "__main__":
    main()
