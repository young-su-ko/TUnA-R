import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import hydra
from omegaconf import DictConfig
import wandb
from hydra.utils import instantiate
from tuna.datamodule.ppi_module import PPIDataModule

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    pl.seed_everything(cfg.seed)
    
    wandb_logger = WandbLogger(
        project=cfg.wandb.project
    )
    
    callbacks = [
        ModelCheckpoint(
            dirpath="checkpoints",
            filename=f"{cfg.model}-{cfg.dataset}-{{epoch:02d}}-{{val_loss:.2f}}",
            monitor="val_loss",
            mode="min",
            save_top_k=1
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    lit_module = instantiate(cfg.module)
    
    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        learning_rate=cfg.trainer.learning_rate,
        weight_decay=cfg.trainer.weight_decay,
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
    
    trainer.fit(lit_module, data_module)
    trainer.test(lit_module, data_module)
    
    wandb.finish()

if __name__ == "__main__":
    main()
