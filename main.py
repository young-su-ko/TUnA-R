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

def get_model(cfg: DictConfig):
    """
    Initialize the appropriate model based on configuration.
    
    Available models:
    - TUnA: Transformer-based model for protein-protein interaction
    - TUnA-GP: TUnA with last layer Gaussian Process
    - MLP: Multi-layer perceptron baseline
    - MLP-GP: MLP with last layer Gaussian Process
    """
    model_type = cfg.model.type.lower()
    arch_cfg = cfg.model.architecture
    
    if model_type in ["tuna", "tuna-gp"]:
        model = LitTUnA(
            protein_dim=arch_cfg.protein_dim,
            hid_dim=arch_cfg.hid_dim,
            dropout=arch_cfg.dropout,
            n_layers=arch_cfg.n_layers,
            n_heads=arch_cfg.n_heads,
            ff_dim=arch_cfg.ff_dim,
            llgp=model_type == "tuna-gp",
            spectral_norm=arch_cfg.spectral_norm,
            rff_features=arch_cfg.get("rff_features", 4096),
            gp_cov_momentum=arch_cfg.get("gp_cov_momentum", -1),
            gp_ridge_penalty=arch_cfg.get("gp_ridge_penalty", 1),
        )
    elif model_type in ["mlp", "mlp-gp"]:
        model = LitMLP(
            protein_dim=arch_cfg.protein_dim,
            hid_dim=arch_cfg.hid_dim,
            dropout=arch_cfg.dropout,
            llgp=model_type == "mlp-gp",
            spectral_norm=arch_cfg.spectral_norm,
            rff_features=arch_cfg.get("rff_features", 4096),
            gp_cov_momentum=arch_cfg.get("gp_cov_momentum", -1),
            gp_ridge_penalty=arch_cfg.get("gp_ridge_penalty", 1),
        )
    else:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            "Available options: tuna, tuna-gp, mlp, mlp-gp"
        )
    
    # Set up optimizer and scheduler
    optimizer = get_optimizer(model, cfg)
    scheduler = get_scheduler(optimizer, cfg)
    model.configure_optimizers = lambda: {"optimizer": optimizer, "lr_scheduler": scheduler}
    
    return model

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

    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        logger=wandb_logger,
        callbacks=callbacks,
        gradient_clip_val=cfg.model.training.gradient_clip_val,
        accumulate_grad_batches=cfg.model.training.accumulate_grad_batches,
    )
    
    model = get_model(cfg)
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
