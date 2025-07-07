import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from tuna.pl_modules.lit_tuna import LitTUnA

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    # Hydra automatically instantiates the config class based on _target_
    model_config = hydra.utils.instantiate(cfg.model)
    
    # Create the model
    model = LitTUnA(config=model_config)
    
    # Your training setup here
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        # ... other trainer configs ...
    )
    
    # Train the model
    trainer.fit(model)

if __name__ == "__main__":
    main() 