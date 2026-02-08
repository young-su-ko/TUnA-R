import json
import os

import torch
from omegaconf import DictConfig, OmegaConf


def _clean_model_cfg(model_cfg: DictConfig) -> dict:
    cfg = OmegaConf.to_container(model_cfg, resolve=True)
    if isinstance(cfg, dict):
        target = cfg.pop("_target_", "")
        if "Transformer" in target:
            cfg["backbone"] = "transformer"
        elif "MLP" in target:
            cfg["backbone"] = "mlp"
    return cfg


def _save_backbone(model_backbone, model_cfg: DictConfig, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    cfg = _clean_model_cfg(model_cfg)

    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    torch.save(
        model_backbone.state_dict(),
        os.path.join(output_dir, "pytorch_model.bin"),
    )


def save_backbone_from_checkpoint(
    ckpt_path: str, model_cfg: DictConfig, output_dir: str
) -> None:
    """
    Save model backbone weights from LightningModule for inference.

    Args:
        ckpt_path: Path to the checkpoint file.
        model_cfg: Configuration of the model.
        output_dir: Path to the output directory for the exported model.
    """
    from tuna.pl_modules.lit_ppi import LitPPI

    lit_module = LitPPI.load_from_checkpoint(ckpt_path)
    _save_backbone(lit_module.model.model_backbone, model_cfg, output_dir)
