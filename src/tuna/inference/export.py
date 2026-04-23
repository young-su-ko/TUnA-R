import json
import os

import torch
from hydra.utils import instantiate
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
    model_backbone.eval()

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
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"]

    model_backbone = instantiate(model_cfg)
    backbone_state_dict = {}
    prefix = "model.model_backbone."
    for key, value in state_dict.items():
        if key.startswith(prefix):
            backbone_state_dict[key.removeprefix(prefix)] = value

    if not backbone_state_dict:
        raise ValueError(
            f"No backbone weights with prefix '{prefix}' were found in checkpoint: {ckpt_path}"
        )

    model_backbone.load_state_dict(backbone_state_dict, strict=True)
    _save_backbone(model_backbone, model_cfg, output_dir)
