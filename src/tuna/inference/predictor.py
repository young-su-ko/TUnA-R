import json
import os

import torch

from tuna.models._mlp import MLP
from tuna.models._transformer import Transformer
from tuna.models.llgp_utils import LLGPMode
from tuna.models.ppi_predictor import PPIPredictor

PRETRAINED_MODELS = {
    "tuna": "yk0/tuna-r-tuna",
    "tfc": "yk0/tuna-r-tfc",
    "esm_mlp": "yk0/tuna-r-esm_mlp",
    "esm_gp": "yk0/tuna-r-esm_gp",
}


def resolve_model_id(repo_or_dir: str) -> str:
    # If local path, return as is
    if os.path.isdir(repo_or_dir):
        return repo_or_dir
    return PRETRAINED_MODELS.get(repo_or_dir, repo_or_dir)


class Predictor:
    """Inference-only wrapper around PPIPredictor."""

    def __init__(self, model_backbone, device: str | torch.device):
        self.device = torch.device(device) if isinstance(device, str) else device
        self.model = PPIPredictor(
            model_backbone=model_backbone
        )  # Predictor returns logits, mean-field already applied if LLGP
        self.model.to(self.device)
        self.model.eval()

    def predict_from_embeddings(self, embA: torch.Tensor, embB: torch.Tensor) -> float:
        if embA.device != self.device:
            embA = embA.to(self.device)
        if embB.device != self.device:
            embB = embB.to(self.device)

        if isinstance(self.model.model_backbone, Transformer):
            # Use length by dim representation for Transformer model
            embA = embA.unsqueeze(0)
            embB = embB.unsqueeze(0)
            lenA = embA.size(1)
            lenB = embB.size(1)
            logits = self.model(
                proteinA=embA,
                proteinB=embB,
                mode=LLGPMode.INFERENCE,
                lengthsA=[lenA],
                lengthsB=[lenB],
                is_last_epoch=False,
            )
        elif isinstance(self.model.model_backbone, MLP):
            # Used mean-pooled embeddings for MLP model
            pooledA = embA.mean(dim=0).unsqueeze(0)
            pooledB = embB.mean(dim=0).unsqueeze(0)
            logits = self.model(
                proteinA=pooledA,
                proteinB=pooledB,
                mode=LLGPMode.INFERENCE,
                lengthsA=None,
                lengthsB=None,
                is_last_epoch=False,
            )
        else:
            raise TypeError("Unsupported model backbone for inference.")

        prob = torch.sigmoid(logits)
        uncertainty = (1 - prob) * prob / 0.25  # Defined by Liu et al. 2020
        return float(prob.item()), float(uncertainty.item())

    @classmethod
    def from_pretrained(cls, repo_or_dir: str, device: str | torch.device):
        repo_or_dir = resolve_model_id(repo_or_dir)
        if os.path.isdir(repo_or_dir):
            config_path = os.path.join(repo_or_dir, "config.json")
            ckpt_path = os.path.join(repo_or_dir, "pytorch_model.bin")
        else:
            from huggingface_hub import hf_hub_download

            config_path = hf_hub_download(repo_or_dir, "config.json")
            ckpt_path = hf_hub_download(repo_or_dir, "pytorch_model.bin")

        with open(config_path) as f:
            model_cfg = json.load(f)

        state_dict = torch.load(ckpt_path, map_location=device)

        backbone = model_cfg.pop("backbone", None)
        if backbone is None:
            raise ValueError(
                "config.json missing 'backbone'. Please specify either 'transformer' or 'mlp' in the config."
            )

        if isinstance(model_cfg.get("gp_config"), dict):
            from omegaconf import OmegaConf

            model_cfg["gp_config"] = OmegaConf.create(model_cfg["gp_config"])

        if backbone == "transformer":
            model = Transformer(**model_cfg)
        elif backbone == "mlp":
            model = MLP(**model_cfg)
        else:
            raise ValueError(f"Unknown backbone type: {backbone}")

        model.load_state_dict(state_dict)
        return cls(model, device)
