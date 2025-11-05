import json
import os

import esm
import torch

from tuna.models._transformer import Transformer
from tuna.models.mask_utils import make_masks


class ESMWrapper:
    def __init__(self, device: torch.device | str):
        self.device = torch.device(device) if isinstance(device, str) else device
        self.esm_model, self.esm_alphabet = esm.pretrained.esm2_t30_150M_UR50D()
        self.batch_converter = self.esm_alphabet.get_batch_converter()

        self.esm_model.to(self.device)
        self.esm_model.eval()

    @torch.no_grad()
    def embed(self, sequences: list[str]) -> list[torch.Tensor]:
        embeddings = []
        for seq in sequences:
            _, _, batch_tokens = self.batch_converter([("sequence", seq)])
            results = self.esm_model(
                batch_tokens.to(self.device), repr_layers=[30], return_contacts=False
            )
            token_representation = results["representations"][30][0, 1:-1, :]
            embeddings.append(token_representation)
        return embeddings  # Returned on GPU/CPU based on self.device


class InferenceWrapper:
    def __init__(self, model, device: torch.device | str):
        self.device = torch.device(device) if isinstance(device, str) else device
        self.model = model
        self.model.to(self.device)
        self.model.eval()

        self.esm = ESMWrapper(self.device)

    @torch.no_grad()
    def predict(self, seqA: str, seqB: str) -> float:
        # 1. Embed sequences → list[Tensor]
        embA, embB = self.esm.embed([seqA, seqB])

        # 2. Convert to batch dims (1, L, D)
        embA = embA.unsqueeze(0)
        embB = embB.unsqueeze(0)

        # 3. Construct masks (just like training)
        lenA = embA.size(1)
        lenB = embB.size(1)
        masks = make_masks([lenA], [lenB], lenA, lenB, device=self.device)

        prob = self.model(embA, embB, masks)  # shape (1,)
        return float(prob.item())

    @torch.no_grad()
    def batch_predict(self, csv_path: str, batch_size: int) -> torch.Tensor: ...

    @classmethod
    def from_pretrained(cls, repo_or_dir: str, device: str, map_location: str):
        if os.path.isdir(repo_or_dir):
            config_path = os.path.join(repo_or_dir, "config.json")
            ckpt_path = os.path.join(repo_or_dir, "pytorch_model.bin")
        else:  # fallback to HF Hub
            from huggingface_hub import hf_hub_download

            config_path = hf_hub_download(repo_or_dir, "config.json")
            ckpt_path = hf_hub_download(repo_or_dir, "pytorch_model.bin")

        with open(config_path) as f:
            model_cfg = json.load(f)

        model = Transformer(**model_cfg)
        model.load_state_dict(torch.load(ckpt_path, map_location=map_location))
        return cls(model, device)
