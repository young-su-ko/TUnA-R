import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import torch
from tuna.datamodule.datamodule_utils import make_masks, pad_batch, combine

# TODO: add docstrings in numpy format

class PPIDataset(Dataset):
    def __init__(self, interaction_file: str, embeddings: dict[str, torch.Tensor]):
        self.embeddings = embeddings

        self.data = []
        with open(interaction_file, "r") as f:
            for line in f:
                proteinA, proteinB, interaction = line.strip().split("\t")
                self.data.append((proteinA, proteinB, interaction))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        proteinA, proteinB, interaction = self.data[idx]
        proteinA_embedding = self.embeddings[proteinA]
        proteinB_embedding = self.embeddings[proteinB]

        return proteinA_embedding, proteinB_embedding, interaction

def collate_protein_batch(batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]], device: torch.device, embedding_type: str = "residue") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Collate function for PPI batches.
    Args:
        batch: list of (protA, protB, label)
        device: torch device
        embedding_type: 'residue' for when you expect [batch, len, dim], 'protein' for when you expect [batch, dim]
    Returns:
        - For 'protein': (protA_pooled, protB_pooled, labels)
        - For 'residue': (protA_padded, protB_padded, labels, masks)
    """
    protAs, protBs, labels = zip(*batch)
    labels = torch.tensor(labels, dtype=torch.long, device=device)

    if embedding_type == "protein":
        # Pool over sequence length dimension (dim=0)
        protA_pooled = torch.stack([p.mean(dim=0) for p in protAs])
        protB_pooled = torch.stack([p.mean(dim=0) for p in protBs])
        return protA_pooled, protB_pooled, labels
    
    else:
        lensA = [p.size(0) for p in protAs]
        lensB = [p.size(0) for p in protBs]
        batch_max_len = max(max(lensA), max(lensB))

        protA_padded = pad_batch(protAs, batch_max_len, device)
        protB_padded = pad_batch(protBs, batch_max_len, device)

        maskA = make_masks(lensA, batch_max_len, device)
        maskB = make_masks(lensB, batch_max_len, device)
        masks = (maskA, maskB, combine_masks(maskA, maskB, device), combine_masks(maskB, maskA, device))

        return protA_padded, protB_padded, labels, masks

class PPIDataModule(pl.LightningDataModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        if self.config.model == "tuna" or self.config.model == "tfc":
            self.embedding_type = "residue"
        elif self.config.model == "esm_gp" or self.config.model == "esm_mlp":
            self.embedding_type = "protein"

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = PPIDataset(
                self.config.data.train_file,
                self.config.data.train_embeddings_dir,
            )
            self.val_dataset = PPIDataset(
                self.config.data.val_file,
                self.config.data.train_embeddings_dir,
            )
        elif stage == "test":
            self.test_dataset = PPIDataset(
                self.config.data.test_file,
                self.config.data.train_embeddings_dir,
            )
        elif stage == "predict":
            self.predict_dataset = PPIDataset(
                self.config.data.predict_file,
                self.config.data.predict_embeddings_dir,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=self.config.trainer.num_workers,
            pin_memory=True,
            collate_fn=lambda batch: collate_protein_batch(batch, self.config.trainer.device, self.embedding_type),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.config.trainer.num_workers,
            pin_memory=True,
            collate_fn=lambda batch: collate_protein_batch(batch, self.config.trainer.device, self.embedding_type),
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.config.trainer.num_workers,
            pin_memory=True,
            collate_fn=lambda batch: collate_protein_batch(batch, self.config.trainer.device, self.embedding_type),
        )
    
    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=1
            shuffle=False,
            num_workers=self.config.trainer.num_workers,
            pin_memory=True,
            collate_fn=lambda batch: collate_protein_batch(batch, self.config.trainer.device, self.embedding_type),
        )