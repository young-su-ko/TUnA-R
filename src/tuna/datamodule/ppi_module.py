import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from tuna.datamodule.datamodule_utils import combine_masks, make_masks, pad_batch


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


def collate_protein_batch(
    batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    embedding_type: str = "residue",
) -> (
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    | tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ]
):
    """
    Collate function for PPI batches.
    Args:
        batch: list of (protA, protB, label)
        embedding_type: 'residue' for when you expect [batch, len, dim], 'protein' for when you expect [batch, dim]
    Returns:
        - For 'protein': (protA_pooled, protB_pooled, labels)
        - For 'residue': (protA_padded, protB_padded, labels, masks)
    """
    protAs, protBs, labels = zip(*batch)
    labels = torch.tensor(labels, dtype=torch.long)

    if embedding_type == "protein":
        # Pool over sequence length dimension (dim=0)
        protA_pooled = torch.stack([p.mean(dim=0) for p in protAs])
        protB_pooled = torch.stack([p.mean(dim=0) for p in protBs])
        return protA_pooled, protB_pooled, labels

    else:
        # Get lengths and max length
        lensA = [p.size(0) for p in protAs]
        lensB = [p.size(0) for p in protBs]
        batch_max_len = max(max(lensA), max(lensB))

        # Pad each sequence
        padded_protAs = []
        padded_protBs = []
        for prot in protAs:
            padded = pad_batch(prot, batch_max_len)
            padded_protAs.append(padded)
        for prot in protBs:
            padded = pad_batch(prot, batch_max_len)
            padded_protBs.append(padded)

        # Stack padded sequences
        protA_padded = torch.stack(padded_protAs)
        protB_padded = torch.stack(padded_protBs)

        # Create masks
        maskA = make_masks(lensA, batch_max_len)
        maskB = make_masks(lensB, batch_max_len)
        masks = (
            maskA,
            maskB,
            combine_masks(maskA, maskB),
            combine_masks(maskB, maskA),
        )

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
        self.embeddings = torch.load(self.config.dataset.paths.embeddings)
        if stage == "fit":
            self.train_dataset = PPIDataset(
                self.config.dataset.paths.train,
                self.embeddings,
            )
            self.val_dataset = PPIDataset(
                self.config.dataset.paths.val,
                self.embeddings,
            )
        elif stage == "test":
            self.test_dataset = PPIDataset(
                self.config.dataset.paths.test,
                self.embeddings,
            )
        elif stage == "predict":
            self.predict_dataset = PPIDataset(
                self.config.dataset.paths.predict,
                self.embeddings,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.datamodule.batch_size,
            shuffle=True,
            num_workers=self.config.datamodule.num_workers,
            pin_memory=True,
            collate_fn=lambda batch: collate_protein_batch(
                batch, embedding_type=self.embedding_type
            ),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.datamodule.batch_size,
            shuffle=False,
            num_workers=self.config.datamodule.num_workers,
            pin_memory=True,
            collate_fn=lambda batch: collate_protein_batch(
                batch, embedding_type=self.embedding_type
            ),
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.config.datamodule.num_workers,
            pin_memory=True,
            collate_fn=lambda batch: collate_protein_batch(
                batch, embedding_type=self.embedding_type
            ),
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.config.datamodule.num_workers,
            pin_memory=True,
            collate_fn=lambda batch: collate_protein_batch(
                batch, embedding_type=self.embedding_type
            ),
        )
