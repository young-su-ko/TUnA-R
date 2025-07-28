from pathlib import Path

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from tuna.datamodule.datamodule_utils import pad_batch


class PPIDataset(Dataset):
    def __init__(
        self, interaction_file_path: Path, embeddings: dict[str, torch.Tensor]
    ):
        self.embeddings = embeddings
        self.protein_to_idx = {name: i for i, name in enumerate(self.embeddings.keys())}
        self.embeddings_list = list(self.embeddings.values())
        self.data = []
        # Read a tsv file with columns: proteinA, proteinB, interaction
        # Not sure if this is best way to do this..
        # Either way, it will have to be some reading of (proteinA, proteinB, interaction)
        # Then map names to embeddings
        with open(interaction_file_path, "r") as f:
            for line in f:
                proteinA, proteinB, interaction = line.strip().split("\t")
                a_idx = self.protein_to_idx[proteinA]
                b_idx = self.protein_to_idx[proteinB]
                self.data.append(
                    (a_idx, b_idx, torch.tensor(int(interaction), dtype=torch.long))
                )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        a_idx, b_idx, interaction = self.data[idx]
        return self.embeddings_list[a_idx], self.embeddings_list[b_idx], interaction


def collate_protein_batch(
    batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    max_length: int,
    embedding_type: str,
) -> (
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int], list[int]]
):
    """
    Collate function for PPI batches.

    - For embedding_type='protein': just pool and return (A_pooled, B_pooled, labels)
    - For embedding_type='residue': pad proteins and return lengths (for masks to be made later on GPU)
    """
    protAs, protBs, labels = zip(*batch)
    labels = torch.stack(labels)

    if embedding_type == "protein":
        protA_pooled = torch.stack([p.mean(dim=0) for p in protAs])
        protB_pooled = torch.stack([p.mean(dim=0) for p in protBs])
        return protA_pooled, protB_pooled, labels

    else:
        lensA = [p.size(0) for p in protAs]
        lensB = [p.size(0) for p in protBs]

        padded_protAs = pad_batch(protAs, max_length)
        padded_protBs = pad_batch(protBs, max_length)

        return padded_protAs, padded_protBs, labels, lensA, lensB


class PPIDataModule(pl.LightningDataModule):
    def __init__(self, config: DictConfig, embedding_type: str):
        super().__init__()
        self.config = config
        self.max_length = config.datamodule.max_sequence_length
        self.embedding_type = embedding_type
        self.embeddings_path = Path(self.config.dataset.paths.embeddings)
        self.train_path = Path(self.config.dataset.paths.train)
        self.val_path = Path(self.config.dataset.paths.val)
        self.test_path = Path(self.config.dataset.paths.test)
        self.batch_size = self.config.datamodule.batch_size

    def setup(self, stage: str):
        # Load embeddings once and keep in memory
        self.embeddings = torch.load(self.embeddings_path)

        if stage == "fit":
            self.train_dataset = PPIDataset(self.train_path, self.embeddings)
            self.val_dataset = PPIDataset(self.val_path, self.embeddings)
        elif stage == "test":
            self.test_dataset = PPIDataset(self.test_path, self.embeddings)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=lambda batch: collate_protein_batch(
                batch, self.max_length, self.embedding_type
            ),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=lambda batch: collate_protein_batch(
                batch, self.max_length, self.embedding_type
            ),
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,  # Use same batch size as training
            shuffle=False,
            collate_fn=lambda batch: collate_protein_batch(
                batch, self.max_length, self.embedding_type
            ),
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=lambda batch: collate_protein_batch(
                batch, embedding_type=self.embedding_type
            ),
        )
