from pathlib import Path

import pytorch_lightning as pl
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from tuna.datamodule.datamodule_utils import pad_batch
from tuna.datamodule.embedding_utils import ensure_embeddings


class PPIDataset(Dataset):
    def __init__(
        self, interaction_file_path: Path, embeddings: dict[str, torch.Tensor]
    ):
        self.embeddings = embeddings
        self.protein_to_idx = {name: i for i, name in enumerate(self.embeddings.keys())}
        self.embeddings_list = list(self.embeddings.values())
        self.data = []
        # Read a tsv file with columns: proteinA, proteinB, interaction
        # Not sure if this is best way to do this.
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
    stage: str,
) -> tuple:
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

        if stage == "train":
            pad_len_A = pad_len_B = max_length
        else:
            pad_len_A = max(lensA)
            pad_len_B = max(lensB)

        padded_protAs = pad_batch(protAs, pad_len_A)
        padded_protBs = pad_batch(protBs, pad_len_B)

        lensA_clamped = [min(length, pad_len_A) for length in lensA]
        lensB_clamped = [min(length, pad_len_B) for length in lensB]

        return padded_protAs, padded_protBs, labels, lensA_clamped, lensB_clamped


class PPIDataModule(pl.LightningDataModule):
    def __init__(self, config: DictConfig, embedding_type: str):
        super().__init__()
        self.config = config
        self.max_length = config.datamodule.max_sequence_length
        self.embedding_type = embedding_type
        embeddings_path = getattr(self.config.dataset.paths, "embeddings", None)
        self.embeddings_path = (
            Path(to_absolute_path(embeddings_path)) if embeddings_path else None
        )
        fasta_path = getattr(self.config.dataset.paths, "fasta", None)
        self.fasta_path = Path(to_absolute_path(fasta_path)) if fasta_path else None
        self.train_path = Path(to_absolute_path(self.config.dataset.paths.train))
        self.val_path = Path(to_absolute_path(self.config.dataset.paths.val))
        self.test_path = Path(to_absolute_path(self.config.dataset.paths.test))
        self.batch_size = self.config.datamodule.batch_size

    def setup(self, stage: str):
        self.embeddings = ensure_embeddings(
            embeddings_path=self.embeddings_path,
            fasta_path=self.fasta_path,
            train_path=self.train_path,
            val_path=self.val_path,
            test_path=self.test_path,
            device=self.config.trainer.get("accelerator", "cpu"),
            batch_size=self.batch_size,
        )

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
                batch, self.max_length, self.embedding_type, "train"
            ),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=lambda batch: collate_protein_batch(
                batch, self.max_length, self.embedding_type, "val"
            ),
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=lambda batch: collate_protein_batch(
                batch, self.max_length, self.embedding_type, "test"
            ),
        )
