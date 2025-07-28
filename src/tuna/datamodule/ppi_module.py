from pathlib import Path

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from tuna.datamodule.datamodule_utils import (
    combine_masks,
    make_masks,
    pad_batch,
)


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
    embedding_type: str = "residue",
) -> (
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]  # when embedding_type == "protein"
    | tuple[
        torch.Tensor,  # protA_padded
        torch.Tensor,  # protB_padded
        torch.Tensor,  # labels
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],  # masks
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
    labels = torch.stack(labels)

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

        padded_protAs = pad_batch(protAs, batch_max_len)
        padded_protBs = pad_batch(protBs, batch_max_len)

        # Create masks
        maskA = make_masks(lensA, batch_max_len)
        maskB = make_masks(lensB, batch_max_len)
        masks = (
            maskA,
            maskB,
            combine_masks(maskA, maskB),
            combine_masks(maskB, maskA),
        )

        return padded_protAs, padded_protBs, labels, masks


class PPIDataModule(pl.LightningDataModule):
    def __init__(self, config: DictConfig, embedding_type: str):
        super().__init__()
        self.config = config
        self.embedding_type = embedding_type

        self.embeddings_path = Path(self.config.dataset.paths.embeddings)
        self.train_path = Path(self.config.dataset.paths.train)
        self.val_path = Path(self.config.dataset.paths.val)
        self.test_path = Path(self.config.dataset.paths.test)

    def setup(self, stage: str):
        self.embeddings = torch.load(self.embeddings_path)
        if stage == "fit":
            self.train_dataset = PPIDataset(
                self.train_path,
                self.embeddings,
            )
            self.val_dataset = PPIDataset(
                self.val_path,
                self.embeddings,
            )
        elif stage == "test":
            self.test_dataset = PPIDataset(
                self.test_path,
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
