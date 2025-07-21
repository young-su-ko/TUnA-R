import tempfile

import pytest
import torch

from tuna.datamodule.ppi_module import PPIDataset, collate_protein_batch


@pytest.fixture
def sample_embeddings():
    return {
        "protein1": torch.randn(5, 640),
        "protein2": torch.randn(3, 640),
        "protein3": torch.randn(4, 640),
    }


@pytest.fixture
def interaction_file_path():
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        f.write("protein1\tprotein2\t1\n")  # positive interaction
        f.write("protein2\tprotein3\t0\n")  # negative interaction
        return f.name


class TestPPIDataset:
    def test_ppi_dataset(self, sample_embeddings, interaction_file_path):
        dataset = PPIDataset(interaction_file_path, sample_embeddings)

        assert len(dataset) == 2

        item1 = dataset[0]
        assert isinstance(item1, tuple)
        assert len(item1) == 3
        protA, protB, interaction = item1
        assert protA.shape == (5, 640)  # protein1
        assert protB.shape == (3, 640)  # protein2
        assert interaction.item() == 1
        assert interaction.dtype == torch.int64

        item2 = dataset[1]
        protA, protB, interaction = item2
        assert protA.shape == (3, 640)  # protein2
        assert protB.shape == (4, 640)  # protein3
        assert interaction.item() == 0
        assert interaction.dtype == torch.int64


class TestCollateProteinBatch:
    def test_collate_protein_batch_residue(
        self, sample_embeddings, interaction_file_path
    ):
        dataset = PPIDataset(interaction_file_path, sample_embeddings)
        batch = [dataset[0], dataset[1]]

        result = collate_protein_batch(batch, embedding_type="residue")
        assert len(result) == 4  # protA, protB, labels, masks
        protA, protB, labels, masks = result

        max_len = 5  # Longest sequence in the batch
        assert protA.shape == (2, max_len, 640)
        assert protB.shape == (2, max_len, 640)
        assert labels.shape == (2,)
        assert len(masks) == 4  # maskA, maskB, combined1, combined2

        # Check labels
        assert torch.equal(labels, torch.tensor([1, 0], dtype=torch.long))
        assert masks[0].dtype == torch.bool
        assert masks[1].dtype == torch.bool
        assert masks[2].dtype == torch.bool
        assert masks[3].dtype == torch.bool

    def test_collate_protein_batch_protein(
        self, sample_embeddings, interaction_file_path
    ):
        dataset = PPIDataset(interaction_file_path, sample_embeddings)
        batch = [dataset[0], dataset[1]]  # Both items

        result = collate_protein_batch(batch, embedding_type="protein")
        assert len(result) == 3  # protA, protB, labels
        protA, protB, labels = result

        assert protA.shape == (2, 640)  # Pooled to protein level
        assert protB.shape == (2, 640)
        assert labels.shape == (2,)
