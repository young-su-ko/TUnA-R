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
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".tsv") as f:
        f.write("protein1\tprotein2\t1\n")
        f.write("protein2\tprotein3\t0\n")
        return f.name


def test_ppi_dataset(sample_embeddings, interaction_file_path):
    """Smoke test for PPIDataset."""
    dataset = PPIDataset(interaction_file_path, sample_embeddings)
    assert len(dataset) == 2

    protA, protB, label = dataset[0]
    assert protA.shape == (5, 640)
    assert protB.shape == (3, 640)
    assert label.item() == 1


def test_collate_protein_batch_residue(sample_embeddings, interaction_file_path):
    """Smoke test for collate_protein_batch with residue embeddings."""
    dataset = PPIDataset(interaction_file_path, sample_embeddings)
    batch = [dataset[0], dataset[1]]

    result = collate_protein_batch(
        batch, max_length=5, embedding_type="residue", stage="train"
    )
    assert len(result) == 5
    protA, protB, labels, lensA, lensB = result
    assert protA.shape == (2, 5, 640)
    assert protB.shape == (2, 5, 640)
    assert labels.shape == (2,)


def test_collate_protein_batch_protein(sample_embeddings, interaction_file_path):
    """Smoke test for collate_protein_batch with protein embeddings."""
    dataset = PPIDataset(interaction_file_path, sample_embeddings)
    batch = [dataset[0], dataset[1]]

    result = collate_protein_batch(
        batch, max_length=5, embedding_type="protein", stage="train"
    )
    assert len(result) == 3
    protA, protB, labels = result
    assert protA.shape == (2, 640)
    assert protB.shape == (2, 640)
    assert labels.shape == (2,)
