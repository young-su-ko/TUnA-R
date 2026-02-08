import warnings
from pathlib import Path

import torch
from Bio import SeqIO

from tuna.inference.embeddings import ESMEmbedder


def _read_fasta(path: Path) -> dict[str, str]:
    return {record.id: str(record.seq) for record in SeqIO.parse(path, "fasta")}


def _collect_ids(path: Path) -> set[str]:
    """
    Get all the unique protein IDs from the TSV file
    """
    ids: set[str] = set()
    with open(path, "r") as f:
        for line in f:
            columns = line.strip().split("\t")
            if len(columns) < 2:
                warnings.warn(f"Skipping entry due to missing columns: {line.strip()}")
                continue
            proteinA, proteinB = columns
            ids.add(proteinA.strip())
            ids.add(proteinB.strip())
    return ids


def _validate_embeddings(ids: set[str], embeddings: dict[str, torch.Tensor]) -> None:
    """
    We need to check that the provided FASTA file has all the ids in the tsv file in order to generate embeddings.
    """
    missing = [pid for pid in ids if pid not in embeddings]
    if missing:
        preview = ", ".join(missing[:10])
        raise ValueError(f"Missing {len(missing)} embeddings for ids: {preview}")


def ensure_embeddings(
    embeddings_path: Path | None,
    fasta_path: Path | None,
    train_path: Path,
    val_path: Path,
    test_path: Path,
    device: str,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    """
    Ensure that the embeddings are loaded from the given path or generated from the FASTA file if not provided.
    """

    ids = _collect_ids(train_path) | _collect_ids(val_path) | _collect_ids(test_path)

    if embeddings_path is not None and embeddings_path.exists():
        embeddings = torch.load(embeddings_path, map_location="cpu")
        _validate_embeddings(ids, embeddings)
        return embeddings

    if fasta_path is None:
        raise ValueError(
            "No embeddings provided and no FASTA file provided. Either provide embeddings or a FASTA file."
        )
    fasta_dict = _read_fasta(fasta_path)
    if len(fasta_dict) >= 20000:
        warnings.warn(
            "Generating a large number of embeddings (>20000). Saving and loading large embeddings may be slow."
        )
    _validate_embeddings(ids, fasta_dict)

    embedder = ESMEmbedder(device=device)
    embeddings: dict[str, torch.Tensor] = {}

    ids_list = list(ids)
    for start in range(0, len(ids_list), batch_size):
        chunk_ids = ids_list[start : start + batch_size]
        chunk_seqs = [fasta_dict[pid] for pid in chunk_ids]
        chunk_embs = embedder.embed(chunk_seqs)
        for pid, emb in zip(chunk_ids, chunk_embs, strict=True):
            embeddings[pid] = emb.detach().cpu()

    if embeddings_path is not None:
        embeddings_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(embeddings, embeddings_path)

    return embeddings
