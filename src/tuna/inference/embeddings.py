import esm
import torch


class ESMEmbedder:
    """ESM2 embedder that converts amino-acid sequences to residue embeddings."""

    def __init__(self, device: str | torch.device):
        self.device = torch.device(device) if isinstance(device, str) else device
        self.esm_model, self.esm_alphabet = esm.pretrained.esm2_t30_150M_UR50D()
        self.batch_converter = self.esm_alphabet.get_batch_converter()

        self.esm_model.to(self.device)
        self.esm_model.eval()

    @torch.no_grad()
    def embed(self, sequences: list[str]) -> list[torch.Tensor]:
        labels = [f"seq_{i}" for i in range(len(sequences))]
        batch = list(zip(labels, sequences))
        _, batch_strs, batch_tokens = self.batch_converter(batch)
        results = self.esm_model(
            batch_tokens.to(self.device), repr_layers=[30], return_contacts=False
        )
        reps = results["representations"][
            30
        ]  # Since we are using ESM2-150M, we are using the 30th layer
        embeddings = []
        for i, seq in enumerate(batch_strs):
            seq_len = len(seq)
            embeddings.append(
                reps[i, 1 : seq_len + 1, :]
            )  # Remove the first and last token (start and end of sequence)
        return embeddings


class EmbeddingStore:
    """In-memory storage for sequence embeddings with load/save helpers."""

    def __init__(self, embeddings: dict[str, torch.Tensor] | None = None):
        self._embeddings = embeddings or {}

    def __contains__(self, sequence: str) -> bool:
        return sequence in self._embeddings

    def __getitem__(self, sequence: str) -> torch.Tensor:
        return self._embeddings[sequence]

    @property
    def embeddings(self) -> dict[str, torch.Tensor]:
        return self._embeddings

    @classmethod
    def load(cls, path: str) -> "EmbeddingStore":
        embeddings = torch.load(path, map_location="cpu")
        return cls(embeddings)

    def save(self, path: str) -> None:
        cpu_embeddings = {k: v.detach().cpu() for k, v in self._embeddings.items()}
        torch.save(cpu_embeddings, path)

    @torch.no_grad()
    def ensure(
        self, sequences: list[str], embedder: ESMEmbedder, batch_size: int = 32
    ) -> None:
        missing = [s for s in sequences if s not in self._embeddings]
        if not missing:
            return

        for start in range(0, len(missing), batch_size):
            chunk = missing[start : start + batch_size]
            chunk_embs = embedder.embed(chunk)
            for seq, emb in zip(chunk, chunk_embs, strict=True):
                self._embeddings[seq] = emb.detach().cpu()


def unique_sequences_from_pairs(pairs: list[tuple[str, str]]) -> list[str]:
    return list(dict.fromkeys(seq for pair in pairs for seq in pair))
