import torch

from tuna.inference.embeddings import (
    EmbeddingStore,
    ESMEmbedder,
    unique_sequences_from_pairs,
)
from tuna.inference.predictor import Predictor


class InferencePipeline:
    """High-level inference pipeline that unifies embedder, store, and predictor."""

    def __init__(
        self,
        predictor: Predictor,
        embedder: ESMEmbedder | None = None,
        store: EmbeddingStore | None = None,
    ):
        self.predictor = predictor
        self.embedder = embedder or ESMEmbedder(predictor.device)
        self.store = store or EmbeddingStore()

    def predict_pair(self, seqA: str, seqB: str) -> tuple[float, float]:
        embA, embB = self.embedder.embed([seqA, seqB])
        return self.predictor.predict_from_embeddings(
            embA, embB
        )  # Returns tuple of score and uncertainty

    @torch.no_grad()
    def predict_pairs(
        self, pairs: list[tuple[str, str]], batch_size: int = 32
    ) -> list[tuple[float, float]]:
        unique = unique_sequences_from_pairs(pairs)
        self.store.ensure(unique, self.embedder, batch_size=batch_size)
        return [
            self.predictor.predict_from_embeddings(
                self.store[a], self.store[b]
            )  # batch size =1
            for a, b in pairs
        ]
