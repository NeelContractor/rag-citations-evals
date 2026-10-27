from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path


def late_interaction_score(
    query_vectors: Sequence[Sequence[float]],
    document_vectors: Sequence[Sequence[float]],
) -> float:
    if not query_vectors or not document_vectors:
        return 0.0

    total = 0.0
    for query_vector in query_vectors:
        best = max(
            sum(q * d for q, d in zip(query_vector, document_vector, strict=False))
            for document_vector in document_vectors
        )
        total += best
    return total


class LateInteractionReranker:
    def __init__(self, model_name: str, cache_dir: Path) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self._model = None

    def score(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        model = self._get_model()
        query_vectors = next(iter(model.query_embed([query]))).tolist()
        document_vectors = [vectors.tolist() for vectors in model.passage_embed(documents)]
        return [
            late_interaction_score(query_vectors, document_vector)
            for document_vector in document_vectors
        ]

    def _get_model(self):
        if self._model is None:
            from fastembed import LateInteractionTextEmbedding

            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._model = LateInteractionTextEmbedding(
                model_name=self.model_name,
                cache_dir=str(self.cache_dir),
            )
        return self._model
